"""
denoise_mos2_stem.py
====================
Automated Bragg-filter denoising pipeline for MoS2 HAADF-STEM images
with heavy surface contamination.

Pipeline (applied to every open 2-D image in the workspace):
  1. FFT  – fast Fourier transform (numpy)
  2. Bragg filter  – detect hexagonal MoS2 diffraction spots, build
                     Gaussian soft-mask, zero out amorphous background
  3. IFFT – reconstruct filtered real-space image
  4. Gaussian blur – suppress residual ringing artifacts
  5. Gamma correction – power-law contrast enhancement (gamma < 1)
  6. Output – new DM image with original pixel calibration preserved

Requirements:
  - Gatan DigitalMicrograph GMS 3+ (embedded Python + numpy + scipy)
  - Run via Script Editor  (Script > Open Script … > F5)

Author: auto-generated for MoS2-STEM-CIF repository
"""

# ============================================================
# SECTION 1 – IMPORTS & GMS COMPATIBILITY GUARD
# ============================================================

import math
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

try:
    import DigitalMicrograph as DM
except ImportError:
    raise ImportError(
        "This script must be executed inside Gatan DigitalMicrograph (GMS 3+). "
        "It cannot be run as a standalone Python script."
    )


# ============================================================
# SECTION 2 – TUNABLE PARAMETERS
#   Adjust these values at the top without touching the body.
# ============================================================

# --- Post-IFFT Gaussian smoothing ---
# Sigma (pixels).  1–2 px is appropriate for STEM data.
# Larger values produce smoother but slightly blurred columns.
GAUSSIAN_RADIUS = 1.2

# --- Gamma correction ---
# Output = (normalized_input) ** GAMMA_VALUE
# Values < 1 brighten dim atoms; typical range 0.3 – 0.7.
GAMMA_VALUE = 0.5

# --- Bragg-spot mask ---
# Gaussian soft-mask half-width per detected spot (pixels).
# Too small → loss of atomic detail; too large → contamination leaks back.
MASK_RADIUS = 8

# --- Peak detection ---
# Percentile of FFT amplitude (inside the annular search band) that must
# be exceeded for a pixel to be called a Bragg-spot candidate.
# Raise toward 99.9 if too many spurious spots are found.
# Lower toward 97 if known MoS2 spots are missed.
SPOT_THRESHOLD_PERCENTILE = 99.0

# --- Annular search band (pixels from DC / image centre) ---
# Spots closer than MIN are low-freq drift/background; spots farther than
# MAX are beyond physically meaningful d-spacings.
#
# Rough conversion:  fft_px = image_size_px / (d_angstrom / pixel_angstrom)
# Example: 512-px image, 0.05 Å/px, d=2.73 Å → fft_px ≈ 9 px
# Adjust for your actual image size and calibration.
MIN_SPOT_DISTANCE = 15   # px  (excludes DC halo and long-range drift)
MAX_SPOT_DISTANCE = 220  # px  (excludes aliased high-freq corners)

# --- 6-fold symmetry enforcement ---
# When True, every detected peak is rotated by k×60° (k=1..5) and all
# 6 symmetry partners are also masked.  This guarantees correct hexagonal
# coverage even when individual spots are weak or contamination-obscured.
ENFORCE_6FOLD_SYMMETRY = True

# --- Local-maximum suppression radius (pixels) ---
# Prevents double-counting the same Bragg spot.
LOCAL_MAX_NEIGHBORHOOD = 10


# ============================================================
# SECTION 3 – HELPER FUNCTIONS
# ============================================================

def compute_fft(image_array):
    """
    Compute the 2-D FFT of a real image.

    Parameters
    ----------
    image_array : np.ndarray, shape (H, W), dtype float64

    Returns
    -------
    fft_raw : np.ndarray, complex128   – unshifted FFT (for IFFT path)
    amp_shifted : np.ndarray, float64  – |FFT| with DC at centre (for peak detection)
    cy, cx : int                       – pixel coords of DC component
    """
    fft_raw = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_raw)
    amp_shifted = np.abs(fft_shifted)
    H, W = image_array.shape
    cy, cx = H // 2, W // 2
    return fft_raw, amp_shifted, cy, cx


def find_bragg_peaks(amp_shifted, cy, cx):
    """
    Detect MoS2 Bragg spots in the shifted FFT amplitude image.

    Strategy
    --------
    1. Build distance map from DC; restrict search to annular band.
    2. Threshold at the given percentile of amplitude inside that band.
    3. Apply local-maximum suppression to find discrete peaks.
    4. Optionally rotate each peak by 60° increments (6-fold symmetry).
    5. Deduplicate.

    Returns
    -------
    List of (row, col) integer tuples in the shifted frame.
    Raises ValueError if no peaks are found.
    """
    H, W = amp_shifted.shape
    rows, cols = np.mgrid[0:H, 0:W]

    # Distance from DC
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)

    # Annular validity mask
    annular = (dist >= MIN_SPOT_DISTANCE) & (dist <= MAX_SPOT_DISTANCE)

    if not np.any(annular):
        raise ValueError(
            "Annular search band is empty – check MIN_SPOT_DISTANCE / "
            "MAX_SPOT_DISTANCE relative to image size."
        )

    # Percentile threshold (computed only inside the annulus)
    threshold = np.percentile(amp_shifted[annular], SPOT_THRESHOLD_PERCENTILE)

    # Local-maximum filter
    footprint_size = LOCAL_MAX_NEIGHBORHOOD * 2 + 1
    local_max = maximum_filter(amp_shifted, size=footprint_size)
    is_peak = (amp_shifted == local_max) & (amp_shifted >= threshold) & annular

    peak_coords = list(zip(*np.where(is_peak)))   # list of (r, c)

    if not peak_coords:
        raise ValueError(
            f"No Bragg peaks found above the {SPOT_THRESHOLD_PERCENTILE}th "
            "percentile threshold. Try lowering SPOT_THRESHOLD_PERCENTILE or "
            "widening the MIN/MAX_SPOT_DISTANCE range."
        )

    # 6-fold symmetry expansion
    if ENFORCE_6FOLD_SYMMETRY:
        expanded = set()
        for (r, c) in peak_coords:
            dy = r - cy
            dx = c - cx
            d = math.sqrt(dy ** 2 + dx ** 2)
            if d < 1e-6:
                continue
            base_angle = math.atan2(dy, dx)
            for k in range(6):
                angle = base_angle + k * math.pi / 3.0
                nr = int(round(cy + d * math.sin(angle)))
                nc = int(round(cx + d * math.cos(angle)))
                # Clamp to image bounds
                nr = max(0, min(H - 1, nr))
                nc = max(0, min(W - 1, nc))
                expanded.add((nr, nc))
        peak_coords = list(expanded)

    # Deduplicate: remove peaks within LOCAL_MAX_NEIGHBORHOOD px of each other
    deduped = []
    peak_arr = np.array(peak_coords)  # shape (N, 2)
    used = [False] * len(peak_arr)
    for i in range(len(peak_arr)):
        if used[i]:
            continue
        deduped.append(tuple(peak_arr[i]))
        for j in range(i + 1, len(peak_arr)):
            if used[j]:
                continue
            if np.linalg.norm(peak_arr[i] - peak_arr[j]) < LOCAL_MAX_NEIGHBORHOOD:
                used[j] = True

    return deduped


def build_bragg_mask(shape, peaks_shifted, cy, cx):
    """
    Build a soft (Gaussian-falloff) binary mask in the shifted FFT frame.

    A Gaussian blob is placed at each Bragg peak and at the DC component.
    Clipped to [0, 1].  Soft edges prevent Gibbs ringing in the IFFT result.

    Returns
    -------
    np.ndarray, float64, shape == shape, values in [0, 1]
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float64)
    sigma = max(MASK_RADIUS / 2.0, 0.5)

    # Helper: add a Gaussian blob centred at (cr, cc)
    def add_blob(cr, cc):
        # Work on a local bounding box to avoid iterating all pixels
        r_min = max(0, cr - 4 * int(math.ceil(sigma)))
        r_max = min(H, cr + 4 * int(math.ceil(sigma)) + 1)
        c_min = max(0, cc - 4 * int(math.ceil(sigma)))
        c_max = min(W, cc + 4 * int(math.ceil(sigma)) + 1)
        rs = np.arange(r_min, r_max)
        cs = np.arange(c_min, c_max)
        rr, cc_ = np.meshgrid(rs, cs, indexing='ij')
        blob = np.exp(-((rr - cr) ** 2 + (cc_ - cc) ** 2) / (2 * sigma ** 2))
        mask[r_min:r_max, c_min:c_max] += blob

    # DC component (always retained)
    add_blob(cy, cx)

    # Bragg spots
    for (r, c) in peaks_shifted:
        add_blob(r, c)

    np.clip(mask, 0.0, 1.0, out=mask)
    return mask


def apply_mask_and_ifft(fft_raw, mask_shifted):
    """
    Apply the Bragg mask to the FFT and compute the IFFT.

    The mask lives in the fftshift-ed (DC-centred) frame, so we must
    shift the raw FFT before multiplying, then un-shift before ifft2.

    Returns
    -------
    np.ndarray, float64 – real-space filtered image
    """
    fft_shifted = np.fft.fftshift(fft_raw)
    fft_masked  = fft_shifted * mask_shifted
    fft_back    = np.fft.ifftshift(fft_masked)
    reconstructed = np.real(np.fft.ifft2(fft_back))
    return reconstructed


def postprocess(image_array):
    """
    Gaussian blur → normalize to [0, 1] → gamma correction.

    Returns
    -------
    np.ndarray, float64, values in [0, 1]
    """
    # Gaussian smoothing
    blurred = gaussian_filter(image_array, sigma=GAUSSIAN_RADIUS)

    # Min-max normalization (guard against flat image)
    lo, hi = blurred.min(), blurred.max()
    if hi - lo < 1e-12:
        raise ValueError(
            "Post-IFFT image is effectively flat – the mask may be too "
            "restrictive or the image has no signal."
        )
    normalized = (blurred - lo) / (hi - lo)

    # Gamma correction  (V_out = V_in ^ gamma)
    result = np.power(normalized, GAMMA_VALUE)
    return result


def get_calibration(dm_image):
    """
    Read pixel calibration from a DM Image object.

    Returns
    -------
    dict with keys: origin_x, scale_x, units_x, origin_y, scale_y, units_y
    """
    try:
        # DM dimension convention: 0 = columns (x), 1 = rows (y)
        ox, sx, ux = dm_image.GetDimensionCalibration(0, 0)
        oy, sy, uy = dm_image.GetDimensionCalibration(1, 0)
        return dict(origin_x=ox, scale_x=sx, units_x=ux,
                    origin_y=oy, scale_y=sy, units_y=uy)
    except Exception as exc:
        raise RuntimeError(f"Failed to read calibration: {exc}") from exc


def set_calibration(dm_image, calib):
    """
    Write pixel calibration to a DM Image object.

    Parameters
    ----------
    calib : dict returned by get_calibration()
    """
    try:
        dm_image.SetDimensionCalibration(
            0, calib['origin_x'], calib['scale_x'], calib['units_x'], 0)
        dm_image.SetDimensionCalibration(
            1, calib['origin_y'], calib['scale_y'], calib['units_y'], 0)
    except Exception as exc:
        raise RuntimeError(f"Failed to write calibration: {exc}") from exc


def enumerate_open_images():
    """
    Enumerate all valid 2-D images currently open in the GMS workspace.

    Returns
    -------
    List of (DM.Image, str) tuples – (image object, image name).
    """
    results = []
    try:
        n_windows = DM.GetNumberOfDocumentWindows()
    except AttributeError:
        # Older GMS builds may not have GetNumberOfDocumentWindows;
        # fall back to iterating until an exception is raised.
        n_windows = None

    def try_add_window(win):
        if not win.IsValid():
            return
        try:
            img = win.GetImage()
        except Exception:
            return  # Not an image window (script, spectrum, etc.)
        try:
            if img.GetNumDimensions() != 2:
                return  # Skip stacks, profiles, spectrum images
            name = img.GetName()
            results.append((img, name))
        except Exception:
            return

    if n_windows is not None:
        for i in range(n_windows):
            try:
                win = DM.GetDocumentWindow(i)
                try_add_window(win)
            except Exception:
                continue
    else:
        i = 0
        while True:
            try:
                win = DM.GetDocumentWindow(i)
                try_add_window(win)
                i += 1
            except Exception:
                break  # No more windows

    return results


# ============================================================
# SECTION 4 – PER-IMAGE PROCESSING FUNCTION
# ============================================================

def process_single_image(dm_image, image_name):
    """
    Run the full denoising pipeline on one DM Image.

    Returns
    -------
    (output_dm_image, n_peaks) : (DM.Image, int)

    Raises on any failure (caller catches and logs).
    """
    # --- 1. Extract numpy array ---
    raw_array = dm_image.GetNumArray()
    if raw_array.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {raw_array.ndim}-D.")
    img_float = raw_array.astype(np.float64)

    # --- 2. Read calibration before any processing ---
    calib = get_calibration(dm_image)

    # --- 3. FFT ---
    fft_raw, amp_shifted, cy, cx = compute_fft(img_float)

    # --- 4. Detect Bragg peaks (raises if none found) ---
    peaks = find_bragg_peaks(amp_shifted, cy, cx)
    n_peaks = len(peaks)

    # --- 5. Build soft mask ---
    mask = build_bragg_mask(img_float.shape, peaks, cy, cx)

    # --- 6. Masked IFFT ---
    filtered = apply_mask_and_ifft(fft_raw, mask)

    # --- 7. Post-processing (blur + gamma) ---
    result = postprocess(filtered)

    # --- 8. Validate output ---
    if not np.isfinite(result).all():
        raise ValueError("Post-processing produced non-finite values (NaN/Inf).")

    # --- 9. Create output DM image ---
    result_f32 = result.astype(np.float32)
    out_img = DM.CreateImage(result_f32.copy())   # .copy() transfers ownership

    out_name = image_name + "_denoised"
    out_img.SetName(out_name)

    # --- 10. Copy calibration ---
    set_calibration(out_img, calib)

    # --- 11. Display ---
    out_img.ShowImage()

    return out_img, n_peaks


# ============================================================
# SECTION 5 – MAIN LOOP
# ============================================================

def run_denoising():
    """
    Enumerate all open 2-D images and apply the denoising pipeline to each.
    Failed images are logged and skipped; processing continues for the rest.
    """
    print("=" * 60)
    print("MoS2 STEM Bragg-Filter Denoising  –  starting")
    print(f"  GAUSSIAN_RADIUS           = {GAUSSIAN_RADIUS}")
    print(f"  GAMMA_VALUE               = {GAMMA_VALUE}")
    print(f"  MASK_RADIUS               = {MASK_RADIUS}")
    print(f"  SPOT_THRESHOLD_PERCENTILE = {SPOT_THRESHOLD_PERCENTILE}")
    print(f"  MIN/MAX_SPOT_DISTANCE     = {MIN_SPOT_DISTANCE} / {MAX_SPOT_DISTANCE}")
    print(f"  ENFORCE_6FOLD_SYMMETRY    = {ENFORCE_6FOLD_SYMMETRY}")
    print("=" * 60)

    images = enumerate_open_images()

    if not images:
        msg = "No open 2-D images found in the workspace.\nPlease open your STEM images first."
        print("[!] " + msg)
        DM.OkCancelDialog(msg)
        return

    print(f"Found {len(images)} open 2-D image(s).\n")

    success_count = 0
    error_log     = []

    for dm_image, name in images:
        print(f"Processing: '{name}' …")
        try:
            _out, n_peaks = process_single_image(dm_image, name)
            success_count += 1
            print(f"  [OK]  →  '{name}_denoised'  ({n_peaks} Bragg peaks masked)\n")
        except Exception as exc:
            msg = f"  [ERROR]  '{name}':  {type(exc).__name__}: {exc}"
            print(msg + "\n")
            error_log.append(f"• {name}: {exc}")

    # ---- Summary ----
    summary_lines = [
        f"Denoising complete.",
        f"",
        f"Processed successfully: {success_count} / {len(images)} image(s).",
    ]
    if error_log:
        summary_lines += ["", "Errors:"] + error_log
    else:
        summary_lines.append("All images processed without errors.")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    DM.OkCancelDialog(summary)


# ============================================================
# SECTION 6 – ENTRY POINT
# ============================================================

try:
    run_denoising()
except Exception as fatal:
    fatal_msg = f"Fatal script error (outside per-image handling):\n{type(fatal).__name__}: {fatal}"
    print("[FATAL] " + fatal_msg)
    DM.OkCancelDialog(fatal_msg)
    raise
