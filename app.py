"""
app.py  –  MoS₂ STEM Bragg-Filter Denoiser
==========================================
Standalone Streamlit web application for denoising HAADF-STEM images of
MoS₂ that are heavily covered with surface contamination.

Pipeline (applied interactively to the uploaded image):
  1. FFT        – 2D Fast Fourier Transform (numpy)
  2. Bragg mask – Detect hexagonal MoS₂ diffraction spots; build soft
                  Gaussian mask retaining ONLY those spots.
  3. IFFT       – Reconstruct filtered real-space image
  4. Gaussian   – Light blur to suppress Gibbs-ringing artifacts
  5. Gamma      – Power-law contrast boost (V_out = V_in ** γ, γ < 1)
  6. Output     – Display + download as float32 TIFF

Supported input formats: .dm3, .dm4 (Gatan), .tif / .tiff

Run:
    streamlit run app.py
"""

# ── Standard library ────────────────────────────────────────────────────────
import io
import math
import os
import tempfile

# ── Third-party ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import streamlit as st
import tifffile

# ── Page configuration (must be the first Streamlit call) ───────────────────
st.set_page_config(
    page_title="MoS₂ STEM Denoiser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme CSS ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"]          { background: #161b22; }
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

    /* Headings */
    h1  { color: #79c0ff !important; font-size: 1.6rem !important; }
    h2  { color: #58a6ff !important; font-size: 1.1rem !important; margin-top: 1rem !important; }
    h3  { color: #d2a8ff !important; }

    /* Download button */
    [data-testid="stDownloadButton"] button {
        background: #238636; color: #fff; border: none;
        width: 100%; padding: 0.6rem; font-size: 1rem;
        border-radius: 6px; cursor: pointer;
    }
    [data-testid="stDownloadButton"] button:hover { background: #2ea043; }

    /* Info / warning boxes */
    [data-testid="stAlert"] { border-radius: 8px; }

    /* Figure borders */
    .figure-box {
        border: 1px solid #30363d; border-radius: 8px;
        padding: 4px; background: #161b22;
    }

    /* Subtle caption colour */
    .stCaption { color: #8b949e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 – IMAGE LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_image(file_bytes: bytes, filename: str):
    """
    Load a STEM image from raw bytes.

    Returns
    -------
    data  : np.ndarray, float32, shape (H, W)
    meta  : dict  – calibration metadata (scale, units) when available
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("dm3", "dm4"):
        return _load_dm(file_bytes, ext)
    elif ext in ("tif", "tiff"):
        return _load_tif(file_bytes)
    else:
        raise ValueError(f"Unsupported format: .{ext}")


def _load_dm(file_bytes: bytes, ext: str):
    """Load .dm3 / .dm4 using hyperspy (preferred) or ncempy (fallback)."""
    # ── Attempt 1: hyperspy ─────────────────────────────────────────────────
    try:
        import hyperspy.api as hs

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            s = hs.load(tmp_path, lazy=False)
        finally:
            os.unlink(tmp_path)

        data = s.data
        # Handle image stacks: take the middle frame
        while data.ndim > 2:
            data = data[data.shape[0] // 2]

        meta = {}
        try:
            ax = s.axes_manager
            meta["scale_x"] = ax[-1].scale
            meta["scale_y"] = ax[-2].scale
            meta["units"]   = ax[-1].units
        except Exception:
            pass

        return data.astype(np.float32), meta

    except ImportError:
        pass  # hyperspy not available; try ncempy

    # ── Attempt 2: ncempy ───────────────────────────────────────────────────
    try:
        import ncempy.io.dm as ncemdm

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            dm = ncemdm.dmReader(tmp_path)
        finally:
            os.unlink(tmp_path)

        data = dm["data"]
        while data.ndim > 2:
            data = data[data.shape[0] // 2]

        meta = {}
        try:
            meta["scale_x"] = dm["pixelSize"][1]
            meta["scale_y"] = dm["pixelSize"][0]
            meta["units"]   = dm["pixelUnit"][0]
        except Exception:
            pass

        return data.astype(np.float32), meta

    except ImportError:
        raise ImportError(
            "Reading .dm3/.dm4 files requires **hyperspy** or **ncempy**.\n\n"
            "Install one of them:\n"
            "  pip install hyperspy\n"
            "  pip install ncempy"
        )


def _load_tif(file_bytes: bytes):
    """Load TIFF using tifffile (handles float32, 16-bit, etc.)."""
    buf = io.BytesIO(file_bytes)
    try:
        arr = tifffile.imread(buf)
    except Exception:
        from PIL import Image
        arr = np.array(Image.open(buf))

    # Convert colour to grayscale
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = (
                0.2989 * arr[:, :, 0]
                + 0.5870 * arr[:, :, 1]
                + 0.1140 * arr[:, :, 2]
            )
        else:
            arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {arr.shape}.")

    return arr.astype(np.float32), {}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 – FFT PROCESSING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def compute_fft(image: np.ndarray):
    """
    Returns
    -------
    fft_raw     : complex128  unshifted FFT  (→ IFFT path)
    amp_shifted : float64     log(|FFT|+1) with DC at centre  (→ peak detection)
    cy, cx      : int         DC pixel coordinates in the shifted frame
    """
    fft_raw     = np.fft.fft2(image.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft_raw)
    amp_shifted = np.abs(fft_shifted)          # linear amplitude (used for masking)
    H, W        = image.shape
    cy, cx      = H // 2, W // 2
    return fft_raw, amp_shifted, cy, cx


def find_bragg_peaks(
    amp_shifted: np.ndarray,
    cy: int,
    cx: int,
    threshold_percentile: float,
    min_spot_dist: float,
    max_spot_dist: float,
    enforce_6fold: bool,
    local_max_neighborhood: int,
):
    """
    Detect MoS₂ Bragg spots in the shifted FFT amplitude image.

    Returns list of (row, col) integer tuples.
    Raises ValueError if no peaks are found.
    """
    H, W   = amp_shifted.shape
    rows, cols = np.mgrid[0:H, 0:W]
    dist   = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    annular = (dist >= min_spot_dist) & (dist <= max_spot_dist)

    if not np.any(annular):
        raise ValueError(
            "The annular search band is empty. "
            "Reduce MIN or increase MAX spot distance."
        )

    threshold   = np.percentile(amp_shifted[annular], threshold_percentile)
    fsize       = local_max_neighborhood * 2 + 1
    local_max   = maximum_filter(amp_shifted, size=fsize)
    is_peak     = (
        (amp_shifted == local_max)
        & (amp_shifted >= threshold)
        & annular
    )
    peak_coords = list(zip(*np.where(is_peak)))

    if not peak_coords:
        raise ValueError(
            f"No Bragg peaks found above the {threshold_percentile}th percentile. "
            "Try lowering the threshold or widening the distance range."
        )

    # 6-fold symmetry expansion
    if enforce_6fold:
        expanded = set()
        for (r, c) in peak_coords:
            dy = r - cy
            dx = c - cx
            d  = math.hypot(dy, dx)
            if d < 1e-6:
                continue
            base = math.atan2(dy, dx)
            for k in range(6):
                angle = base + k * math.pi / 3.0
                nr = int(round(cy + d * math.sin(angle)))
                nc = int(round(cx + d * math.cos(angle)))
                expanded.add((max(0, min(H - 1, nr)), max(0, min(W - 1, nc))))
        peak_coords = list(expanded)

    # Deduplicate
    if len(peak_coords) > 1:
        arr    = np.array(peak_coords)
        used   = [False] * len(arr)
        deduped = []
        for i in range(len(arr)):
            if used[i]:
                continue
            deduped.append(tuple(arr[i]))
            for j in range(i + 1, len(arr)):
                if not used[j] and np.linalg.norm(arr[i] - arr[j]) < local_max_neighborhood:
                    used[j] = True
        peak_coords = deduped

    return peak_coords


def build_bragg_mask(
    shape: tuple,
    peaks: list,
    cy: int,
    cx: int,
    mask_radius: float,
):
    """
    Build a soft (Gaussian-falloff) mask in the shifted FFT frame.

    Always includes a DC blob. Clipped to [0, 1] to avoid artefacts
    from overlapping blobs (possible after 6-fold expansion).
    """
    H, W  = shape
    mask  = np.zeros((H, W), dtype=np.float64)
    sigma = max(mask_radius / 2.0, 0.5)
    pad   = int(math.ceil(4 * sigma))

    def _add_blob(cr: int, cc: int):
        r0, r1 = max(0, cr - pad), min(H, cr + pad + 1)
        c0, c1 = max(0, cc - pad), min(W, cc + pad + 1)
        rs = np.arange(r0, r1)
        cs = np.arange(c0, c1)
        rr, cc_ = np.meshgrid(rs, cs, indexing="ij")
        mask[r0:r1, c0:c1] += np.exp(
            -((rr - cr) ** 2 + (cc_ - cc) ** 2) / (2 * sigma ** 2)
        )

    _add_blob(cy, cx)         # DC component (always retained)
    for r, c in peaks:
        _add_blob(r, c)

    np.clip(mask, 0.0, 1.0, out=mask)
    return mask


def apply_mask_and_ifft(fft_raw: np.ndarray, mask_shifted: np.ndarray):
    """Apply Bragg mask in shifted frame and reconstruct via IFFT."""
    fft_masked = np.fft.fftshift(fft_raw) * mask_shifted
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft_masked)))


def postprocess(arr: np.ndarray, gaussian_radius: float, gamma: float):
    """Gaussian blur → normalize [0,1] → gamma correction."""
    blurred = gaussian_filter(arr, sigma=gaussian_radius) if gaussian_radius > 0 else arr.copy()
    lo, hi  = blurred.min(), blurred.max()
    if hi - lo < 1e-12:
        raise ValueError("Result image is flat (no contrast).")
    normed  = (blurred - lo) / (hi - lo)
    return np.power(normed, gamma, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 – CACHED PIPELINE ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline(
    image: np.ndarray,
    gaussian_radius: float,
    gamma: float,
    mask_radius: float,
    threshold_percentile: float,
    min_spot_dist: float,
    max_spot_dist: float,
    enforce_6fold: bool,
    local_max_neighborhood: int,
):
    """
    Full denoising pipeline.  Cached: re-runs only when parameters change.

    Returns a dict with all intermediate results for display.
    """
    fft_raw, amp_shifted, cy, cx = compute_fft(image)

    result = {
        "amp_shifted":  amp_shifted,
        "cy":           cy,
        "cx":           cx,
        "peaks":        [],
        "mask":         np.zeros_like(amp_shifted),
        "denoised":     None,
        "error":        None,
    }

    try:
        peaks = find_bragg_peaks(
            amp_shifted, cy, cx,
            threshold_percentile, min_spot_dist, max_spot_dist,
            enforce_6fold, local_max_neighborhood,
        )
        result["peaks"] = peaks

        mask = build_bragg_mask(image.shape, peaks, cy, cx, mask_radius)
        result["mask"] = mask

        filtered         = apply_mask_and_ifft(fft_raw, mask)
        result["denoised"] = postprocess(filtered, gaussian_radius, gamma)

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 – VISUALISATION HELPERS
# ════════════════════════════════════════════════════════════════════════════

_FIG_BG  = "#161b22"
_FIG_DPI = 120
_FIG_SZ  = (4.8, 4.8)


def _base_fig():
    fig, ax = plt.subplots(figsize=_FIG_SZ, dpi=_FIG_DPI)
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_FIG_BG)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return fig, ax


def fig_image(arr: np.ndarray, title: str, cmap: str = "gray"):
    """Render a 2-D float array with min-max scaling."""
    lo, hi = arr.min(), arr.max()
    display = (arr - lo) / (hi - lo + 1e-12)
    fig, ax = _base_fig()
    ax.imshow(display, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, color="#c9d1d9", fontsize=9, pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def fig_fft(
    amp_shifted: np.ndarray,
    cy: int,
    cx: int,
    title: str,
    peaks: list = None,
    mask: np.ndarray = None,
    marker_radius_px: float = 8,
):
    """
    Render the log-amplitude FFT.

    If *mask* is provided, the amplitude is multiplied by the mask first
    (→ 'masked FFT' panel).
    If *peaks* is provided, cyan circles are drawn at each peak.
    """
    if mask is not None:
        display_amp = amp_shifted * mask
    else:
        display_amp = amp_shifted

    log_amp = np.log1p(display_amp)
    lo, hi  = log_amp.min(), log_amp.max()
    display = (log_amp - lo) / (hi - lo + 1e-12)

    fig, ax = _base_fig()
    ax.imshow(display, cmap="inferno", interpolation="nearest", vmin=0, vmax=1)

    if peaks:
        # Scale marker radius so it's visually informative (not too tiny)
        vis_r = max(marker_radius_px, 6)
        for (r, c) in peaks:
            circ = plt.Circle(
                (c, r), vis_r,
                color="#00e5ff", fill=False, linewidth=1.2, alpha=0.85,
            )
            ax.add_patch(circ)
        # Legend proxy
        proxy = mpatches.Patch(
            facecolor="none", edgecolor="#00e5ff",
            label=f"{len(peaks)} spots",
        )
        ax.legend(
            handles=[proxy], loc="lower right", fontsize=7,
            framealpha=0.4, facecolor=_FIG_BG, edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )

    ax.set_title(title, color="#c9d1d9", fontsize=9, pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def render_fig(fig):
    """Render a matplotlib figure into Streamlit and close it."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 – TIFF EXPORT
# ════════════════════════════════════════════════════════════════════════════

def to_tiff_bytes(arr: np.ndarray) -> bytes:
    """Serialise a float32 array to an in-memory 32-bit TIFF."""
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr.astype(np.float32), photometric="minisblack")
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 – STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════

def sidebar() -> dict:
    """Render the sidebar and return a dict of all parameter values."""
    with st.sidebar:
        st.markdown("## 🔬 MoS₂ STEM Denoiser")
        st.caption("Upload a STEM image and tune the parameters below.")
        st.divider()

        # ── File upload ──────────────────────────────────────────────────────
        uploaded = st.file_uploader(
            "Upload STEM Image",
            type=["dm3", "dm4", "tif", "tiff"],
            help="Accepts Gatan .dm3/.dm4 and standard .tif/.tiff files.",
        )

        st.divider()

        # ── Post-processing ──────────────────────────────────────────────────
        st.markdown("### Post-Processing")
        gamma = st.slider(
            "Gamma (γ)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.05,
            help="Power-law exponent applied after normalization to [0,1].  "
                 "γ < 1 brightens dim atomic columns; γ > 1 suppresses them.",
        )
        gaussian_radius = st.slider(
            "Gaussian Blur Radius σ (px)",
            min_value=0.0, max_value=5.0, value=1.2, step=0.1,
            help="Standard deviation (σ) of the Gaussian smoothing applied "
                 "after IFFT to suppress Gibbs-ringing. 0 = disabled.",
        )

        st.divider()

        # ── Bragg mask ───────────────────────────────────────────────────────
        st.markdown("### Bragg Mask")
        mask_radius = st.slider(
            "Mask Radius (px)",
            min_value=2, max_value=40, value=8, step=1,
            help="Gaussian soft-mask half-width per detected Bragg spot.  "
                 "Larger = more bandwidth retained per spot.",
        )

        st.divider()

        # ── Peak detection ───────────────────────────────────────────────────
        st.markdown("### Peak Detection")
        threshold_pct = st.slider(
            "Threshold Percentile",
            min_value=90.0, max_value=99.9, value=99.0, step=0.1,
            help="Only FFT pixels above this amplitude percentile (within the "
                 "annular band) are considered Bragg-spot candidates.  "
                 "Raise to find fewer, stronger spots; lower to catch weak ones.",
        )
        min_dist = st.slider(
            "Min Distance from DC (px)",
            min_value=5, max_value=150, value=15, step=1,
            help="Annular search band inner radius.  "
                 "Excludes the DC component and low-frequency drift artefacts.",
        )
        max_dist = st.slider(
            "Max Distance from DC (px)",
            min_value=30, max_value=600, value=220, step=5,
            help="Annular search band outer radius.  "
                 "Excludes aliased high-frequency corners.",
        )
        local_neigh = st.slider(
            "Local-Max Suppression Radius (px)",
            min_value=3, max_value=40, value=10, step=1,
            help="Peaks within this distance of each other are merged into one.",
        )
        enforce_6fold = st.checkbox(
            "Enforce 6-fold Symmetry",
            value=True,
            help="Rotate every detected peak by k×60° (k=1..5) and mask all "
                 "6 symmetry partners.  Strongly recommended for MoS₂.",
        )

        st.divider()
        st.markdown("#### Reference d-spacings for MoS₂")
        st.caption(
            "• {10-10} family: d ≈ 2.73 Å  \n"
            "• {11-20} family: d ≈ 1.58 Å  \n\n"
            "FFT peak distance (px) ≈ image_size / (d / pixel_size)"
        )

    return dict(
        uploaded=uploaded,
        gamma=gamma,
        gaussian_radius=gaussian_radius,
        mask_radius=mask_radius,
        threshold_pct=threshold_pct,
        min_dist=float(min_dist),
        max_dist=float(max_dist),
        local_neigh=local_neigh,
        enforce_6fold=enforce_6fold,
    )


def landing_page():
    st.markdown(
        """
        ## Welcome to the MoS₂ STEM Bragg-Filter Denoiser

        Upload a STEM image using the **sidebar on the left** to begin.

        ---

        ### How it works

        | Step | Operation | Purpose |
        |------|-----------|---------|
        | 1 | **FFT** | Convert image to frequency space |
        | 2 | **Bragg masking** | Keep only MoS₂ crystal spots (6-fold); zero out amorphous background |
        | 3 | **IFFT** | Reconstruct filtered real-space image |
        | 4 | **Gaussian blur** | Suppress Gibbs-ringing artefacts (σ slider) |
        | 5 | **Gamma correction** | Non-linear contrast boost: V_out = V_in^γ |

        ---

        ### Why does this work?

        Surface contamination (amorphous carbon, organics) produces **diffuse,
        structureless** intensity in the FFT.  Crystalline MoS₂ produces a
        sharp **hexagonal pattern** of Bragg spots at well-defined d-spacings
        (~2.73 Å and ~1.58 Å).  By keeping only those spots we selectively
        reconstruct the crystal signal and suppress the contamination.

        ---

        ### Accepted file formats
        - **`.dm3` / `.dm4`** – Gatan DigitalMicrograph (requires *hyperspy* or *ncempy*)
        - **`.tif` / `.tiff`** – Standard TIFF (8-bit, 16-bit, or 32-bit float)
        """
    )


def main():
    params = sidebar()

    st.markdown("# 🔬 MoS₂ STEM Bragg-Filter Denoiser")
    st.caption(
        "FFT → Bragg Masking (6-fold symmetry) → IFFT → "
        "Gaussian Blur → Gamma Correction"
    )

    uploaded = params["uploaded"]

    if uploaded is None:
        landing_page()
        return

    # ── Load image ───────────────────────────────────────────────────────────
    with st.spinner("Loading image…"):
        try:
            image, meta = load_image(uploaded.getvalue(), uploaded.name)
        except Exception as exc:
            st.error(f"**Failed to load image:** {exc}")
            return

    H, W = image.shape
    info_cols = st.columns(4)
    info_cols[0].metric("Width",  f"{W} px")
    info_cols[1].metric("Height", f"{H} px")
    info_cols[2].metric("dtype",  str(image.dtype))
    if meta.get("scale_x"):
        info_cols[3].metric(
            "Pixel size",
            f"{meta['scale_x']:.4g} {meta.get('units', 'Å')}",
        )

    st.divider()

    # ── Run pipeline ─────────────────────────────────────────────────────────
    with st.spinner("Computing FFT and Bragg filter…"):
        res = run_pipeline(
            image,
            gaussian_radius=params["gaussian_radius"],
            gamma=params["gamma"],
            mask_radius=float(params["mask_radius"]),
            threshold_percentile=params["threshold_pct"],
            min_spot_dist=params["min_dist"],
            max_spot_dist=params["max_dist"],
            enforce_6fold=params["enforce_6fold"],
            local_max_neighborhood=params["local_neigh"],
        )

    # ── Error / success feedback ─────────────────────────────────────────────
    if res["error"]:
        st.warning(f"⚠️ Peak detection issue: {res['error']}")
        st.info(
            "Tip: try lowering **Threshold Percentile**, or adjust "
            "**Min / Max Distance from DC** to match your image's pixel size."
        )
    else:
        n_peaks = len(res["peaks"])
        st.sidebar.success(f"✅ {n_peaks} Bragg peaks detected")

    # ════════════════════════════════════════════════════════════════════════
    # ROW 1 – Real-space: Original  vs  Denoised
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("## Real-Space Images")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Original STEM Image**")
        render_fig(fig_image(image, "Original", cmap="gray"))

    with r1c2:
        st.markdown(
            f"**Denoised** &nbsp; (γ = {params['gamma']},  "
            f"σ = {params['gaussian_radius']} px)"
        )
        if res["denoised"] is not None:
            render_fig(fig_image(res["denoised"], "Denoised", cmap="gray"))
        else:
            st.info("Denoised image will appear once Bragg peaks are detected.")

    # ════════════════════════════════════════════════════════════════════════
    # ROW 2 – Frequency domain: FFT with spots  vs  Masked FFT
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("## Frequency Domain (FFT, log scale)")
    r2c1, r2c2 = st.columns(2)

    n_found = len(res["peaks"])
    with r2c1:
        st.markdown(
            f"**FFT Amplitude** &nbsp; — &nbsp; "
            f"{'cyan circles = detected Bragg spots' if n_found else 'no spots detected'}"
        )
        render_fig(
            fig_fft(
                res["amp_shifted"],
                res["cy"], res["cx"],
                title=f"FFT  ({n_found} spots marked)" if n_found else "FFT",
                peaks=res["peaks"] if n_found else None,
                marker_radius_px=params["mask_radius"],
            )
        )

    with r2c2:
        st.markdown("**Masked FFT** &nbsp; — &nbsp; only Bragg spots retained")
        render_fig(
            fig_fft(
                res["amp_shifted"],
                res["cy"], res["cx"],
                title="Masked FFT",
                mask=res["mask"],
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # DOWNLOAD
    # ════════════════════════════════════════════════════════════════════════
    if res["denoised"] is not None:
        st.divider()
        tiff_bytes = to_tiff_bytes(res["denoised"])
        base_name  = uploaded.name.rsplit(".", 1)[0]
        out_name   = f"{base_name}_denoised_g{params['gamma']}_s{params['gaussian_radius']}.tif"

        st.download_button(
            label="⬇️  Download Denoised Image (float32 TIFF)",
            data=tiff_bytes,
            file_name=out_name,
            mime="image/tiff",
            help="Saves as a 32-bit floating-point TIFF preserving full dynamic range.",
        )
        st.caption(
            f"Output: **{out_name}**  ·  "
            f"Shape: {res['denoised'].shape[1]} × {res['denoised'].shape[0]} px  ·  "
            f"Format: float32 TIFF"
        )


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
