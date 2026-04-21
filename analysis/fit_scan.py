"""
fit_scan.py
-----------
Fit a piecewise double-edge error-function model (razor-blade method) to
each x-line and y-line scan to extract the Gaussian beam sigma.

Model — parameterized by the two edge positions directly:
  x <= x_c:  f = offset + (A/2) * [1 + erf((x - x_left)  / (sqrt(2)*sigma))]
  x  > x_c:  f = offset + (A/2) * [1 - erf((x - x_right) / (sqrt(2)*sigma))]

where x_c = (x_left + x_right) / 2 is derived, not fitted.

Parameters: offset, amplitude, x_left (left edge), x_right (right edge), sigma
  - x_left and x_right float freely; bounds keep each edge inside the scan range
    to prevent the degenerate solution where both edges are pushed outside the data
  - Both edges share the same sigma (symmetric Gaussian beam)

Sigma vs radius convention:
  The erf argument is (x - edge)/(sqrt(2)*sigma), so popt[4] is the Gaussian
  standard deviation (sigma), NOT the beam radius (2*sigma).  darroch_error.py
  uses sqrt(2)*(center-x)/width so its popt[4] = 2*sigma (diameter); saturation_curve.py
  therefore divides by 2.  fit_scan.py stores sigma directly — no factor-of-2 needed
  when passing to Nfired_func.

Scan mapping:
  xline_scan  (x fixed, y scans) → sigma_y  (beam width in Y direction)
  yline_scan  (y fixed, x scans) → sigma_x  (beam width in X direction)

Results are saved as attributes on each run group in the HDF5 file:
  sigma_x_mm, sigma_x_err_mm, sigma_y_mm, sigma_y_err_mm,
  chi2_red_xline, chi2_red_yline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'measurements.h5')
PLOT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'center_fit')
os.makedirs(PLOT_DIR, exist_ok=True)

COLS = ['x', 'y', 'sipm_current', 'sipm_std', 'sipm_stderr',
        'sipm_time', 'pd_current', 'pd_std', 'pd_stderr', 'pd_time']
COL = {name: i for i, name in enumerate(COLS)}

N_PARAMS = 5   # offset, amplitude, x_c, w, sigma

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def razor_blade(pos, offset, amplitude, x_left, x_right, sigma):
    """
    Symmetric piecewise double-edge profile.
    x_left:  left edge position  (free parameter)
    x_right: right edge position (free parameter)
    sigma:   beam Gaussian sigma — shared by both edges (symmetric)
    x_c = (x_left + x_right) / 2 is the derived device centre.

    Parameterizing by x_left/x_right (rather than x_c/w) lets the optimizer
    be bounded to keep each edge inside the scan range, preventing the degenerate
    solution where w >> scan range and both edges are pushed outside the data.
    """
    pos    = np.asarray(pos, dtype=float)
    x_c    = (x_left + x_right) / 2.0
    out    = np.empty_like(pos)
    m      = pos <= x_c
    out[ m] = offset + (amplitude / 2) * (1 + erf((pos[ m] - x_left)  / (np.sqrt(2) * sigma)))
    out[~m] = offset + (amplitude / 2) * (1 - erf((pos[~m] - x_right) / (np.sqrt(2) * sigma)))
    return out

# ---------------------------------------------------------------------------
# Initial parameter estimation
# ---------------------------------------------------------------------------

def estimate_p0(pos, signal):
    offset    = float(np.percentile(signal, 5))
    amplitude = float(np.percentile(signal, 95) - offset)

    # Gradient-magnitude edge detection — works regardless of signal sign.
    # Split into left and right halves so we find one edge per half.
    grad_mag = np.abs(np.gradient(signal, pos))
    mid = len(pos) // 2
    x_left  = float(pos[np.argmax(grad_mag[:mid])])
    x_right = float(pos[mid + np.argmax(grad_mag[mid:])])

    sigma = max(abs(x_right - x_left) * 0.10, 0.10)
    return [offset, amplitude, x_left, x_right, sigma]

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_profile(pos, signal, signal_err):
    """
    Fit razor_blade to a single line scan.
    Errors are floored to 0.1% of the local signal to avoid
    numerical issues with near-zero stderr values.
    Returns (popt, perr, chi2_red, residuals, err_used) or Nones on failure.

    Bounds on x_left and x_right keep each edge inside the scan range, preventing
    the degenerate solution where the optimizer pushes both edges outside the data.
    """
    pos_min    = float(pos.min())
    pos_max    = float(pos.max())
    pos_center = (pos_min + pos_max) / 2.0

    err = np.maximum(signal_err, np.abs(signal) * 0.001 + 1e-12)
    p0  = estimate_p0(pos, signal)

    try:
        popt, pcov = curve_fit(
            razor_blade, pos, signal, p0=p0,
            sigma=err, absolute_sigma=True,
            bounds=([-np.inf,    0,  pos_min, pos_center, 1e-4],
                    [ np.inf, np.inf, pos_center, pos_max, np.inf]),
            maxfev=20000,
        )
        perr      = np.sqrt(np.diag(pcov))
        residuals = signal - razor_blade(pos, *popt)
        chi2      = float(np.sum((residuals / err) ** 2))
        chi2_red  = chi2 / max(len(signal) - N_PARAMS, 1)
        return popt, perr, chi2_red, residuals, err
    except (RuntimeError, ValueError):
        return None, None, np.nan, None, err

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_best_nd_erf_fits(
    nd,
    pos_xl, sig_xl, err_xl_f, popt_xl, perr_xl, chi2_xl, res_xl,
    pos_yl, sig_yl, err_yl_f, popt_yl, perr_yl, chi2_yl, res_yl,
    ov_label, fname,
):
    """
    Combined erf-fit figure for the highest-ND (least-saturated) run.

    2×2 layout — left column: X-line scan (scans Y → σ_y);
                  right column: Y-line scan (scans X → σ_x).
    Top row: data + fit.  Bottom row: residuals.
    """
    def _label(popt, perr, chi2):
        sigma_mm = popt[4]; sigma_err = perr[4]
        x_c = (popt[2] + popt[3]) / 2.0
        x_c_err = 0.5 * np.hypot(perr[2], perr[3])
        return (f'σ = {sigma_mm:.3f} ± {sigma_err:.3f} mm'
                f'   x_c = {x_c:.2f} ± {x_c_err:.2f} mm'
                f'   χ²/dof = {chi2:.2f}')

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex='col',
    )
    fig.subplots_adjust(hspace=0.06, wspace=0.28)
    fig.suptitle(f"{ov_label}  —  erf fit at highest ND = {nd:.2f}  "
                 f"(least-saturated run)")

    for col, (pos, sig, err, res, popt, perr, chi2, color, xlabel, scan_title) in enumerate([
        (pos_xl, sig_xl, err_xl_f, res_xl,
         popt_xl, perr_xl, chi2_xl, 'steelblue',
         'Y position (mm)', 'X-line scan  (scans Y → σ_y)'),
        (pos_yl, sig_yl, err_yl_f, res_yl,
         popt_yl, perr_yl, chi2_yl, 'darkorange',
         'X position (mm)', 'Y-line scan  (scans X → σ_x)'),
    ]):
        pos_fine = np.linspace(pos[0], pos[-1], 800)
        fit_fine = razor_blade(pos_fine, *popt)

        axes[0, col].errorbar(pos, sig * 1e3, yerr=err * 1e3,
                              fmt='o', markersize=3, color='gray',
                              elinewidth=0.8, capsize=2, label='Data', zorder=2)
        axes[0, col].plot(pos_fine, fit_fine * 1e3, color=color, linewidth=1.8,
                          label=_label(popt, perr, chi2), zorder=3)
        axes[0, col].set_ylabel('SiPM current (mA)')
        axes[0, col].set_title(scan_title)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        axes[1, col].errorbar(pos, res * 1e3, yerr=err * 1e3,
                              fmt='o', markersize=3, color=color,
                              elinewidth=0.8, capsize=2)
        axes[1, col].axhline(0, color='k', linewidth=0.8, linestyle='--')
        axes[1, col].set_xlabel(xlabel)
        axes[1, col].set_ylabel('Residual (mA)')
        axes[1, col].grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    fig.savefig(fname, dpi=150)
    plt.close(fig)


def _rolling_mad_outliers(nd, sigma, window=2, threshold=3.0):
    """
    Flag points that deviate from the rolling median of their nearest neighbors
    in ND-sorted order.

    For each point, the reference is the median of up to `window` neighbors on
    each side (the point itself is excluded).  The global MAD of all residuals
    sets the rejection scale, converted to a Gaussian-equivalent sigma by the
    factor 1/0.6745.  This catches isolated spikes that diverge from the local
    trend without penalising a smooth monotone region (like the saturation ramp).

    Returns outlier mask in the original (unsorted) index order.
    """
    order  = np.argsort(nd)
    sig_s  = sigma[order]
    n      = len(sig_s)

    rolling_med = np.empty(n)
    for i in range(n):
        lo   = max(0, i - window)
        hi   = min(n, i + window + 1)
        nbrs = np.concatenate([sig_s[lo:i], sig_s[i + 1:hi]])
        rolling_med[i] = np.median(nbrs) if len(nbrs) > 0 else sig_s[i]

    resid = sig_s - rolling_med
    mad   = np.median(np.abs(resid))
    if mad < 1e-8:                        # degenerate case (all same value)
        mad = np.std(resid) * 0.6745 + 1e-8

    outlier_s = np.abs(resid) > threshold * mad / 0.6745
    outlier   = np.zeros(len(nd), dtype=bool)
    outlier[order] = outlier_s
    return outlier


def _hockey_stick(nd, A, B, nd_c):
    """Piecewise: constant A for nd >= nd_c; ramp A + B*(nd_c - nd) below."""
    return A + B * np.maximum(nd_c - nd, 0.0)


def _fit_hockey_stick(nd, sigma, sigma_err):
    """
    Fit sigma(ND) = A + B * max(ND_c - ND, 0).

    A   : plateau beam sigma (the quantity we want)
    B   : ramp slope  (> 0, since sigma grows as ND decreases toward 0)
    ND_c: crossover — estimated saturation onset

    Returns (popt, perr) or (None, None) on failure.
    """
    high_nd = sigma[nd >= np.percentile(nd, 50)]
    A0   = float(np.median(high_nd)) if len(high_nd) > 0 else float(np.median(sigma))
    span = max(nd.max() - nd.min(), 0.1)
    B0   = max((sigma.max() - A0) / span, 0.01)
    nd_c0 = max(float(np.percentile(nd, 50)), nd.min() + 0.01)

    try:
        popt, pcov = curve_fit(
            _hockey_stick, nd, sigma,
            p0=[A0, B0, nd_c0],
            sigma=sigma_err, absolute_sigma=True,
            bounds=([0.0,   0.0,  nd.min()],
                    [2.0,  50.0,  nd.max()]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception:
        return None, None


SIGMA_HARD_MAX_MM = 2.0   # values outside (0, 2) mm are physically implausible


def plot_sigma_vs_nd(nd_arr, sigma_x, sigma_x_err, sigma_y, sigma_y_err, title, fname_prefix):
    """
    Beam sigma vs ND filter with hockey-stick fit for plateau extraction.

    Cleaning pipeline (per direction):
      1. Hard range cut: keep only 0 < sigma < SIGMA_HARD_MAX_MM (= 2 mm).
      2. Rolling-MAD outlier rejection (window=2 neighbours, 3-sigma threshold):
         flags isolated points that deviate from the local ND-sorted trend.
      3. Hockey-stick fit to cleaned data:
           sigma(ND) = A + B * max(ND_c - ND, 0)
         Plateau value = A ± A_err from the covariance matrix.
         Crossover ND_c is the fitted saturation onset.

    Returns dict with plateau sigma_x, sigma_x_err, sigma_y, sigma_y_err.
    """
    dir_results = {}

    for direction, sigma, sigma_err, color, marker, scan_note in [
        ('x', sigma_x, sigma_x_err, 'steelblue',  'o', 'Y-line scan'),
        ('y', sigma_y, sigma_y_err, 'darkorange',  's', 'X-line scan'),
    ]:
        # Step 1 — hard range cut
        valid = (
            ~np.isnan(sigma) & ~np.isnan(sigma_err) & (sigma_err > 0)
            & (sigma > 0) & (sigma < SIGMA_HARD_MAX_MM)
        )
        nd_v  = nd_arr[valid]
        sig_v = sigma[valid]
        err_v = sigma_err[valid]

        hard_cut_nd  = nd_arr[~valid & ~np.isnan(sigma)]
        hard_cut_sig = sigma[~valid & ~np.isnan(sigma)]

        # Step 2 — rolling-MAD outlier detection on the range-cut set
        if len(sig_v) >= 4:
            outlier = _rolling_mad_outliers(nd_v, sig_v, window=2, threshold=3.0)
        else:
            outlier = np.zeros(len(sig_v), dtype=bool)

        nd_in  = nd_v[~outlier]
        sig_in = sig_v[~outlier]
        err_in = err_v[~outlier]

        # Step 3 — hockey-stick fit
        popt, perr = (None, None)
        if len(sig_in) >= 4:
            popt, perr = _fit_hockey_stick(nd_in, sig_in, err_in)

        dir_results[direction] = dict(
            nd_v=nd_v, sig_v=sig_v, err_v=err_v,
            hard_cut_nd=hard_cut_nd, hard_cut_sig=hard_cut_sig,
            nd_in=nd_in, sig_in=sig_in, err_in=err_in,
            outlier=outlier, popt=popt, perr=perr,
            color=color, marker=marker, scan_note=scan_note,
        )

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title}  —  hockey-stick fit  |  "
                 f"hard cut: 0 < σ < {SIGMA_HARD_MAX_MM} mm  |  "
                 f"rolling-MAD outlier removal")

    out_dict = {}
    for ax, (direction, r) in zip(axes, dir_results.items()):
        color = r['color']; marker = r['marker']

        # hard-cut rejects
        if len(r['hard_cut_nd']) > 0:
            ax.scatter(r['hard_cut_nd'], r['hard_cut_sig'],
                       marker='x', color='grey', s=40, zorder=2,
                       label=f'Hard cut (|σ| > {SIGMA_HARD_MAX_MM} mm or invalid)')

        # rolling-MAD outliers
        if r['outlier'].any():
            ax.errorbar(r['nd_v'][r['outlier']], r['sig_v'][r['outlier']],
                        yerr=r['err_v'][r['outlier']],
                        fmt=marker, color='crimson', capsize=3, markersize=6,
                        alpha=0.8, label='Outliers (rolling-MAD)', zorder=3)

        # inliers
        ax.errorbar(r['nd_in'], r['sig_in'], yerr=r['err_in'],
                    fmt=marker, color=color, capsize=3, markersize=5,
                    label=f'σ_{direction}  ({r["scan_note"]})', zorder=4)

        # hockey-stick fit
        if r['popt'] is not None:
            A, B, nd_c   = r['popt']
            A_err, _, nd_c_err = r['perr']
            nd_fine = np.linspace(r['nd_in'].min(), r['nd_in'].max(), 400)
            ax.plot(nd_fine, _hockey_stick(nd_fine, A, B, nd_c),
                    color='k', linewidth=1.8,
                    label=(f'Hockey-stick fit\n'
                           f'  A = {A:.4f} ± {A_err:.4f} mm\n'
                           f'  ND_c = {nd_c:.2f} ± {nd_c_err:.2f}'))
            ax.axvline(nd_c, color='crimson', linewidth=1.2, linestyle='--',
                       label=f'Saturation onset ND = {nd_c:.2f}')
            ax.fill_between([r['nd_in'].min(), r['nd_in'].max()],
                            A - A_err, A + A_err,
                            alpha=0.20, color='k', label='Plateau ±1σ')
            out_dict[direction] = dict(mean=float(A), err=float(A_err),
                                       nd_c=float(nd_c))
            print(f"  sigma_{direction}: A = {A:.4f} ± {A_err:.4f} mm, "
                  f"ND_c = {nd_c:.2f} ± {nd_c_err:.2f}")
        else:
            # fallback: weighted mean of all inliers
            if len(r['sig_in']) > 0:
                w = 1.0 / r['err_in'] ** 2
                A     = float(np.sum(w * r['sig_in']) / np.sum(w))
                A_err = float(1.0 / np.sqrt(np.sum(w)))
                ax.axhline(A, color='k', linewidth=1.4,
                           label=f'Weighted mean σ_{direction} = {A:.4f} ± {A_err:.4f} mm')
                out_dict[direction] = dict(mean=A, err=A_err, nd_c=None)
            else:
                out_dict[direction] = dict(mean=np.nan, err=np.nan, nd_c=None)

        ax.set_xlabel("ND filter value")
        ax.set_ylabel("Beam sigma (mm)")
        ax.set_title(f"σ_{direction}  ({r['scan_note']})")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{fname_prefix}_saturation_plateau.png", dpi=150)
    plt.close(fig)

    return {
        'sigma_x':     out_dict.get('x', {}).get('mean', np.nan),
        'sigma_x_err': out_dict.get('x', {}).get('err',  np.nan),
        'sigma_y':     out_dict.get('y', {}).get('mean', np.nan),
        'sigma_y_err': out_dict.get('y', {}).get('err',  np.nan),
    }

def plot_all_scans_colormap(xline_scans, yline_scans, ov_label, fname):
    """
    Inspection plot: every line scan for one OV setting, coloured by ND value.

    xline_scans, yline_scans : list of (nd, pos_mm, signal_A) tuples
    No per-curve legend — colorbar encodes ND value.

    Use to visually verify:
      - Curves at high ND (dim beam) should all collapse onto the same shape
      - As ND decreases (brighter) the flat top widens → saturation distortion
      - The erf-fit sigma is the beam std dev; flat-top onset marks the
        saturation threshold used by _find_plateau.
    """
    nd_vals = np.array([nd for nd, _, _ in xline_scans])
    nd_min, nd_max = nd_vals.min(), nd_vals.max()
    if nd_min == nd_max:
        nd_min -= 0.1; nd_max += 0.1

    cmap = plt.cm.plasma
    norm = plt.Normalize(nd_min, nd_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{ov_label} — all line scans  |  colour = ND  (bright = low ND)\n"
        f"Fitted quantity is beam σ (std dev).  "
        f"Flat-top onset at low ND indicates SiPM saturation."
    )

    def _norm01(sig):
        lo, hi = sig.min(), sig.max()
        return (sig - lo) / (hi - lo) if hi > lo else np.zeros_like(sig)

    # Plot dim-beam first so bright-beam curves render on top
    for nd, pos, sig in sorted(xline_scans, key=lambda t: t[0], reverse=True):
        ax1.plot(pos, _norm01(sig), color=cmap(norm(nd)), linewidth=0.9, alpha=0.9)
    ax1.set_xlabel("Y position (mm)")
    ax1.set_ylabel("Normalised signal  (sig − min) / (max − min)")
    ax1.set_title("X-line scans  (scan along Y → σ_y extracted)")
    ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    for nd, pos, sig in sorted(yline_scans, key=lambda t: t[0], reverse=True):
        ax2.plot(pos, _norm01(sig), color=cmap(norm(nd)), linewidth=0.9, alpha=0.9)
    ax2.set_xlabel("X position (mm)")
    ax2.set_ylabel("Normalised signal  (sig − min) / (max − min)")
    ax2.set_title("Y-line scans  (scan along X → σ_x extracted)")
    ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.01, shrink=0.9)
    cbar.set_label("ND filter value")

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

with h5py.File(DATA_FILE, 'r+') as f:
    for ov_key in ['OVfive', 'OVfour']:
        if ov_key not in f:
            continue

        ov_label = '2.5 V OV' if ov_key == 'OVfive' else '4.0 V OV'
        results      = []   # (nd, sigma_x, sigma_x_err, sigma_y, sigma_y_err)
        xline_scans  = []   # (nd, pos_mm, signal_A) for colormap inspection plot
        yline_scans  = []
        best_run     = None  # data for the highest-ND (least-saturated) run

        run_names = sorted(
            f[ov_key].keys(),
            key=lambda n: (f[ov_key][n].attrs['nd'], f[ov_key][n].attrs['timestamp'])
        )

        for run_name in run_names:
            run = f[ov_key][run_name]
            nd  = float(run.attrs['nd'])

            if 'xline_scan' not in run or 'yline_scan' not in run:
                continue

            # xline_scan: x fixed, y varies  → profile in Y  → sigma_y
            xd     = run['xline_scan'][:]
            pos_xl = xd[:, COL['y']]
            sig_xl = xd[:, COL['sipm_current']]
            err_xl = xd[:, COL['sipm_stderr']]

            # yline_scan: y fixed, x varies  → profile in X  → sigma_x
            yd     = run['yline_scan'][:]
            pos_yl = yd[:, COL['x']]
            sig_yl = yd[:, COL['sipm_current']]
            err_yl = yd[:, COL['sipm_stderr']]

            popt_xl, perr_xl, chi2_xl, res_xl, err_xl_f = fit_profile(pos_xl, sig_xl, err_xl)
            popt_yl, perr_yl, chi2_yl, res_yl, err_yl_f = fit_profile(pos_yl, sig_yl, err_yl)

            # sigma_y from xline, sigma_x from yline
            sigma_y     = popt_xl[4] if popt_xl is not None else np.nan
            sigma_y_err = perr_xl[4] if perr_xl is not None else np.nan
            sigma_x     = popt_yl[4] if popt_yl is not None else np.nan
            sigma_x_err = perr_yl[4] if perr_yl is not None else np.nan

            results.append((nd, sigma_x, sigma_x_err, sigma_y, sigma_y_err))
            xline_scans.append((nd, pos_xl, sig_xl))
            yline_scans.append((nd, pos_yl, sig_yl))

            # Track the highest-ND run that has both fits (least-saturated)
            if popt_xl is not None and popt_yl is not None:
                if best_run is None or nd > best_run['nd']:
                    best_run = dict(
                        nd=nd,
                        pos_xl=pos_xl, sig_xl=sig_xl, err_xl_f=err_xl_f,
                        popt_xl=popt_xl, perr_xl=perr_xl, chi2_xl=chi2_xl, res_xl=res_xl,
                        pos_yl=pos_yl, sig_yl=sig_yl, err_yl_f=err_yl_f,
                        popt_yl=popt_yl, perr_yl=perr_yl, chi2_yl=chi2_yl, res_yl=res_yl,
                    )

            # --- Save to HDF5 ---
            run.attrs['sigma_x_mm']      = sigma_x     if not np.isnan(sigma_x)     else -1.0
            run.attrs['sigma_x_err_mm']  = sigma_x_err if not np.isnan(sigma_x_err) else -1.0
            run.attrs['sigma_y_mm']      = sigma_y     if not np.isnan(sigma_y)     else -1.0
            run.attrs['sigma_y_err_mm']  = sigma_y_err if not np.isnan(sigma_y_err) else -1.0
            run.attrs['chi2_red_xline']  = chi2_xl if np.isfinite(chi2_xl) else -1.0
            run.attrs['chi2_red_yline']  = chi2_yl if np.isfinite(chi2_yl) else -1.0

        # --- Colormap inspection plot (all scans, normalised) ---
        if xline_scans:
            plot_all_scans_colormap(
                xline_scans, yline_scans, ov_label,
                os.path.join(PLOT_DIR, f'{ov_key}_all_scans_colormap.png')
            )

        # --- Erf fit to the highest-ND (least-saturated) run ---
        if best_run is not None:
            plot_best_nd_erf_fits(
                ov_label=ov_label,
                fname=os.path.join(PLOT_DIR, f'{ov_key}_highest_nd_erf_fit.png'),
                **best_run,
            )

        # --- Sigma vs ND plot + plateau detection ---
        if results:
            results.sort(key=lambda r: r[0])
            nd_arr, sx, sx_e, sy, sy_e = map(np.array, zip(*results))
            plat = plot_sigma_vs_nd(
                nd_arr, sx, sx_e, sy, sy_e,
                f"Beam sigma vs ND filter  —  {ov_label}",
                os.path.join(PLOT_DIR, f'{ov_key}_sigma_vs_nd')
            )
            # Save plateau (pre-saturation) beam sigmas as group-level attrs
            f[ov_key].attrs['plateau_sigma_x_mm']     = plat['sigma_x']
            f[ov_key].attrs['plateau_sigma_x_err_mm'] = plat['sigma_x_err']
            f[ov_key].attrs['plateau_sigma_y_mm']     = plat['sigma_y']
            f[ov_key].attrs['plateau_sigma_y_err_mm'] = plat['sigma_y_err']
            print(f"  plateau sigma_x = {plat['sigma_x']:.4f} +/- {plat['sigma_x_err']:.4f} mm")
            print(f"  plateau sigma_y = {plat['sigma_y']:.4f} +/- {plat['sigma_y_err']:.4f} mm")
        print(f"{ov_key}: processed {len(results)} runs")

print("Done — sigma values saved to HDF5, plots written.")
plt.show()
