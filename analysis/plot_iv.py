"""
plot_iv.py
----------
Authoritative saturation analysis: fit the Gaussian lattice model to the
measured saturation curve and produce the publication figures.

Pipeline position:  build_h5 -> parse_datasheets -> fit_scan -> plot_iv

Usage
-----
    python plot_iv.py                     # all points, full diagnostics
    python plot_iv.py --max-current 2e-3  # keep only points below 2 mA
    python plot_iv.py --no-show           # write files, no interactive window

What changed relative to the previous version, and why
------------------------------------------------------
1.  CELL PITCH.  The lattice spacing is the microcell TILING PERIOD,
    40.66 um, not the 35 um active dimension (whose fill factor is already
    inside PDE).  See the geometry block in parse_datasheets.py.  This is an
    exact reparameterisation — the model depends on pitch and tau only through
    pitch^2 * tau — so it rescales tau by (40.66/35)^2 = 1.35 and changes
    nothing else.

2.  ERROR BUDGET.  Calibration constants (SiPM gain, ECF, photodiode
    responsivity, beamsplitter ratio, PDE, beam widths) move every point
    coherently and no longer sit in the per-point sigma.  They are propagated
    by shifting each by +/- 1 sigma and refitting.  The previous treatment put
    them all in sigma_i, which gave chi2/dof = 0.17 and hid real structure.

3.  REPRODUCIBILITY FLOOR.  The standard error of the averaged readings
    (0.03-0.1%) is not the point-to-point uncertainty.  Repeat runs at the same
    ND setting reproduce the SiPM/PD ratio to 0.3-1.9%, so a 1% floor is
    applied.  See rates.REPRODUCIBILITY_FLOOR.

4.  x-AXIS ERRORS ARE NOW USED.  Folded in with the effective-variance method
    using the closed-form derivative; previously plotted but ignored.

5.  THE HIGH-CURRENT POINTS ARE FLAGGED.  The device reaches 22 mA at 2.5 V
    and 59 mA at 4.0 V, four orders of magnitude above the datasheet operating
    current.  Constant gain cannot be assumed there and the residuals show it.
    `--max-current` applies a cut and `tau_vs_current_cut` reports the
    stability of tau against it.  No cut is applied by default: this is a
    physics decision, and the diagnostic is printed so it can be made
    explicitly rather than silently.  (The previous version silently dropped
    the three brightest 4.0 V points with a bare `order[:-3]`.)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import matplotlib.pyplot as plt

import parse_datasheets as ds
import saturation_model as sm
import rates
import fit_saturation as fs
import plotstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, 'data', 'measurements.h5')
PLOT_DIR = os.path.join(HERE, 'plots', 'plot_iv')
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def residual_structure(residuals):
    """
    Sign-change count of the ordered residuals.

    For residuals that are pure noise the expected number of sign changes is
    (n-1)/2.  A value far below that means the deviation from the model is
    coherent — i.e. the model is wrong in a systematic way — and cannot be
    fixed by enlarging the error bars.
    """
    s = np.sign(residuals)
    changes = int(np.sum(s[1:] != s[:-1]))
    expected = (len(residuals) - 1) / 2.0
    return changes, expected


def tau_vs_current_cut(ms, free_pde=False, cuts_mA=(np.inf, 20, 10, 5, 2, 1)):
    """
    Refit while progressively removing the highest-current points.

    A tau that drifts with the cut means the high-current data are pulling the
    fit; a tau that plateaus means the cut is safe.
    """
    rows = []
    for cut in cuts_mA:
        sub = ms if not np.isfinite(cut) else ms.current_cut(cut * 1e-3)
        if len(sub) < 8:
            continue
        r = fs.fit_tau(sub, free_pde=free_pde)
        frac = sub.y / sm.gaussian_rate(sub.x, r.tau, r.pde, sub.sigma_x,
                                        sub.sigma_y, sub.pitch) - 1.0
        changes, expected = residual_structure(frac)
        rows.append(dict(cut_mA=cut, n=len(sub), tau=r.tau, tau_err=r.tau_stat,
                         pde=r.pde, chi2_red=r.chi2_red,
                         frac_rms=float(frac.std(ddof=1)),
                         sign_changes=changes, sign_expected=expected))
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_raw_iv(sets, path):
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, ps.COLUMN_WIDTH * 0.8))
    for ms in sets:
        st = ps.OV_STYLE[ms.ov_key]
        i_pd = ms.x / ds.PD_PARAMS["sipm_fraction"] * ds.PD_PARAMS["pd_fraction"] \
            * rates.E_PHOTON * ds.PD_PARAMS["effective_responsivity_A_per_W"]
        ax.plot(i_pd * 1e9, ms.i_sipm * 1e6, st['marker'], color=st['color'],
                label=st['label'], linestyle='none')
    ax.set_xlabel(r'Photodiode current (nA)')
    ax.set_ylabel(r'SiPM current ($\mu$A)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ps.grid(ax); ax.legend(loc='lower right')
    fig.tight_layout()
    return ps.save(fig, path)


def figure_saturation(sets, fits, path):
    """The money plot: response, efficiency ratio, and pulls."""
    fig, (axr, axm, axp) = plt.subplots(
        3, 1, figsize=(ps.COLUMN_WIDTH * 1.55, ps.COLUMN_WIDTH * 2.5),
        gridspec_kw={'height_ratios': [1.1, 2.4, 1.0]}, sharex=True)
    fig.subplots_adjust(hspace=0.06)

    for ms, fit in zip(sets, fits):
        st = ps.OV_STYLE[ms.ov_key]
        xs = np.geomspace(ms.x.min() * 0.85, ms.x.max() * 1.2, 400)
        model = sm.gaussian_rate(xs, fit.tau, fit.pde, ms.sigma_x, ms.sigma_y,
                                 ms.pitch)

        # --- main panel ---
        axm.errorbar(ms.x, ms.y, xerr=ms.x_err, yerr=ms.y_err,
                     fmt=st['marker'], color=st['color'], linestyle='none',
                     elinewidth=0.7, label=f"{st['label']} data")
        axm.plot(xs, model, '-', color=st['color'],
                 label=(rf"{st['label']} fit: $\tau={fit.tau*1e9:.0f}\pm"
                        rf"{fit.tau_syst*1e9:.0f}$ ns "
                        rf"$={fit.tau_multiple:.1f}\,\tau_0$"))

        # --- efficiency ratio ---
        axr.errorbar(ms.x, ms.y / ms.x, yerr=ms.y_err / ms.x,
                     fmt=st['marker'], color=st['color'], linestyle='none',
                     elinewidth=0.7)
        axr.plot(xs, model / xs, '-', color=st['color'])
        axr.axhline(ms.pde, color=st['color'], linestyle=':', linewidth=0.9)

        # --- pulls ---
        axp.plot(ms.x, fit.residuals, st['marker'], color=st['color'],
                 linestyle='none',
                 label=rf"{st['label']}  $\chi^2/\nu={fit.chi2_red:.1f}$")

    axm.set_ylabel(r'$N_{\rm fired}$  (primary avalanches s$^{-1}$)')
    axm.set_xscale('log'); axm.set_yscale('log')
    ps.grid(axm); axm.legend(loc='upper left')

    axr.set_ylabel(r'$N_{\rm fired}/N_{\rm incident}$')
    axr.set_xscale('log'); axr.set_yscale('log')
    ps.grid(axr)

    axp.axhline(0, color='k', linewidth=0.7, linestyle='--')
    axp.set_xlabel(r'$N_{\rm incident}$  (photons s$^{-1}$ on the SiPM face)')
    axp.set_ylabel(r'Pull ($\sigma$)')
    axp.set_xscale('log')
    ps.grid(axp); axp.legend(loc='upper left')

    fig.tight_layout()
    return ps.save(fig, path)


def figure_cut_stability(scans, path):
    """tau and residual rms against the high-current cut."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(ps.COLUMN_WIDTH * 1.35, ps.COLUMN_WIDTH * 1.6),
        sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ov_key, rows in scans.items():
        st = ps.OV_STYLE[ov_key]
        cuts = [r['cut_mA'] if np.isfinite(r['cut_mA']) else 100 for r in rows]
        ax1.errorbar(cuts, [r['tau'] * 1e9 for r in rows],
                     yerr=[r['tau_err'] * 1e9 for r in rows],
                     fmt=st['marker'] + '-', color=st['color'],
                     label=st['label'], elinewidth=0.7)
        ax2.plot(cuts, [100 * r['frac_rms'] for r in rows],
                 st['marker'] + '-', color=st['color'])

    ax1.set_ylabel(r'Fitted $\tau$ (ns)')
    ax1.set_xscale('log')
    ps.grid(ax1); ax1.legend()
    ax2.set_ylabel('Fractional residual rms (%)')
    ax2.set_xlabel('SiPM current cut (mA);  100 = no cut')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ps.grid(ax2)
    fig.tight_layout()
    return ps.save(fig, path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(ms, fit, scan):
    print("=" * 78)
    print(f"{ms.ov_key}   {ms.label}   N = {len(ms)} points"
          + (f"   ({ms.dropped} brightest dropped)" if ms.dropped else ""))
    print(f"  cell pitch      : {ms.pitch*1e6:.3f} um   (tiling period)")
    print(f"  beam widths     : sigma_x = {ms.sigma_x*1e3:.4f} mm, "
          f"sigma_y = {ms.sigma_y*1e3:.4f} mm")
    print(f"  SiPM current    : {ms.i_sipm.min()*1e6:.1f} uA .. "
          f"{ms.i_sipm.max()*1e3:.2f} mA")
    print(f"  readout errors  : median {np.median(ms.y_err_readout/ms.y):.3%}"
          f"   + {ms.floor:.1%} reproducibility floor")
    print()
    print(f"  {fit.summary()}")

    changes, expected = residual_structure(fit.residuals)
    print(f"  residual sign changes: {changes} of {len(fit.residuals)-1} "
          f"(noise would give ~{expected:.0f})")
    if changes < expected / 3:
        print("    -> residuals are COHERENT: model-data disagreement is "
              "systematic, not scatter")

    print(f"\n  systematic budget on tau (ns):")
    for k, v in sorted(fit.syst_terms.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20s} {v*1e9:8.2f}")
    print(f"    {'TOTAL':<20s} {fit.tau_syst*1e9:8.2f}")

    print(f"\n  stability against the high-current cut:")
    print(f"    {'cut':>8} {'n':>4} {'tau (ns)':>12} {'PDE':>8} "
          f"{'chi2/dof':>9} {'resid rms':>10} {'signchg':>8}")
    for r in scan:
        lbl = 'none' if not np.isfinite(r['cut_mA']) else f"{r['cut_mA']:g} mA"
        print(f"    {lbl:>8} {r['n']:>4d} {r['tau']*1e9:>8.1f}"
              f" +/-{r['tau_err']*1e9:<4.1f} {r['pde']:>8.4f} "
              f"{r['chi2_red']:>9.1f} {100*r['frac_rms']:>9.2f}% "
              f"{r['sign_changes']:>4d}/{r['sign_expected']:.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--max-current', type=float, default=None,
                   help='discard points above this SiPM current (A)')
    p.add_argument('--free-pde', action='store_true',
                   help='also float the PDE (cross-check fit)')
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args()

    ps.use_publication_style()

    sets, fits, scans, results = [], [], {}, {}
    with h5py.File(DATA_FILE, 'r') as f:
        for ov_key in ('OVfive', 'OVfour'):
            if ov_key not in f:
                continue
            ms = rates.load_overvoltage(f, ov_key)
            if args.max_current:
                ms = ms.current_cut(args.max_current)
            fit = fs.fit_with_systematics(ms, free_pde=args.free_pde)
            scan = tau_vs_current_cut(ms, free_pde=args.free_pde)

            sets.append(ms); fits.append(fit); scans[ov_key] = scan
            report(ms, fit, scan)

            results[ov_key] = dict(
                overvoltage_V=ms.ov, n_points=len(ms),
                cell_pitch_um=ms.pitch * 1e6,
                sigma_x_mm=ms.sigma_x * 1e3, sigma_y_mm=ms.sigma_y * 1e3,
                tau_ns=fit.tau * 1e9, tau_stat_ns=fit.tau_stat * 1e9,
                tau_syst_ns=fit.tau_syst * 1e9,
                tau_over_tau0=fit.tau_multiple,
                pde=fit.pde, pde_free=bool(args.free_pde),
                chi2=fit.chi2, ndof=fit.ndof, chi2_red=fit.chi2_red,
                systematics_ns={k: v * 1e9 for k, v in fit.syst_terms.items()},
                current_cut_scan=[
                    {k: (None if k == 'cut_mA' and not np.isfinite(v) else v)
                     for k, v in row.items()} for row in scan],
            )

    print("=" * 78)
    written = [
        figure_raw_iv(sets, os.path.join(PLOT_DIR, 'raw_iv')),
        figure_saturation(sets, fits, os.path.join(PLOT_DIR, 'saturation_curve')),
        figure_cut_stability(scans, os.path.join(PLOT_DIR, 'cut_stability')),
    ]
    out_json = os.path.join(PLOT_DIR, 'results.json')
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2)

    print("Written:")
    for w in written:
        print(f"  {w}")
    print(f"  {out_json}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
