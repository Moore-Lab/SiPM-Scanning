"""
Plot N_avalanches vs N_photons for both overvoltage settings.

Physics:
  SiPM:  I_sipm = gain * e * pde_eff * N_sipm * (1 + ct + ap)
         N_avalanches_total = (I_sipm - I_dark) / (gain * e)
         N_avalanches_primary = N_avalanches_total / (1 + ct + ap)
           (primary = photon-triggered only; ct and ap removed)

  PD:    I_pd = R_eff * P_pd    (R_eff in A/W; I_dark_pd = 0 at 0 V bias)
         P_pd = I_pd / R_eff                          (optical power on PD, W)
         N_photons_on_pd = P_pd / E_photon            (E_photon = hc / lambda)
         N_photons_total = N_photons_on_pd / pd_fraction
         N_photons_on_sipm = N_photons_total * sipm_fraction

Theory (Gaussian illumination — see docs/gaussian_rate_model.md):
  <R> = (2*pi*sigma_x*sigma_y / (dx*dy*tau)) * [gamma + ln(u) - Ei(-u)]
  u = PDE * R_gamma * dx * dy * tau / (2*pi*sigma_x*sigma_y)
  Single free parameter: tau (SPAD reset time)
  Fixed: PDE, sigma_x/y (plateau from fit_scan.py), dx=dy=35 um (datasheet)

Error propagation (all in quadrature):
  Instrument systematic errors from parse_datasheets.py
  Statistical errors from repeated measurements (stderr stored in HDF5)
  Estimated errors noted where no datasheet value is available
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.special import expi
from scipy.optimize import curve_fit

import parse_datasheets as ds

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'measurements.h5')
PLOT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'plot_iv')
os.makedirs(PLOT_DIR, exist_ok=True)

COLS = ['x', 'y', 'sipm_current', 'sipm_std', 'sipm_stderr',
        'sipm_time', 'pd_current', 'pd_std', 'pd_stderr', 'pd_time']
COL = {name: i for i, name in enumerate(COLS)}

# Map HDF5 group name → overvoltage key in SIPM_PARAMS
OV_MAP = {
    'OVfive': 2.5,
    'OVfour': 4.0,
}

OV_STYLE = {
    'OVfive': dict(label='2.5 V OV', color='steelblue',  marker='o'),
    'OVfour': dict(label='4.0 V OV', color='darkorange', marker='s'),
}

e        = const.e   # electron charge
E_PHOTON = const.h * const.c / (ds.LASER_D405["wavelength_typ_nm"] * 1e-9)   # J per 405 nm photon

# ---------------------------------------------------------------------------
# Error on N_photons (from PD):
#   N_pd = I_pd / R_eff
#   dN_pd/N_pd = sqrt( (dI_pd/I_pd)^2 + (dR_eff/R_eff)^2 )
#
# dI_pd: statistical stderr from HDF5 + systematic from Keithley datasheet
# dR_eff: calibration cert 25031152700 (≈1.45% 1-sigma at 405 nm) + spectral
#         model uncertainty added in quadrature.
#
# Error on N_avalanches (from SiPM):
#   N_av = (I_sipm - I_dark) / (gain * e)
#   dN_av = sqrt( dI_sipm^2 + dI_dark^2 ) / (gain * e)
#           gain uncertainty propagated separately below
#
# gain: estimated 5% relative uncertainty (datasheet gives values at 2 OV
#       points; interpolation uncertainty not characterised); NOTE — estimated.
# pde_eff: estimated 5% relative (graph digitisation + spectral model);
#          NOTE — estimated.
# crosstalk, afterpulsing: estimated 10% relative on the correction factor
#          (1 + ct + ap); NOTE — estimated, no calibration measurement.
# beamsplitter fraction: calibrated value 0.0902 from parse_datasheets.BEAMSPLITTER;
#          2% relative uncertainty retained.
# ---------------------------------------------------------------------------

R_eff       = ds.PD_PARAMS["effective_responsivity_A_per_W"]
I_dark_pd   = ds.PD_PARAMS["dark_current_A"]
pd_frac     = ds.PD_PARAMS["pd_fraction"]
sipm_frac   = ds.PD_PARAMS["sipm_fraction"]

REL_ERR_R_EFF         = 0.02   # calibration cert 25031152700: 2.9% k=2 at 405 nm → 1.45% 1-sigma; ~0.5% spectral model added in quadrature
REL_ERR_BS_FRAC       = 0.02   # calibrated fraction 0.0902; 2% relative uncertainty
REL_ERR_GAIN_DIGITIZE = 0.02   # residual graph-digitization uncertainty on gain curve
REL_ERR_CT_AP_DIGITIZE= 0.05   # residual digitization on ECF = (1+ct+ap)
# NOTE: dominant OV-dependent errors (gain, PDE, CT, AP) are now computed via
# VBD_ERR_V propagation in parse_datasheets.vbd_errors_at_ov() and stored
# in SIPM_PARAMS[ov]['*_err_vbd']; they are combined in avalanches_and_err().

# ---------------------------------------------------------------------------
# Gaussian illumination theory model
# ---------------------------------------------------------------------------

EULER_GAMMA  = 0.5772156649015329
CELL_PITCH_M = ds.SIPM_60035_GENERAL["microcell_pitch_um"] * 1e-6   # 35 µm → m
TAU_INIT_S   = ds.SIPM_60035_PERFORMANCE["microcell_recharge_tau_ns"] * 1e-9  # 50 ns (τ₀)
TAU_0_S      = TAU_INIT_S                                            # reference unit for multiplier fit


def gaussian_avalanche_rate(R_gamma, tau, pde, sigma_x_m, sigma_y_m):
    """
    Expected PRIMARY avalanche rate for a centred Gaussian beam.

    R_gamma   : photons/s incident on SiPM face
    tau       : SPAD reset time (s)  — the single fitted parameter
    pde       : laser-weighted effective PDE (dimensionless)
    sigma_x_m : beam sigma in X (m)
    sigma_y_m : beam sigma in Y (m)

    Returns avalanche rate (avalanches/s), primary only (CT/AP excluded).
    """
    dx = dy = CELL_PITCH_M
    u  = pde * R_gamma * dx * dy * tau / (2.0 * np.pi * sigma_x_m * sigma_y_m)
    u  = np.maximum(u, 1e-30)   # guard against log(0)
    prefactor = 2.0 * np.pi * sigma_x_m * sigma_y_m / (dx * dy * tau)
    return prefactor * (EULER_GAMMA + np.log(u) - expi(-u))


def Nfired_func(n, x_stdv, y_stdv, pde, reset):
    """
    Saturation-curve model: fired-microcell rate for Gaussian illumination.

    n      : N_incident — photons/s on SiPM face
    x_stdv : beam sigma_x (m)
    y_stdv : beam sigma_y (m)
    pde    : detection efficiency (dimensionless; fixed per OV setting)
    reset  : microcell recovery time tau (s) — free fit parameter

    Returns N_fired (primary avalanches/s, CT/AP excluded).

    Parameterisation matches saturation_curve.md:
      A   = 2*pi*sigma_x*sigma_y / (pitch^2 * reset)   [effective illuminated cells]
      arg = pde * n / A
      N_fired = A * [gamma + ln(arg) - Ei(-arg)]
    Identical to gaussian_avalanche_rate(n, reset, pde, x_stdv, y_stdv).
    """
    a   = CELL_PITCH_M**2 * reset
    A   = 2.0 * np.pi * x_stdv * y_stdv / a
    arg = np.maximum(pde * n / A, 1e-30)
    return A * (EULER_GAMMA + np.log(arg) - expi(-arg))


def photons_and_err(I_pd, I_pd_stat_err, keithley_sys_err):
    """
    Return (N_photons_total, N_photons_total_err, N_photons_on_sipm, N_photons_on_sipm_err).

    N_photons_total = photon rate at the beamsplitter input (photons/s).
    N_photons_on_sipm = N_photons_total * sipm_fraction (10% arm).

    Conversion chain:
      P_pd   = (I_pd - I_dark_pd) / R_eff          optical power on PD (W)
      N_pd   = P_pd / E_PHOTON                      photon rate on PD (photons/s)
      N_total = N_pd / pd_frac                      total rate at beamsplitter
      N_sipm  = N_total * sipm_frac                 rate on SiPM face
    """
    dI_pd   = np.sqrt(I_pd_stat_err**2 + keithley_sys_err**2)
    P_pd    = (I_pd - I_dark_pd) / R_eff      # optical power on PD (W)
    N_pd    = P_pd / E_PHOTON                  # photon rate on PD (photons/s)
    N_total = N_pd / pd_frac
    rel_err = np.sqrt((dI_pd / I_pd)**2 + REL_ERR_R_EFF**2 + REL_ERR_BS_FRAC**2)
    N_sipm  = N_total * sipm_frac
    return N_total, N_total * rel_err, N_sipm, N_sipm * rel_err


def avalanches_and_err(I_sipm, I_sipm_stat_err, siglent_sys_err,
                       gain, I_dark_sipm, pde_eff, ct, ap,
                       gain_err_vbd=0.0, ct_err_vbd=0.0, ap_err_vbd=0.0):
    """
    Return (N_av_total, N_av_total_err, N_av_primary, N_av_primary_err).

    N_av_total   = (I_sipm - I_dark) / (gain * e)  — includes CT and AP
    N_av_primary = N_av_total / (1 + ct + ap)       — photon-triggered only

    gain_err_vbd : absolute 1-sigma error on gain from VBD uncertainty (SIPM_PARAMS)
    ct_err_vbd   : absolute 1-sigma error on crosstalk fraction from VBD uncertainty
    ap_err_vbd   : absolute 1-sigma error on afterpulsing fraction from VBD uncertainty

    Gain error: VBD contribution added in quadrature with residual digitization.
    ECF error:  VBD contributions on ct and ap added in quadrature with digitization.
    """
    dI_sipm  = np.sqrt(I_sipm_stat_err**2 + siglent_sys_err**2)
    I_net    = I_sipm - I_dark_sipm
    N_av     = I_net / (gain * e)
    dN_av    = np.sqrt(dI_sipm**2 + (I_dark_sipm * 0.05)**2) / (gain * e)

    gain_rel_err = np.sqrt((gain_err_vbd / gain)**2 + REL_ERR_GAIN_DIGITIZE**2)
    dN_av        = np.sqrt(dN_av**2 + (N_av * gain_rel_err)**2)

    ecf        = 1.0 + ct + ap
    N_av_pri   = N_av / ecf
    dECF_vbd   = np.sqrt(ct_err_vbd**2 + ap_err_vbd**2)   # absolute, fraction units
    dECF_total = np.sqrt(dECF_vbd**2 + (ecf * REL_ERR_CT_AP_DIGITIZE)**2)
    dN_av_pri  = N_av_pri * np.sqrt((dN_av / N_av)**2 + (dECF_total / ecf)**2)
    return N_av, dN_av, N_av_pri, dN_av_pri


fig, ax = plt.subplots(figsize=(8, 6))

fig2, (ax2ratio, ax2, ax2r) = plt.subplots(
    3, 1, figsize=(8, 10),
    gridspec_kw={'height_ratios': [1.5, 3, 1]},
    sharex=True,
)
fig2.subplots_adjust(hspace=0.05)

with h5py.File(DATA_FILE, 'r') as f:
    for ov_key, style in OV_STYLE.items():
        if ov_key not in f:
            continue
        grp = f[ov_key]
        ov  = OV_MAP[ov_key]
        sp  = ds.SIPM_PARAMS[ov]
        gain             = sp["gain"]
        pde              = sp["effective_pde"]
        ct               = sp["crosstalk"]
        ap               = sp["afterpulsing"]
        I_dark_sipm      = sp["dark_current_measured_A"]
        gain_err_vbd     = sp["gain_err_vbd"]
        ct_err_vbd       = sp["crosstalk_err_vbd"]
        ap_err_vbd       = sp["afterpulsing_err_vbd"]

        # --- plateau beam sigmas (saved by fit_scan.py) ---
        if 'plateau_sigma_x_mm' in grp.attrs:
            sigma_x_m = float(grp.attrs['plateau_sigma_x_mm']) * 1e-3
            sigma_y_m = float(grp.attrs['plateau_sigma_y_mm']) * 1e-3
            print(f"{ov_key}: sigma_x={sigma_x_m*1e3:.3f} mm  sigma_y={sigma_y_m*1e3:.3f} mm")
        else:
            sigma_x_m = sigma_y_m = None
            print(f"{ov_key}: plateau_sigma not in HDF5 — run fit_scan.py first; theory fits skipped")

        N_ph_list, N_ph_err_list         = [], []
        N_av_list, N_av_err_list         = [], []
        N_pri_list, N_pri_err_list       = [], []
        N_sipm_list, N_sipm_err_list     = [], []
        I_pd_list, I_pd_err_list         = [], []
        I_sipm_list, I_sipm_err_list     = [], []

        for run_name in sorted(grp.keys()):
            run = grp[run_name]
            if 'center_scan' not in run:
                continue
            row = run['center_scan'][0]

            I_pd   = row[COL['pd_current']]
            I_pd_s = row[COL['pd_stderr']]
            I_sipm = row[COL['sipm_current']]
            I_si_s = row[COL['sipm_stderr']]

            k_sys  = ds.keithley_current_accuracy(I_pd)
            si_sys = ds.siglent_current_accuracy(I_sipm)

            N_ph, N_ph_err, N_sipm, N_sipm_err = photons_and_err(I_pd, I_pd_s, k_sys)
            N_av, N_av_err, N_pri, N_pri_err = avalanches_and_err(
                I_sipm, I_si_s, si_sys, gain, I_dark_sipm, pde, ct, ap,
                gain_err_vbd, ct_err_vbd, ap_err_vbd)

            N_ph_list.append(N_ph);         N_ph_err_list.append(N_ph_err)
            N_av_list.append(N_av);         N_av_err_list.append(N_av_err)
            N_pri_list.append(N_pri);       N_pri_err_list.append(N_pri_err)
            N_sipm_list.append(N_sipm);     N_sipm_err_list.append(N_sipm_err)
            I_pd_list.append(I_pd);         I_pd_err_list.append(np.sqrt(I_pd_s**2 + k_sys**2))
            I_sipm_list.append(I_sipm);     I_sipm_err_list.append(np.sqrt(I_si_s**2 + si_sys**2))

        N_ph      = np.array(N_ph_list);       N_ph_err    = np.array(N_ph_err_list)
        N_av_arr  = np.array(N_av_list);       N_av_err_arr = np.array(N_av_err_list)
        N_pri     = np.array(N_pri_list);      N_pri_err   = np.array(N_pri_err_list)
        N_sipm    = np.array(N_sipm_list);     N_sipm_err  = np.array(N_sipm_err_list)
        I_pd_arr  = np.array(I_pd_list);       I_pd_err_arr  = np.array(I_pd_err_list)
        I_si_arr  = np.array(I_sipm_list);     I_si_err_arr  = np.array(I_sipm_err_list)

        order = np.argsort(N_ph)
        if ov_key == 'OVfour':
            order = order[:-3]

        # Figure 1 uses its own sort — all points, indexed within I_pd_arr
        order_raw = np.argsort(I_pd_arr)

        # --- Figure 1: raw I_sipm vs I_pd ---
        ax.errorbar(
            I_pd_arr[order_raw] * 1e9, I_si_arr[order_raw] * 1e6,
            xerr=I_pd_err_arr[order_raw] * 1e9, yerr=I_si_err_arr[order_raw] * 1e6,
            fmt=style['marker'], color=style['color'],
            label=style['label'],
            markersize=5, linewidth=0.8, capsize=3, elinewidth=0.8,
        )

        # N_incident / N_fired arrays for Figure 2 (independent of sigma)
        R_gamma     = N_sipm[order]           # photons/s on SiPM face
        R_gamma_err = N_sipm_err[order]
        R_obs       = N_pri[order]            # primary avalanches/s (ECF corrected)
        R_err       = N_pri_err[order]
        ecf         = 1.0 + ct + ap

        ax2.errorbar(
            R_gamma, R_obs,
            xerr=R_gamma_err, yerr=R_err,
            fmt=style['marker'], color=style['color'],
            label=style['label'],
            markersize=5, linewidth=0.8, capsize=3, elinewidth=0.8,
        )

        # ratio N_fired / N_incident — approaches PDE in the unsaturated regime
        ratio_mask = (R_gamma > 0) & (R_obs > 0)
        ratio     = np.where(ratio_mask, R_obs / R_gamma, np.nan)
        ratio_err = np.where(ratio_mask,
                             ratio * np.sqrt((R_err / np.where(R_obs > 0, R_obs, 1))**2
                                             + (R_gamma_err / R_gamma)**2),
                             np.nan)
        ax2ratio.errorbar(
            R_gamma[ratio_mask], ratio[ratio_mask],
            xerr=R_gamma_err[ratio_mask], yerr=ratio_err[ratio_mask],
            fmt=style['marker'], color=style['color'],
            label=style['label'],
            markersize=5, linewidth=0.8, capsize=3, elinewidth=0.8,
        )
        ax2ratio.axhline(pde, color=style['color'], linewidth=1.0, linestyle='--',
                         label=f"{style['label']} PDE$_{{\\rm DS}}$ = {pde:.3f}")

        if sigma_x_m is not None:
            valid2 = (R_gamma > 0) & (R_obs > 0) & np.isfinite(R_err) & (R_err > 0)
            print(f"{ov_key}: {valid2.sum()} / {len(R_gamma)} points pass validity cut")

            if valid2.sum() >= 2:
                # Free parameters: a = tau/tau_0, pde_fit (PDE; datasheet value as p0)
                try:
                    popt2, pcov2 = curve_fit(
                        lambda n, a, pde_fit: Nfired_func(n, sigma_x_m, sigma_y_m, pde_fit, a * TAU_0_S),
                        R_gamma[valid2], R_obs[valid2],
                        sigma=R_err[valid2], absolute_sigma=True,
                        p0=[1.0, pde],
                        bounds=([0.01, 0.01], [200.0, 0.99]),
                        maxfev=10000,
                    )
                    perr2     = np.sqrt(np.diag(pcov2))
                    a_fit2    = float(popt2[0]);  a_err2   = float(perr2[0])
                    pde_fit2  = float(popt2[1]);  pde_err2 = float(perr2[1])
                    reset_fit = a_fit2 * TAU_0_S

                    n_fine  = np.geomspace(R_gamma[valid2].min(), R_gamma[valid2].max(), 300)
                    nf_fine = Nfired_func(n_fine, sigma_x_m, sigma_y_m, pde_fit2, reset_fit)
                    ax2.plot(n_fine, nf_fine, color=style['color'], linewidth=1.6,
                             label=(rf"{style['label']} fit  "
                                    rf"$a={a_fit2:.2f}\pm{a_err2:.2f}$  ($\tau={reset_fit*1e9:.0f}$ ns)"
                                    rf"  PDE$={pde_fit2:.3f}\pm{pde_err2:.3f}$"))

                    ax2ratio.plot(n_fine, nf_fine / n_fine, color=style['color'], linewidth=1.4,
                                 label=rf"{style['label']} model  PDE$_{{fit}}={pde_fit2:.3f}\pm{pde_err2:.3f}$")

                    nf_pred = Nfired_func(R_gamma[valid2], sigma_x_m, sigma_y_m, pde_fit2, reset_fit)
                    resid   = (R_obs[valid2] - nf_pred) / R_err[valid2]
                    chi2r   = float(np.sum(resid**2) / max(valid2.sum() - 2, 1))
                    ax2r.errorbar(
                        R_gamma[valid2], resid,
                        xerr=R_gamma_err[valid2], yerr=np.ones_like(resid),
                        fmt=style['marker'], color=style['color'],
                        markersize=5, linewidth=0.8, capsize=3, elinewidth=0.8,
                        label=f"{style['label']}  chi2/dof={chi2r:.2f}",
                    )
                    print(f"{ov_key} fit: a = {a_fit2:.3f} +/- {a_err2:.3f} "
                          f"(tau = {reset_fit*1e9:.1f} ns),  "
                          f"PDE = {pde_fit2:.4f} +/- {pde_err2:.4f} "
                          f"(DS: {pde:.4f}),  chi2/dof = {chi2r:.2f}")
                except Exception as exc:
                    print(f"{ov_key}: fit failed — {exc}")


ax.set_xlabel('Photodiode current (nA)')
ax.set_ylabel('SiPM current (µA)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Raw IV: SiPM current vs photodiode current')
ax.legend(fontsize=8)
ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'raw_iv.png'), dpi=150)

# --- Saturation curve figure ---
ax2.set_ylabel('$N_{\\rm fired}$ (primary avalanches / s, ECF corrected)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
ax2.set_title('SiPM saturation curve: $N_{\\rm fired}$ vs $N_{\\rm incident}$')

ax2ratio.set_ylabel('$N_{\\rm fired}\\,/\\,N_{\\rm incident}$')
ax2ratio.set_xscale('log')
ax2ratio.legend(fontsize=7)
ax2ratio.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

ax2r.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax2r.set_xlabel('$N_{\\rm incident}$ (photons / s on SiPM face)')
ax2r.set_ylabel('Residuals (σ)')
ax2r.legend(fontsize=7)
ax2r.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

fig2.tight_layout()
fig2.savefig(os.path.join(PLOT_DIR, 'saturation_curve.png'), dpi=150)

plt.show()
