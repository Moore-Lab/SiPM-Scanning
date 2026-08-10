"""
rates.py
--------
Convert measured currents into the physical quantities the saturation model
predicts, with the uncertainty budget separated into correlated and
uncorrelated parts.

Conversion chain
----------------
Reference photodiode (linear over the whole range, 0 V bias):

    P_pd    = I_pd / R_eff                      optical power on the PD (W)
    N_pd    = P_pd / E_photon                   photon rate on the PD (1/s)
    N_total = N_pd / f_pd                       rate at the beamsplitter input
    N_sipm  = N_total * f_sipm                  rate on the SiPM face

SiPM:

    N_total_av = (I_sipm - I_dark) / (G e)      all avalanches
    N_primary  = N_total_av / (1 + CT + AP)     photon-triggered only

Why the split matters
---------------------
The previous version of this analysis put every uncertainty — including the
gain, the PDE, the crosstalk/afterpulsing correction, the photodiode
responsivity and the beamsplitter ratio — into the per-point sigma used by the
fit.  Those are calibration constants: they move every data point coherently,
they are not point-to-point scatter, and inflating sigma_i with them drives
chi2/dof far below one (0.17 was observed) while making the fit insensitive to
real structure in the residuals.

Here the budget is split:

  * UNCORRELATED (goes into the fit sigma): the statistical standard error of
    the repeated readings at each point, plus the instrument offset/resolution
    term, which is independent per reading.

  * CORRELATED (evaluated by shifting and refitting): instrument gain errors
    ("% of reading"), SiPM gain, the ECF = 1 + CT + AP correction, the
    photodiode responsivity calibration, the beamsplitter ratio, the effective
    PDE, and the fitted beam widths.

The fit therefore returns a statistical uncertainty with a meaningful chi2,
and each correlated term is propagated as a separate systematic by refitting
with that term shifted by +/- 1 sigma.  See `SYSTEMATICS`.
"""

import numpy as np
import scipy.constants as const

import parse_datasheets as ds

__all__ = [
    "COLS", "COL", "OV_MAP",
    "MeasurementSet", "load_overvoltage", "SYSTEMATICS", "Systematic",
]

# Column layout of the raw scan arrays stored in the HDF5 file
COLS = ['x', 'y', 'sipm_current', 'sipm_std', 'sipm_stderr',
        'sipm_time', 'pd_current', 'pd_std', 'pd_stderr', 'pd_time']
COL = {name: i for i, name in enumerate(COLS)}

# HDF5 group name -> overvoltage
OV_MAP = {'OVfive': 2.5, 'OVfour': 4.0}

E_CHARGE = const.e
E_PHOTON = const.h * const.c / (ds.LASER_D405["wavelength_typ_nm"] * 1e-9)


# ---------------------------------------------------------------------------
# Correlated systematics
# ---------------------------------------------------------------------------

class Systematic:
    """
    One correlated (normalisation-like) uncertainty.

    `target` names what the shift applies to:
        'x'      – multiplies the incident photon rate
        'y'      – multiplies the measured primary avalanche rate
        'pde'    – multiplies the PDE handed to the model
        'sigma'  – multiplies both beam widths
    `rel` is the 1-sigma relative size.
    """

    def __init__(self, name, target, rel, note=""):
        self.name = name
        self.target = target
        self.rel = rel
        self.note = note

    def __repr__(self):
        return f"<Systematic {self.name}: {self.target} {self.rel:+.3%}>"


def build_systematics(ov):
    """Correlated systematics for one overvoltage setting."""
    sp = ds.SIPM_PARAMS[ov]
    ecf = 1.0 + sp["crosstalk"] + sp["afterpulsing"]
    ecf_err = np.sqrt(sp["crosstalk_err_vbd"] ** 2 + sp["afterpulsing_err_vbd"] ** 2
                      + (ecf * 0.05) ** 2)
    return [
        Systematic("sipm_gain", "y",
                   np.hypot(sp["gain_err_vbd"] / sp["gain"], 0.02),
                   "V_BD propagation + residual graph digitisation"),
        Systematic("ecf", "y", ecf_err / ecf,
                   "crosstalk + afterpulsing correction (1+CT+AP)"),
        Systematic("pd_responsivity", "x", 0.02,
                   "FDS100-CAL cert 25031152700: 2.9% k=2 at 405 nm, plus spectral model"),
        Systematic("beamsplitter", "x", 0.02,
                   "measured 91:9 split ratio"),
        Systematic("dmm_gain", "y", 0.0005,
                   "Siglent SDM3045X % of reading"),
        Systematic("picoammeter_gain", "x", 0.0015,
                   "Keithley 6487 % of reading"),
        Systematic("pde", "pde",
                   sp["effective_pde_err_vbd"] / sp["effective_pde"],
                   "V_BD propagation on the laser-weighted effective PDE"),
        Systematic("beam_width", "sigma", 0.02,
                   "razor-blade plateau sigma; stored fit error is not "
                   "believable (0.03%), 2% assumed from scan reproducibility"),
    ]


SYSTEMATICS = build_systematics


# ---------------------------------------------------------------------------
# Measurement container
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Point-to-point reproducibility floor
#
# The standard error of the 50 averaged readings at each point is 0.03-0.1%,
# which is NOT the true point-to-point uncertainty.  Each measurement
# re-positions the beam with the motorised stage and reads the two instruments
# ~10.5 s apart, so stage repeatability and laser drift between the SiPM and
# photodiode readings both enter.
#
# Two ND settings were measured twice at 4.0 V OV, which bounds this directly:
#
#   ND = 1.5 : I_pd repeats to 0.44%, I_sipm to 0.12%, RATIO to 0.32%
#   ND = 2.8 : I_pd repeats to 14.8%, I_sipm to 12.9%, RATIO to 1.91%
#
# The large single-instrument spread at ND 2.8 with a stable ratio is the
# ratiometric method working as intended: the laser drifted ~15% between the
# two runs and the reference photodiode tracked it.  What does NOT cancel is
# the residual 0.3-1.9% on the ratio, and that is the floor applied here.
#
# This is an empirical bound from two repeats, not a well-sampled estimate.
# Repeating a handful of ND settings 5-10 times would pin it down properly and
# is the single cheapest improvement available to this measurement.
# ---------------------------------------------------------------------------

REPRODUCIBILITY_FLOOR = 0.010   # 1% relative, applied to y


class MeasurementSet:
    """
    One overvoltage setting: incident photon rate vs primary avalanche rate,
    with uncorrelated errors and the list of correlated systematics.

    Attributes
    ----------
    x, y            : incident photon rate (1/s), primary avalanche rate (1/s)
    x_err, y_err    : UNCORRELATED 1-sigma errors only, including the
                      reproducibility floor
    i_sipm          : SiPM current per point (A) — used for quality cuts
    nd              : ND filter value per point (provenance)
    pde, sigma_x, sigma_y, n_cells, pitch : model inputs
    systematics     : list of Systematic
    """

    def __init__(self, ov_key, ov, x, y, x_err, y_err, nd, i_sipm,
                 pde, sigma_x, sigma_y, dropped=0, floor=REPRODUCIBILITY_FLOOR):
        order = np.argsort(x)
        self.ov_key, self.ov = ov_key, ov
        self.x, self.y = x[order], y[order]
        self.nd = nd[order]
        self.i_sipm = i_sipm[order]
        self.x_err = x_err[order]
        self.y_err = np.hypot(y_err[order], floor * self.y)
        self.y_err_readout = y_err[order]      # before the floor, for reporting
        self.floor = floor
        self.pde = pde
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.dropped = dropped
        self.pitch = ds.SIPM_60035_GENERAL["microcell_pitch_um"] * 1e-6
        self.n_cells = ds.SIPM_60035_GENERAL["n_microcells"]
        self.systematics = build_systematics(ov)

    def __len__(self):
        return len(self.x)

    @property
    def label(self):
        return f"{self.ov:.1f} V OV"

    def current_cut(self, max_current_A):
        """
        Return a copy keeping only points below a given SiPM current.

        The device is driven to 22 mA (2.5 V OV) and 59 mA (4.0 V OV) at the
        top of the sweep, four orders of magnitude above the datasheet
        operating current.  Constant gain cannot be assumed there — see the
        module-level discussion in the analysis driver.
        """
        m = self.i_sipm <= max_current_A
        out = MeasurementSet(self.ov_key, self.ov, self.x[m], self.y[m],
                             self.x_err[m], self.y_err_readout[m], self.nd[m],
                             self.i_sipm[m], self.pde, self.sigma_x,
                             self.sigma_y, self.dropped, self.floor)
        return out


def load_overvoltage(h5file, ov_key, drop_brightest=0):
    """
    Build a MeasurementSet from one HDF5 overvoltage group.

    Parameters
    ----------
    drop_brightest : int
        Number of highest-flux points to discard.  The original analysis
        dropped the three brightest 4.0 V points with a bare `order[:-3]` and
        no justification recorded.  It is exposed here so the cut is explicit,
        and the default is to keep everything.
    """
    grp = h5file[ov_key]
    ov = OV_MAP[ov_key]
    sp = ds.SIPM_PARAMS[ov]

    if 'plateau_sigma_x_mm' not in grp.attrs:
        raise KeyError(f"{ov_key}: plateau beam widths missing — run fit_scan.py first")
    sigma_x = float(grp.attrs['plateau_sigma_x_mm']) * 1e-3
    sigma_y = float(grp.attrs['plateau_sigma_y_mm']) * 1e-3

    gain = sp["gain"]
    ecf = 1.0 + sp["crosstalk"] + sp["afterpulsing"]
    i_dark = sp["dark_current_measured_A"]

    r_eff = ds.PD_PARAMS["effective_responsivity_A_per_W"]
    f_pd = ds.PD_PARAMS["pd_fraction"]
    f_sipm = ds.PD_PARAMS["sipm_fraction"]

    x, y, x_err, y_err, nd, i_sipm = [], [], [], [], [], []

    for run_name in sorted(grp.keys()):
        run = grp[run_name]
        if 'center_scan' not in run:
            continue
        row = run['center_scan'][0]

        i_pd, i_pd_stat = row[COL['pd_current']], row[COL['pd_stderr']]
        i_si, i_si_stat = row[COL['sipm_current']], row[COL['sipm_stderr']]

        # --- incident photon rate on the SiPM face -------------------------
        _, pd_offset = ds.keithley_accuracy_split(i_pd)
        di_pd = np.hypot(i_pd_stat, pd_offset)          # uncorrelated only
        n_sipm = (i_pd / r_eff) / E_PHOTON / f_pd * f_sipm

        # --- primary avalanche rate ---------------------------------------
        _, si_offset = ds.siglent_accuracy_split(i_si)
        di_si = np.hypot(i_si_stat, si_offset)          # uncorrelated only
        n_av = (i_si - i_dark) / (gain * E_CHARGE)
        n_pri = n_av / ecf

        x.append(n_sipm)
        x_err.append(n_sipm * di_pd / abs(i_pd))
        y.append(n_pri)
        y_err.append(n_pri * di_si / abs(i_si - i_dark))
        nd.append(float(run.attrs['nd']))
        i_sipm.append(i_si)

    x, y = np.array(x), np.array(y)
    x_err, y_err = np.array(x_err), np.array(y_err)
    nd, i_sipm = np.array(nd), np.array(i_sipm)

    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (y_err > 0)
    x, y, x_err, y_err = x[keep], y[keep], x_err[keep], y_err[keep]
    nd, i_sipm = nd[keep], i_sipm[keep]

    dropped = 0
    if drop_brightest:
        order = np.argsort(x)
        keep_idx = order[:-drop_brightest]
        x, y = x[keep_idx], y[keep_idx]
        x_err, y_err = x_err[keep_idx], y_err[keep_idx]
        nd, i_sipm = nd[keep_idx], i_sipm[keep_idx]
        dropped = drop_brightest

    return MeasurementSet(ov_key, ov, x, y, x_err, y_err, nd, i_sipm,
                          sp["effective_pde"], sigma_x, sigma_y, dropped)
