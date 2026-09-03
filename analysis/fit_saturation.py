"""
fit_saturation.py
-----------------
Fit the Gaussian lattice-model saturation curve to the measured data.

The model is

    <R> = (2 pi sigma_x sigma_y / (dx dy tau)) * [gamma + ln u - Ei(-u)],
    u   =  PDE * R_gamma * dx dy tau / (2 pi sigma_x sigma_y)

Everything except the SPAD reset time tau is fixed from the datasheet or from
an independent measurement, so tau is the only free parameter.  A two-parameter
variant that also floats the PDE is provided as a cross-check: if the
one-parameter fit is good and the floated PDE lands on the datasheet value,
the model is doing real work rather than absorbing a normalisation.

Two details that the previous analysis did not handle:

1.  x-axis uncertainty.  The incident photon rate carries a few-percent error
    that was plotted but never entered the fit.  Here it is folded in with the
    effective-variance (Orear) method,

        sigma_eff^2 = sigma_y^2 + (d<R>/dR_gamma)^2 sigma_x^2

    iterated to convergence using the closed-form derivative.

2.  Correlated systematics.  Calibration constants move the whole curve
    coherently and must not sit in the per-point sigma.  They are propagated
    by refitting with each term shifted by +/- 1 sigma and taking the induced
    change in tau, reported as a separate systematic uncertainty with a
    per-source breakdown.
"""

import copy
import numpy as np
from scipy.optimize import curve_fit

import saturation_model as sm
import parse_datasheets as ds

__all__ = ["FitResult", "fit_tau", "systematic_breakdown", "TAU_0"]

TAU_0 = ds.SIPM_60035_PERFORMANCE["microcell_recharge_tau_ns"] * 1e-9   # 50 ns


class FitResult:
    """Outcome of one saturation-curve fit."""

    def __init__(self, tau, tau_stat, pde, pde_stat, chi2, ndof,
                 residuals, sigma_eff, free_pde, n_points):
        self.tau = tau
        self.tau_stat = tau_stat
        self.pde = pde
        self.pde_stat = pde_stat
        self.chi2 = chi2
        self.ndof = ndof
        self.residuals = residuals
        self.sigma_eff = sigma_eff
        self.free_pde = free_pde
        self.n_points = n_points
        self.tau_syst = np.nan
        self.syst_terms = {}

    @property
    def chi2_red(self):
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    @property
    def tau_multiple(self):
        """tau in units of the datasheet recharge time tau_0 = 50 ns."""
        return self.tau / TAU_0

    def summary(self):
        s = (f"tau = {self.tau * 1e9:.1f} +/- {self.tau_stat * 1e9:.1f} (stat)")
        if np.isfinite(self.tau_syst):
            s += f" +/- {self.tau_syst * 1e9:.1f} (syst)"
        s += f" ns  =  {self.tau_multiple:.2f} tau_0"
        s += f"   PDE = {self.pde:.4f}"
        if self.free_pde:
            s += f" +/- {self.pde_stat:.4f} (fitted)"
        else:
            s += " (fixed)"
        s += f"   chi2/dof = {self.chi2_red:.2f}  ({self.ndof} dof)"
        return s


def _model(x, tau, pde, ms):
    return sm.gaussian_rate(x, tau, pde, ms.sigma_x, ms.sigma_y, ms.pitch)


def fit_tau(ms, free_pde=False, n_iter=4, x=None, y=None, pde=None,
            sigma_scale=1.0):
    """
    Fit the saturation curve for one MeasurementSet.

    Parameters
    ----------
    ms          : rates.MeasurementSet
    free_pde    : also float the PDE (cross-check fit)
    n_iter      : effective-variance iterations
    x, y        : optional overrides (used when applying a systematic shift)
    pde         : optional PDE override
    sigma_scale : optional multiplier on both beam widths

    Returns
    -------
    FitResult
    """
    x = ms.x if x is None else x
    y = ms.y if y is None else y
    pde_fixed = ms.pde if pde is None else pde

    shifted = copy.copy(ms)
    shifted.sigma_x = ms.sigma_x * sigma_scale
    shifted.sigma_y = ms.sigma_y * sigma_scale

    n_par = 2 if free_pde else 1
    sigma_eff = ms.y_err.copy()

    popt = None
    for _ in range(n_iter):
        if free_pde:
            def f(xx, a, p):
                return _model(xx, a * TAU_0, p, shifted)
            p0 = [popt[0] if popt is not None else 4.0,
                  popt[1] if popt is not None else pde_fixed]
            bounds = ([0.01, 0.01], [200.0, 0.99])
        else:
            def f(xx, a):
                return _model(xx, a * TAU_0, pde_fixed, shifted)
            p0 = [popt[0] if popt is not None else 4.0]
            bounds = ([0.01], [200.0])

        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=sigma_eff,
                               absolute_sigma=True, bounds=bounds, maxfev=20000)

        # Update the effective variance with the current best fit
        tau_now = popt[0] * TAU_0
        pde_now = popt[1] if free_pde else pde_fixed
        dydx = sm.gaussian_rate_derivative(x, tau_now, pde_now,
                                           shifted.sigma_x, shifted.sigma_y,
                                           ms.pitch)
        sigma_eff = np.sqrt(ms.y_err ** 2 + (dydx * ms.x_err) ** 2)

    perr = np.sqrt(np.diag(pcov))
    tau = popt[0] * TAU_0
    pde_out = popt[1] if free_pde else pde_fixed
    pde_err = perr[1] if free_pde else np.nan

    resid = (y - f(x, *popt)) / sigma_eff
    chi2 = float(np.sum(resid ** 2))
    ndof = len(x) - n_par

    return FitResult(tau, perr[0] * TAU_0, pde_out, pde_err,
                     chi2, ndof, resid, sigma_eff, free_pde, len(x))


def systematic_breakdown(ms, free_pde=False):
    """
    Propagate every correlated systematic by refitting with it shifted by
    +/- 1 sigma.  Returns (total_syst_on_tau, {name: delta_tau}).

    Sources are treated as mutually independent and combined in quadrature.
    The per-source delta is the half-spread of the +1 sigma and -1 sigma fits,
    which symmetrises any mild non-linearity.
    """
    terms = {}
    for syst in ms.systematics:
        taus = []
        for sign in (+1.0, -1.0):
            factor = 1.0 + sign * syst.rel
            kwargs = dict(free_pde=free_pde)
            if syst.target == 'x':
                kwargs['x'] = ms.x * factor
            elif syst.target == 'y':
                kwargs['y'] = ms.y * factor
            elif syst.target == 'pde':
                kwargs['pde'] = ms.pde * factor
            elif syst.target == 'sigma':
                kwargs['sigma_scale'] = factor
            else:
                raise ValueError(f"unknown systematic target {syst.target!r}")
            taus.append(fit_tau(ms, **kwargs).tau)
        terms[syst.name] = abs(taus[0] - taus[1]) / 2.0

    total = float(np.sqrt(sum(v ** 2 for v in terms.values())))
    return total, terms


def fit_with_systematics(ms, free_pde=False):
    """Convenience wrapper: central fit plus the full systematic budget."""
    res = fit_tau(ms, free_pde=free_pde)
    res.tau_syst, res.syst_terms = systematic_breakdown(ms, free_pde=free_pde)
    return res


# ---------------------------------------------------------------------------
# Fit with an arbitrary measured profile instead of the Gaussian closed form
# ---------------------------------------------------------------------------

def fit_tau_profile(ms, flux, dA, free_pde=False, n_iter=4, x=None, y=None,
                    pde=None):
    """
    Same fit as `fit_tau`, but the model is the numerical integral of the
    continuum model over a supplied flux map rather than the Gaussian closed
    form. This is the path for the camera-measured beam profile.

    `flux` is the normalised flux density on a pixel grid over the device
    (integral = 1), `dA` the pixel area. The x-error is folded in with the
    effective-variance method using a numerical derivative.
    """
    x = ms.x if x is None else x
    y = ms.y if y is None else y
    pde_fixed = ms.pde if pde is None else pde
    n_par = 2 if free_pde else 1
    sigma_eff = ms.y_err.copy()

    def model(xx, tau, p):
        return sm.continuum_rate(xx, tau, p, ms.pitch, flux, dA)

    popt = None
    for _ in range(n_iter):
        if free_pde:
            def f(xx, a, p):
                return model(xx, a * TAU_0, p)
            p0 = [popt[0] if popt is not None else 4.0,
                  popt[1] if popt is not None else pde_fixed]
            bounds = ([0.01, 0.01], [200.0, 0.99])
        else:
            def f(xx, a):
                return model(xx, a * TAU_0, pde_fixed)
            p0 = [popt[0] if popt is not None else 4.0]
            bounds = ([0.01], [200.0])

        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=sigma_eff,
                               absolute_sigma=True, bounds=bounds, maxfev=20000)

        tau_now = popt[0] * TAU_0
        pde_now = popt[1] if free_pde else pde_fixed
        h = 1.01
        dydx = (model(x * h, tau_now, pde_now) - model(x / h, tau_now, pde_now)) \
            / (x * (h - 1.0 / h))
        sigma_eff = np.sqrt(ms.y_err ** 2 + (dydx * ms.x_err) ** 2)

    perr = np.sqrt(np.diag(pcov))
    tau = popt[0] * TAU_0
    pde_out = popt[1] if free_pde else pde_fixed
    pde_err = perr[1] if free_pde else np.nan
    resid = (y - f(x, *popt)) / sigma_eff
    chi2 = float(np.sum(resid ** 2))
    return FitResult(tau, perr[0] * TAU_0, pde_out, pde_err, chi2,
                     len(x) - n_par, resid, sigma_eff, free_pde, len(x))
