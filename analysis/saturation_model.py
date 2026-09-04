"""
saturation_model.py
-------------------
Closed-form saturation response of a SiPM under non-uniform illumination.

These are the worked examples of the lattice model.  Starting from the
single-SPAD Poisson response, the expected number of avalanches over a
space-time lattice of cells (dx, dy) and reset windows dt = tau is

    <N> = SUM_ijk [ 1 - exp( -PDE * U_ijk ) ]

which in the dense-lattice (continuum) limit becomes

    <N> = INT dx dy dt / (dx dy dt) * [ 1 - exp( -PDE * Phi(x,y,t) dx dy dt ) ]

for an incident photon flux density Phi.  Everything below is a special case.

A structural consequence used throughout: writing the integral in layer-cake
form shows that <N> depends on the profile ONLY through its level-set measure
a(phi) = area where Phi > phi.  Two geometrically unrelated beams with the
same a(phi) saturate identically — see `gaussian_rate` versus
`exponential_strip_rate`.

Note that the response depends on the cell pitch and the reset time only
through the product (pitch^2 * tau).  The saturation scale is a cell-recovery
area-time, which is why a mis-stated pitch maps directly onto the fitted tau.

All functions take and return SI units unless stated otherwise.
"""

import numpy as np
from scipy.special import expi

EULER_GAMMA = float(np.euler_gamma)

__all__ = [
    "ei_kernel",
    "gaussian_rate",
    "uniform_rate",
    "exponential_strip_rate",
    "gaussian_pulse_response",
    "saturation_parameter",
    "effective_cells",
]


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------

def ei_kernel(c):
    """
    The dimensionless saturation kernel  K(c) = gamma + ln(c) - Ei(-c).

    Arises for any profile whose level-set measure is logarithmic in the flux
    (a 2D Gaussian, a 1D two-sided exponential, ...).  Limits:

        c << 1 :  K -> c        (linear, unsaturated)
        c >> 1 :  K -> gamma + ln c   (logarithmic dynamic range)
    """
    c = np.maximum(np.asarray(c, dtype=float), 1e-300)
    return EULER_GAMMA + np.log(c) - expi(-c)


def saturation_parameter(R_gamma, tau, pde, sigma_x, sigma_y, pitch):
    """
    Dimensionless saturation parameter for a centred Gaussian beam,

        u = PDE * R_gamma * dx dy tau / (2 pi sigma_x sigma_y)

    i.e. the mean number of detected photons in the brightest cell during one
    reset window.  u ~ 1 marks the onset of saturation.
    """
    return pde * R_gamma * pitch ** 2 * tau / (2.0 * np.pi * sigma_x * sigma_y)


def effective_cells(sigma_x, sigma_y, pitch, tau):
    """
    Prefactor A = 2 pi sigma_x sigma_y / (dx dy tau), the effective number of
    illuminated cells per unit time.  Units: 1/s.
    """
    return 2.0 * np.pi * sigma_x * sigma_y / (pitch ** 2 * tau)


# ---------------------------------------------------------------------------
# Static worked examples
# ---------------------------------------------------------------------------

def gaussian_rate(R_gamma, tau, pde, sigma_x, sigma_y, pitch):
    """
    Primary avalanche rate for a static, centred 2D Gaussian beam.

        <R> = (2 pi sigma_x sigma_y / (dx dy tau)) * [gamma + ln u - Ei(-u)]

    Parameters
    ----------
    R_gamma  : photon rate incident on the SiPM face (photons/s)
    tau      : SPAD reset time (s)
    pde      : photon detection efficiency (dimensionless)
    sigma_x, sigma_y : Gaussian beam widths (m)
    pitch    : microcell tiling period (m) — NOT the active dimension

    Returns
    -------
    Primary avalanche rate (avalanches/s); crosstalk and afterpulsing excluded.

    Assumes the active area is large compared with the beam, so the Gaussian
    may be integrated over an infinite plane.  For the MicroFJ-60035 with
    sigma ~ 0.58 mm the device half-width is >5 sigma and the truncation is
    negligible.
    """
    A = effective_cells(sigma_x, sigma_y, pitch, tau)
    u = saturation_parameter(R_gamma, tau, pde, sigma_x, sigma_y, pitch)
    return A * ei_kernel(u)


def gaussian_rate_derivative(R_gamma, tau, pde, sigma_x, sigma_y, pitch):
    """
    d<R>/dR_gamma for `gaussian_rate`, in closed form.

    With K(u) = gamma + ln u - Ei(-u) one has K'(u) = (1 - e^-u)/u, so

        d<R>/dR_gamma = A * (1 - exp(-u)) / R_gamma

    which tends to PDE in the unsaturated limit, as it must.  Used to fold the
    x-axis uncertainty into an effective variance during fitting.
    """
    A = effective_cells(sigma_x, sigma_y, pitch, tau)
    u = saturation_parameter(R_gamma, tau, pde, sigma_x, sigma_y, pitch)
    return A * (1.0 - np.exp(-u)) / R_gamma


def uniform_rate(R_gamma, tau, pde, n_cells):
    """
    Primary avalanche rate for uniform illumination — the standard
    saturation law (Gruber et al., NIM A 737 (2014) 11).

        <R> = (N_cells / tau) * [1 - exp(-PDE * R_gamma * tau / N_cells)]

    Recovered from the lattice sum with a flat profile; included so the
    framework can be checked against the known result.
    """
    return (n_cells / tau) * (1.0 - np.exp(-pde * R_gamma * tau / n_cells))


def exponential_strip_rate(R_gamma, tau, pde, decay_length, strip_length, pitch):
    """
    Primary avalanche rate for a 1D two-sided exponential profile,
    Phi ~ exp(-|x| / decay_length), uniform over a strip of length L_y.

    Included to demonstrate level-set universality: this profile has the same
    logarithmic a(phi) as a 2D Gaussian and therefore obeys the *same* Ei law,
    with 2 pi sigma_x sigma_y replaced by 2 * L_y * decay_length.  Verified to
    agree with `gaussian_rate` to machine precision when the two level-set
    measures are matched.
    """
    area = 2.0 * strip_length * decay_length
    A = area / (pitch ** 2 * tau)
    u = pde * R_gamma * pitch ** 2 * tau / area
    return A * ei_kernel(u)


# ---------------------------------------------------------------------------
# Temporal worked example
# ---------------------------------------------------------------------------

def gaussian_pulse_response(n_photons, tau, pde, sigma_x, sigma_y, pitch,
                            tau_s, n_terms=None):
    """
    Expected avalanche count for a Gaussian beam with an exponentially
    decaying temporal envelope — the scintillation-pulse case.

        Phi(x,y,t) = Gaussian(x,y) * (1/tau_s) * exp(-t/tau_s),  t >= 0

    Because the spatial and temporal dependence factorise, the space-time
    integral is the time-integral of the instantaneous static response:

        <N> = A (tau_s/tau) * INT_0^c0  [gamma + ln w - Ei(-w)] dw / w
            = A (tau_s/tau) * SUM_{k>=1} (-1)^(k+1) c0^k / (k^2 k!)
            = A (tau_s/tau) * c0 * 3F3(1,1,1; 2,2,2; -c0)

    where c0 is the peak saturation parameter at t = 0.  High-flux limit:
    <N> -> A (tau_s/tau) * (gamma + ln c0)^2 / 2 — log-SQUARED dynamic range,
    against log for steady illumination.

    Parameters
    ----------
    n_photons : total photons in the pulse
    tau       : SPAD reset time (s)
    tau_s     : scintillation decay constant (s)
    n_terms   : series truncation; default adapts to c0

    Returns
    -------
    Expected primary avalanche count (dimensionless).

    Notes
    -----
    The series above is the analytic statement, but it is an ALTERNATING series
    whose largest term grows like exp(c0) while the sum stays O(ln^2 c0).  In
    double precision it loses all significance by c0 ~ 50.  This function
    therefore evaluates the equivalent integral form by Gauss-Legendre
    quadrature under the substitution w = c0*exp(-s):

        INT_0^c0 K(w) dw/w  =  INT_0^inf K(c0 e^-s) ds

    whose integrand is smooth and bounded for every c0.  Use
    `gaussian_pulse_series` if you want the literal series (small c0 only).
    """
    A_cells = 2.0 * np.pi * sigma_x * sigma_y / pitch ** 2   # effective cells
    c0 = (pde * n_photons * pitch ** 2 * tau
          / (2.0 * np.pi * sigma_x * sigma_y * tau_s))

    scalar = np.ndim(c0) == 0
    result = A_cells * (tau_s / tau) * _pulse_integral(c0, n_nodes=n_terms)
    return float(np.atleast_1d(result)[0]) if scalar else result


_GL_DEFAULT_NODES = 512


def _pulse_integral(c0, n_nodes=None):
    """
    INT_0^c0 [gamma + ln w - Ei(-w)] dw / w, vectorised over c0.

    Substituting w = c0 exp(-s) maps the integral to INT_0^inf K(c0 e^-s) ds.
    The integrand falls off like c0*exp(-s) once s exceeds ln(c0), so the tail
    beyond ln(c0) + 50 contributes below double precision.
    """
    n_nodes = n_nodes or _GL_DEFAULT_NODES
    c0_arr = np.atleast_1d(np.asarray(c0, dtype=float))
    c0_arr = np.maximum(c0_arr, 1e-300)

    s_max = np.maximum(np.log(c0_arr), 0.0) + 50.0
    x, w = np.polynomial.legendre.leggauss(n_nodes)

    s = 0.5 * s_max[:, None] * (x[None, :] + 1.0)          # map to [0, s_max]
    integrand = ei_kernel(c0_arr[:, None] * np.exp(-s))
    out = 0.5 * s_max * np.sum(integrand * w[None, :], axis=1)
    return float(out[0]) if np.ndim(c0) == 0 else out.reshape(np.shape(c0))


def gaussian_pulse_series(c0, n_terms=None):
    """
    The literal alternating series SUM (-1)^(k+1) c0^k / (k^2 k!).

    Provided so the closed form quoted in the paper can be checked directly.
    Reliable only for c0 <~ 20 in double precision — see
    `gaussian_pulse_response` for the numerically stable evaluation.
    """
    from scipy.special import gammaln
    c0 = float(c0)
    k_max = n_terms or int(4 * c0 + 40)
    k = np.arange(1, k_max + 1, dtype=float)
    log_terms = k * np.log(c0) - 2.0 * np.log(k) - gammaln(k + 1.0)
    signs = np.where(k.astype(int) % 2 == 1, 1.0, -1.0)
    return float(np.sum(signs * np.exp(log_terms)))


# ---------------------------------------------------------------------------
# General case: numerical integration over an arbitrary measured profile
# ---------------------------------------------------------------------------

def continuum_rate(R_gamma, tau, pde, pitch, flux, dA):
    """
    Avalanche rate for an ARBITRARY illumination profile by direct numerical
    integration of the continuum model,

        <R> = SUM_pixels  dA / (pitch^2 tau) * [1 - exp(-PDE R_gamma phi pitch^2 tau)]

    Parameters
    ----------
    R_gamma : incident photon rate on the device (1/s), scalar or array
    flux    : normalised flux density on a grid (integral over the device = 1),
              any shape. Only pixels ON the device should be included.
    dA      : pixel area (m^2); <= pitch^2 and fine enough to resolve the beam

    This is what replaces `gaussian_rate` when the beam is not Gaussian: feed
    it the measured profile and it returns the saturation curve with no
    assumption about the shape. It reproduces `gaussian_rate` to 1e-5 when
    given a Gaussian map (see tests).
    """
    phi = np.asarray(flux, dtype=float).ravel()
    rg = np.atleast_1d(np.asarray(R_gamma, dtype=float))
    out = np.empty_like(rg)
    k = pde * pitch ** 2 * tau
    for i, R in enumerate(rg):
        out[i] = np.sum(-np.expm1(-k * R * phi)) * dA / (pitch ** 2 * tau)
    return out if np.ndim(R_gamma) else float(out[0])


def gaussian_flux_map(sigma_x, sigma_y, xlim, ylim, pixel):
    """
    Normalised Gaussian flux density sampled on a pixel grid covering the
    device. Returns (flux, dA). Useful for testing `continuum_rate` against
    the closed form, and as the template for a measured map.
    """
    nx, ny = int(round(xlim / pixel)), int(round(ylim / pixel))
    x = (np.arange(nx) + 0.5) * pixel - 0.5 * xlim
    y = (np.arange(ny) + 0.5) * pixel - 0.5 * ylim
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi = np.exp(-0.5 * (X / sigma_x) ** 2 - 0.5 * (Y / sigma_y) ** 2)
    phi /= 2.0 * np.pi * sigma_x * sigma_y
    return phi, pixel * pixel


def airy_flux_map(a, xlim, ylim, pixel):
    """
    Airy irradiance I(r) = I0 [2 J1(r/a)/(r/a)]^2 sampled on a pixel grid over
    the device, normalised over the INFINITE plane (integral = 1), so that the
    sum over on-device pixels is the fraction of photons that land on the
    device. That is the right normalisation when the incident rate is defined
    at the SiPM position, as it is here: photons in the Airy wings that miss
    the active area are still counted in R_gamma. Returns (flux, dA).
    """
    from scipy.special import j1
    nx, ny = int(round(xlim / pixel)), int(round(ylim / pixel))
    x = (np.arange(nx) + 0.5) * pixel - 0.5 * xlim
    y = (np.arange(ny) + 0.5) * pixel - 0.5 * ylim
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.hypot(X, Y) / a
    phi = np.ones_like(u)
    m = u > 1e-9
    phi[m] = (2.0 * j1(u[m]) / u[m]) ** 2
    return phi / (4.0 * np.pi * a ** 2), pixel * pixel


# ---------------------------------------------------------------------------
# Pinhole-filtered beam: Gaussian focus truncated by a circular aperture
# ---------------------------------------------------------------------------

def pinhole_intensity(kappa, t, n_nodes=400):
    """
    Far-field intensity of a Gaussian focus clipped by a circular pinhole --
    the beam a spatial filter delivers when the pinhole is not much larger
    than the focused spot. In units of the Airy scale a (kappa = r/a) and the
    truncation ratio t = R_pinhole / w_0:

        A(kappa) = INT_0^1 exp(-t^2 u^2) J0(kappa u) u du,   I = |A|^2

    normalised by Parseval so that INT I 2 pi kappa dkappa = 1. The limit
    t -> 0 is the Airy pattern [2 J1(kappa)/kappa]^2 / (4 pi); t -> infinity
    is a Gaussian. Fitting the razor-blade scans with this family returns
    t -> 0: the pinhole is heavily overfilled and the beam is Airy.
    """
    from scipy.special import j0
    ug, wg = np.polynomial.legendre.leggauss(n_nodes)
    u, wu = 0.5 * (ug + 1.0), 0.5 * wg
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    amp = np.exp(-t * t * u * u) * u * wu
    A = (j0(np.outer(kappa, u)) * amp[None, :]).sum(axis=1)
    norm = (1.0 - np.exp(-2.0 * t * t)) / (4.0 * t * t) if t > 1e-6 else 0.5
    return A * A / (2.0 * np.pi * norm)


def pinhole_flux_map(a, t, xlim, ylim, pixel):
    """Pinhole-filtered beam sampled on the device grid; see airy_flux_map."""
    nx, ny = int(round(xlim / pixel)), int(round(ylim / pixel))
    x = (np.arange(nx) + 0.5) * pixel - 0.5 * xlim
    y = (np.arange(ny) + 0.5) * pixel - 0.5 * ylim
    X, Y = np.meshgrid(x, y, indexing='ij')
    kgrid = np.linspace(0.0, 80.0, 6001)
    Ik = pinhole_intensity(kgrid, t)
    phi = np.interp(np.hypot(X, Y) / a, kgrid, Ik, right=0.0) / (a * a)
    return phi, pixel * pixel
