# Gaussian Illumination: Expected Avalanche Rate Derivation

## Setup

Consider a SiPM with uniform PDE, cell pitch $\Delta x \times \Delta y$, and SPAD reset
time $\tau$. The device is illuminated by a static Gaussian beam with total photon
arrival rate $R_\gamma$ (photons s$^{-1}$) and beam widths $\sigma_x$, $\sigma_y$.

The 2D photon flux density (photons per unit area per unit time) is

$$
\Phi(x,y) = \frac{R_\gamma}{2\pi\sigma_x\sigma_y}
             \exp\!\left(-\frac{x^2}{2\sigma_x^2} - \frac{y^2}{2\sigma_y^2}\right).
$$

## General 3D Lattice Integral

Each spatiotemporal voxel $(\Delta x, \Delta y, \Delta t = \tau)$ receives on average

$$
U(x,y) = \Phi(x,y)\,\Delta x\,\Delta y\,\tau
$$

photons per reset window.  The probability that the SPAD at $(x,y)$ fires in one
reset window is $1 - e^{-\mathrm{PDE}\cdot U(x,y)}$.  Over a long observation
$T = n_t\tau$, the expected total number of avalanches is

$$
\langle N \rangle
= \frac{T}{\tau}
  \int \frac{dx\,dy}{\Delta x\,\Delta y}
  \left[1 - \exp\!\bigl(-\mathrm{PDE}\cdot\Phi(x,y)\,\Delta x\,\Delta y\,\tau\bigr)\right].
$$

## Expected Avalanche Rate

Dividing both sides by $T$ gives the avalanche rate:

$$
\langle R \rangle \equiv \frac{\langle N \rangle}{T}
= \frac{1}{\Delta x\,\Delta y\,\tau}
  \int dx\,dy\;
  \left[1 - \exp\!\left(-\frac{\mathrm{PDE}\cdot R_\gamma\,\Delta x\,\Delta y\,\tau}
                               {2\pi\sigma_x\sigma_y}
                  \exp\!\!\left(-\frac{x^2}{2\sigma_x^2}-\frac{y^2}{2\sigma_y^2}\right)
                  \right)\right].
$$

Define the dimensionless saturation parameter

$$
\boxed{
u \;=\; \frac{\mathrm{PDE}\cdot R_\gamma\cdot\Delta x\,\Delta y\,\tau}{2\pi\sigma_x\sigma_y}.
}
$$

The integrand is then $1 - \exp(-u\,e^{-r^2/2})$ in rescaled coordinates, which
evaluates in closed form via the identity

$$
\int_{-\infty}^{\infty}\!\int_{-\infty}^{\infty}
\left[1 - e^{-c\,\exp\!\left(-\tfrac{x^2}{2\sigma_x^2}-\tfrac{y^2}{2\sigma_y^2}\right)}\right]
dx\,dy
= 2\pi\sigma_x\sigma_y\!\left[\gamma + \ln c - \mathrm{Ei}(-c)\right],
$$

where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant and $\mathrm{Ei}$
is the exponential integral.

Substituting $c = u$:

$$
\boxed{
\langle R \rangle
= \frac{2\pi\sigma_x\sigma_y}{\Delta x\,\Delta y\,\tau}
  \left[\,\gamma + \ln u - \mathrm{Ei}(-u)\,\right],
\qquad
u = \frac{\mathrm{PDE}\cdot R_\gamma\cdot\Delta x\,\Delta y\,\tau}{2\pi\sigma_x\sigma_y}.
}
$$

## Fitting

All parameters except $\tau$ are taken from the datasheet or independent
measurements:

| Symbol | Source |
|--------|--------|
| $\mathrm{PDE}$ | `parse_datasheets.py` (laser-weighted, per OV) |
| $\sigma_x$, $\sigma_y$ | `fit_scan.py` (razor-blade scan, per OV) |
| $\Delta x = \Delta y = 35\,\mu\mathrm{m}$ | MicroFJ-60035 datasheet (cell pitch) |

The only free parameter is the **SPAD reset time** $\tau$, which is estimated
from the RC time constant $\tau \approx R_q C_\mathrm{cell}$ and then fitted.

The x-axis of the measured data is $R_\gamma$ (incident photon rate, photons/s);
the y-axis is $\langle R \rangle$ (avalanche rate, avalanches/s).
