# saturation_curve.py — SiPM Saturation Curve Analysis

## Overview

Constructs and fits the **SiPM saturation curve** — the relationship between incident photon rate and the rate of fired microcells. At high photon rates, the finite number of microcells and their recovery time causes the response to saturate. This script loads centre-scan and line-scan data across multiple ND filter settings, converts currents to physical rates, and fits a saturation model.

## Dependencies

| Package              | Purpose                              |
|----------------------|--------------------------------------|
| `numpy`              | Array operations                     |
| `matplotlib.pyplot`  | Plotting                             |
| `scipy.special`      | Exponential integral `expi`          |
| `scipy.constants`    | Physical constants ($h$, $c$)        |
| `scipy.optimize`     | `curve_fit`                          |
| `darroch_error`      | Beam radius extraction from line scans |

## Saturation Model

### `Nfired_func(n, x_stdv, y_stdv, reset=50e-9)`

Models the average number of fired microcells as a function of incident photon count, using a Gaussian beam illumination profile:

$$N_{\text{fired}} = A\left[\gamma + \ln\!\left(\frac{k \cdot n}{A}\right) - \text{Ei}\!\left(-\frac{k \cdot n}{A}\right)\right]$$

where:
- $A = \frac{2\pi\,\sigma_x\,\sigma_y}{a}$ — effective number of illuminated cells
- $a = (35\,\mu\text{m})^2 \times \tau_{\text{reset}}$ — microcell area × recovery time
- $k = 0.3745$ — detection efficiency factor
- $\gamma \approx 0.5772$ — Euler–Mascheroni constant
- $\text{Ei}$ — exponential integral

| Parameter | Type  | Description |
|-----------|-------|-------------|
| `n`       | array | Number of incident photons per second |
| `x_stdv`  | float | Beam σ in x (m) |
| `y_stdv`  | float | Beam σ in y (m) |
| `reset`   | float | Microcell recovery time (s), fit parameter |

## Data Pipeline

### 1. Load Centre Scans

Iterates over `data/OVone/meas_*_center_scan_*.csv` files. Each file is a single-row CSV from a centre-point measurement at a particular ND filter setting.

- **SiPM current** (column 2): inverted and dark-current-subtracted.
- **PD current** (column 6): photodiode reading for photon rate calculation.

### 2. Extract Beam Radii from Line Scans

For ND values > 2.5 (where the beam profile is well-defined), fits X and Y line scans with `darroch_error.fit()` and averages the beam radius parameter `popt[4]`.

Beam standard deviation: $\sigma = \frac{r_{\text{beam}}}{2} \times 10^{-3}$ (converting mm to m and radius to σ).

### 3. Correct SiPM Current

- **Dark current subtraction:** $I_{\text{SiPM}} - 0.9\,\mu\text{A}$
- **Crosstalk + afterpulsing correction:** $I_{\text{corr}} = I / (1 + c + a)$ where $c = 0.08$, $a = 0.0075$

### 4. Convert to Physical Rates

**Photon rate** from PD current:
$$\dot{N}_\gamma = \frac{I_{\text{PD}}}{R \cdot E_\gamma} \times \text{BS}$$

where $R = 0.07475$ A/W (responsivity at 405 nm), $E_\gamma = hc/\lambda$, BS = 0.090 (beam splitter ratio).

**Avalanche rate** from SiPM current:
$$\dot{N}_{\text{aval}} = \frac{I_{\text{SiPM}}}{G \cdot e}$$

where $G = 2.9 \times 10^6$ (gain), $e = 1.6 \times 10^{-19}$ C.

### 5. Fit and Plot

- Fits `Nfired_func` to the data with `reset` as the free parameter.
- Plots on log-log axes with a smooth model curve via `np.logspace`.
- Lower panel shows normalised residuals with $\chi^2$ value.

## Output

A two-panel figure:
- **Top:** Log-log saturation curve (data + model fit).
- **Bottom:** Residuals with reduced chi-squared.

## Physical Constants Used

| Constant | Value | Description |
|----------|-------|-------------|
| Microcell pitch | 35 µm | SiPM pixel size |
| Laser wavelength | 405 nm | Blue/violet diode laser |
| PD responsivity | 0.07475 A/W | At 405 nm |
| SiPM gain | 2.9 × 10⁶ | At operating voltage |
| Crosstalk | 8% | Optical crosstalk probability |
| Afterpulsing | 0.75% | Afterpulse probability |
| Beam splitter | 9.02% | Fraction reaching PD |

## Notes

- Data folder is `data/OVone` (1 V overvoltage above breakdown).
- A `filtered_indices` mask removes photon rates below $10^8$ cps.
- The `reset` parameter (microcell recovery time, ~50 ns) is the sole free parameter in the fit.
- Measurement errors assume Siglent specification: 0.25% of reading + 0.04 µA.
