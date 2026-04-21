"""
Instrument datasheet parameters for the SiPM scanning experiment.

Instruments:
  - onsemi MicroFJ-60035-TSV  (SiPM under test)
  - Siglent SDM3045X          (DMM, measures SiPM current)
  - Keithley 6487             (picoammeter, measures photodiode current)
  - Thorlabs FDS100           (reference photodiode, 0 V bias)
  - D405-20                   (405 nm laser diode)

Graph data (Figs 1-4 of the onsemi datasheet) cannot be extracted
automatically from the PDF; values are manually digitized below.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# onsemi MicroFJ-60035-TSV  (MICROJ-SERIES-D.pdf)
# ---------------------------------------------------------------------------

# General parameters (Table 1)
SIPM_60035_GENERAL = {
    "model":                      "MicroFJ-60035-TSV",
    "manufacturer":               "onsemi",
    "active_area_mm2":            6.07 * 6.07,        # 6.07 x 6.07 mm
    "n_microcells":               22292,
    "microcell_pitch_um":         35.0,
    "vbr_typ_V":                  24.7,
    "vbr_min_V":                  24.2,
    "vbr_tempco_mV_per_C":        21.5,
    "ov_min_V":                   1.0,
    "ov_max_V":                   6.0,
    "spectral_range_nm":          (200, 900),          # where PDE > 2% at Vbr+6V
    "peak_pde_wavelength_nm":     420.0,
    "measurement_temp_C":         21.0,
}

# Breakdown voltage uncertainty (1-sigma, estimated from unit-to-unit spread and
# temperature stability).  Since OV = V_bias − V_BD and V_bias is set precisely,
# sigma_OV = VBD_ERR_V.  This propagates into every OV-dependent parameter.
VBD_ERR_V = 0.25   # V

# Performance parameters (Table 3, 60035 column)
# Each dict has keys for the two standard overvoltages: +2.5 V and +6 V
SIPM_60035_PERFORMANCE = {
    "overvoltage_V":              [2.5, 6.0],
    "pde_pct":                    [38.0, 50.0],        # at peak wavelength 420 nm
    "dark_count_rate_kHz_per_mm2":[50.0, 150.0],
    "gain":                       [2.9e6, 6.3e6],
    "dark_current_typ_uA":        [0.9,   7.5],
    "dark_current_max_uA":        [1.25,  12.0],
    "rise_time_ps":               [180.0, 250.0],      # anode-cathode output
    "microcell_recharge_tau_ns":  50.0,                # same at both OV
    "capacitance_anode_pF":       4140.0,
    "capacitance_fast_pF":        160.0,
    "fast_output_fwhm_ns":        3.0,
    "crosstalk_pct":              [8.0,  25.0],
    "afterpulsing_pct":           [0.75,  5.0],
}

# Figure 1 — PDE vs wavelength (MicroFJ-60035-TSV, Vbr + 2.5 V and Vbr + 6 V)
# Manually digitized from the datasheet graph.
SIPM_60035_PDE_VS_WAVELENGTH = {
    "wavelength_nm": np.array([
        270, 300, 330, 360, 390, 420, 450, 480, 510, 540,
        570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900
    ]),
    "pde_ov2p5_pct": np.array([
         5,  10,  18,  26,  34,  38,  37,  35,  32,  28,
        24,  20,  17,  14,  11,   9,   7,   5,   4,   3,   2,   1
    ]),
    "pde_ov6_pct": np.array([
         7,  14,  25,  36,  46,  50,  49,  46,  42,  37,
        32,  27,  22,  18,  14,  11,   8,   6,   5,   3,   2,   1
    ]),
    "note": "Manually digitized from Fig. 1 of MICROJ-SERIES-D.pdf; approximate."
}

# Figure 2 — PDE vs overvoltage (MicroFJ-60035-TSV, at peak wavelength ~420 nm)
# Manually digitized.
SIPM_60035_PDE_VS_OV = {
    "overvoltage_V":  np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]),
    "pde_pct":        np.array([24,  28,  33,  38,  41,  43,  45,  46,  48,  49,  50]),
    "note": "Manually digitized from Fig. 2 of MICROJ-SERIES-D.pdf; approximate."
}

# Figure 3 — PDE vs crosstalk (MicroFJ-60035-TSV)
# Shows the trade-off between PDE and optical crosstalk as OV increases.
SIPM_60035_PDE_VS_CROSSTALK = {
    "crosstalk_pct": np.array([ 5,  8, 12, 17, 22, 25]),
    "pde_pct":       np.array([34, 38, 42, 45, 48, 50]),
    "note": "Manually digitized from Fig. 3 of MICROJ-SERIES-D.pdf; approximate."
}

# Figure 4 — Gain vs overvoltage (datasheet shows MicroFJ-30035-TSV;
# 60035 shares the same microcell design so scaling applies)
SIPM_60035_GAIN_VS_OV = {
    "overvoltage_V": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]),
    "gain":          np.array([1.2e6, 1.6e6, 2.1e6, 2.9e6, 3.6e6, 4.2e6, 4.8e6, 5.2e6, 5.6e6, 6.0e6, 6.3e6]),
    "note": "Manually digitized from Fig. 4 of MICROJ-SERIES-D.pdf (30035 curve); "
            "60035 shares the same microcell so gain values are consistent with Table 3."
}

# ---------------------------------------------------------------------------
# Siglent SDM3045X  (SDM3045X_DataSheet.pdf)
# DC current accuracy table (1 year, 23 °C ±5 °C)
# Accuracy = ±(pct_rdg % of reading  +  offset_counts × resolution)
# ---------------------------------------------------------------------------

SIGLENT_DC_CURRENT = {
    "model":        "SDM3045X",
    "manufacturer": "Siglent",
    "digits":       4.5,
    "counts":       60000,
    # Each entry: (range_A, resolution_A, pct_rdg, offset_counts)
    "accuracy_table": [
        {"range_A": 600e-6,  "resolution_A": 0.01e-6,  "pct_rdg": 0.05, "offset_counts": 3},
        {"range_A": 6e-3,    "resolution_A": 0.1e-6,   "pct_rdg": 0.05, "offset_counts": 3},
        {"range_A": 60e-3,   "resolution_A": 1e-6,     "pct_rdg": 0.05, "offset_counts": 3},
        {"range_A": 600e-3,  "resolution_A": 10e-6,    "pct_rdg": 0.12, "offset_counts": 6},
        {"range_A": 6.0,     "resolution_A": 100e-6,   "pct_rdg": 0.20, "offset_counts": 5},
        {"range_A": 10.0,    "resolution_A": 1e-3,     "pct_rdg": 0.25, "offset_counts": 4},
    ],
}

def siglent_current_accuracy(current_A):
    """Return absolute accuracy (A) for a given DC current reading."""
    for row in SIGLENT_DC_CURRENT["accuracy_table"]:
        if abs(current_A) <= row["range_A"] * 1.1:   # 10% over-range allowed
            return (row["pct_rdg"] / 100.0) * abs(current_A) + row["offset_counts"] * row["resolution_A"]
    raise ValueError(f"Current {current_A} A out of range for SDM3045X")

# ---------------------------------------------------------------------------
# Keithley 6487 picoammeter  (1KW-73905-1_6487_...pdf)
# DC current accuracy (1 year, 18–28 °C, 0–70% RH, 1 PLC)
# Accuracy = ±(pct_rdg % of reading  +  offset)
# ---------------------------------------------------------------------------

KEITHLEY_6487_CURRENT = {
    "model":        "6487",
    "manufacturer": "Keithley / Tektronix",
    "digits":       5.5,
    "burden_voltage_V": 200e-6,   # < 200 µV on all ranges except 20 mA
    # Each entry: (range_A, resolution_A, pct_rdg, offset_A)
    "accuracy_table": [
        {"range_A": 2e-9,   "resolution_A": 10e-15,  "pct_rdg": 0.30, "offset_A": 400e-15},
        {"range_A": 20e-9,  "resolution_A": 100e-15, "pct_rdg": 0.20, "offset_A": 1e-12},
        {"range_A": 200e-9, "resolution_A": 1e-12,   "pct_rdg": 0.15, "offset_A": 10e-12},
        {"range_A": 2e-6,   "resolution_A": 10e-12,  "pct_rdg": 0.15, "offset_A": 100e-12},
        {"range_A": 20e-6,  "resolution_A": 100e-12, "pct_rdg": 0.10, "offset_A": 1e-9},
        {"range_A": 200e-6, "resolution_A": 1e-9,    "pct_rdg": 0.10, "offset_A": 10e-9},
        {"range_A": 2e-3,   "resolution_A": 10e-9,   "pct_rdg": 0.10, "offset_A": 100e-9},
        {"range_A": 20e-3,  "resolution_A": 100e-9,  "pct_rdg": 0.10, "offset_A": 1e-6},
    ],
}

def keithley_current_accuracy(current_A):
    """Return absolute accuracy (A) for a given DC current reading."""
    for row in KEITHLEY_6487_CURRENT["accuracy_table"]:
        if abs(current_A) <= row["range_A"]:
            return (row["pct_rdg"] / 100.0) * abs(current_A) + row["offset_A"]
    raise ValueError(f"Current {current_A} A out of range for Keithley 6487")

# ---------------------------------------------------------------------------
# Thorlabs FDS100-CAL photodiode  (calibration certificate 25031152700)
# Serial number 250131214, calibrated 31.01.2025 at Thorlabs GmbH Bergkirchen.
# Traceable to PTB/NIST via Hamamatsu reference detector (serial 2K018,
# Cal.-Nr. 73335 23 PTB).
# Operated at 0 V bias (photovoltaic mode) in this experiment.
#
# Calibration expanded uncertainty (k=2, 95% coverage):
#   350–470 nm : 2.9 %   →  1-sigma ≈ 1.45 %
#   470–990 nm : 2.7 %   →  1-sigma ≈ 1.35 %
#   990–1100 nm: 3.1 %   →  1-sigma ≈ 1.55 %
# ---------------------------------------------------------------------------

FDS100_RESPONSIVITY = {
    "model":        "FDS100-CAL",
    "manufacturer": "Thorlabs",
    "serial_number": "250131214",
    "calibration_cert": "25031152700",
    "calibration_date": "31.01.2025",
    # Calibration expanded uncertainty (k=2) by wavelength band:
    "uncertainty_k2_pct": {(350, 470): 2.9, (470, 990): 2.7, (990, 1100): 3.1},
    "wavelength_nm": np.array([
        350, 360, 370, 380, 390, 400, 410, 420, 430, 440,
        450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
        550, 560, 570, 580, 590, 600, 610, 620, 630, 640,
        650, 660, 670, 680, 690, 700, 710, 720, 730, 740,
        750, 760, 770, 780, 790, 800, 810, 820, 830, 840,
        850, 860, 870, 880, 890, 900, 910, 920, 930, 940,
        950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040,
        1050, 1060, 1070, 1080, 1090, 1100
    ]),
    "responsivity_A_per_W": np.array([
        4.70e-2, 4.61e-2, 4.66e-2, 5.08e-2, 5.87e-2, 6.90e-2, 8.05e-2, 9.23e-2, 1.04e-1, 1.16e-1,
        1.27e-1, 1.38e-1, 1.48e-1, 1.59e-1, 1.69e-1, 1.79e-1, 1.89e-1, 1.99e-1, 2.10e-1, 2.20e-1,
        2.30e-1, 2.41e-1, 2.51e-1, 2.62e-1, 2.72e-1, 2.83e-1, 2.94e-1, 3.05e-1, 3.16e-1, 3.28e-1,
        3.39e-1, 3.50e-1, 3.62e-1, 3.73e-1, 3.85e-1, 3.96e-1, 4.07e-1, 4.19e-1, 4.30e-1, 4.41e-1,
        4.52e-1, 4.63e-1, 4.74e-1, 4.85e-1, 4.95e-1, 5.06e-1, 5.16e-1, 5.26e-1, 5.36e-1, 5.46e-1,
        5.56e-1, 5.65e-1, 5.74e-1, 5.83e-1, 5.91e-1, 5.99e-1, 6.07e-1, 6.15e-1, 6.22e-1, 6.29e-1,
        6.35e-1, 6.39e-1, 6.42e-1, 6.41e-1, 6.34e-1, 6.19e-1, 5.93e-1, 5.53e-1, 4.99e-1, 4.32e-1,
        3.58e-1, 2.85e-1, 2.25e-1, 1.81e-1, 1.46e-1, 1.14e-1
    ]),
    # Linear interpolation between 400 nm (6.90e-2) and 410 nm (8.05e-2):
    "responsivity_at_405nm_A_per_W": 0.07475,
    "dark_current_at_0V_A":          0.0,    # negligible at 0 V bias
    "active_area_mm2":               13.0,
    "bias_voltage_V":                0.0,    # photovoltaic mode
}

# ---------------------------------------------------------------------------
# Beam splitter
# ---------------------------------------------------------------------------

BEAMSPLITTER = {
    "sipm_fraction": 0.0902,   # 10% to SiPM
    "pd_fraction":   1 - 0.0902,   # 90% to photodiode
}

# ---------------------------------------------------------------------------
# D405-20 laser diode  (D405-20.pdf)
# ---------------------------------------------------------------------------

LASER_D405 = {
    "model":                    "D405-20",
    "wavelength_typ_nm":        405.0,
    "wavelength_min_nm":        395.0,
    "wavelength_max_nm":        415.0,
    "power_mW":                 20.0,
    "operating_current_typ_mA": 50.0,
    "operating_current_max_mA": 75.0,
    "operating_voltage_typ_V":  5.0,
    "operating_voltage_max_V":  6.5,
    "threshold_current_typ_mA": 25.0,
    "threshold_current_max_mA": 50.0,
    "slope_efficiency_mW_per_mA": 1.3,
    "beam_divergence_parallel_deg":    (6.0, 9.0, 12.0),   # min, typ, max
    "beam_divergence_perp_deg":        (16.0, 20.0, 25.0),
    "package":                  "TO-5.6mm",
    "temp_range_C":             (-10, 70),
}


# ---------------------------------------------------------------------------
# Laser spectrum — Gaussian, mean 405 nm, 395–415 nm span = ±3σ  → σ = 10/3 nm
# ---------------------------------------------------------------------------

LASER_SIGMA_NM = 10.0 / 3.0   # ~3.33 nm

def laser_spectrum(wavelength_nm):
    """Normalised Gaussian laser spectral density (nm^-1)."""
    return np.exp(-0.5 * ((wavelength_nm - 405.0) / LASER_SIGMA_NM) ** 2) / \
           (LASER_SIGMA_NM * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------------------
# Interpolation to 4 V overvoltage
# ---------------------------------------------------------------------------

def interpolate_at_ov(target_ov):
    """
    Return a dict of SiPM parameters interpolated to target_ov (in volts).

    Spectral PDE at the target OV is derived by scaling the 2.5 V spectrum
    using the ratio from the PDE-vs-OV curve, which captures the non-linear
    OV dependence.  Crosstalk is found by inverting the PDE-vs-crosstalk curve
    (which is parameterised implicitly by OV): given PDE(target_ov), look up
    the corresponding crosstalk.
    """
    ov_pts  = np.array(SIPM_60035_PDE_VS_OV["overvoltage_V"])
    pde_pts = np.array(SIPM_60035_PDE_VS_OV["pde_pct"])

    # --- scalar performance parameters ---
    pde       = float(np.interp(target_ov, ov_pts, pde_pts))
    gain      = float(np.interp(target_ov,
                                np.array(SIPM_60035_GAIN_VS_OV["overvoltage_V"]),
                                np.array(SIPM_60035_GAIN_VS_OV["gain"])))

    ov_2pt    = np.array(SIPM_60035_PERFORMANCE["overvoltage_V"])           # [2.5, 6.0]
    dc_typ    = float(np.interp(target_ov, ov_2pt,
                                np.array(SIPM_60035_PERFORMANCE["dark_current_typ_uA"])))
    dc_max    = float(np.interp(target_ov, ov_2pt,
                                np.array(SIPM_60035_PERFORMANCE["dark_current_max_uA"])))
    ap        = float(np.interp(target_ov, ov_2pt,
                                np.array(SIPM_60035_PERFORMANCE["afterpulsing_pct"])))
    dcr       = float(np.interp(target_ov, ov_2pt,
                                np.array(SIPM_60035_PERFORMANCE["dark_count_rate_kHz_per_mm2"])))

    # --- crosstalk: invert PDE-vs-crosstalk (PDE is monotone in this range) ---
    ct_pde_pts = np.array(SIPM_60035_PDE_VS_CROSSTALK["pde_pct"])
    ct_pts     = np.array(SIPM_60035_PDE_VS_CROSSTALK["crosstalk_pct"])
    crosstalk  = float(np.interp(pde, ct_pde_pts, ct_pts))

    # --- spectral PDE at target OV ---
    # Scale the 2.5 V spectrum by the ratio pde(target_ov) / pde(2.5 V),
    # derived from the PDE-vs-OV curve so the non-linearity is preserved.
    pde_at_2p5 = float(np.interp(2.5, ov_pts, pde_pts))
    scale      = pde / pde_at_2p5
    wl         = np.array(SIPM_60035_PDE_VS_WAVELENGTH["wavelength_nm"])
    pde_spec   = np.array(SIPM_60035_PDE_VS_WAVELENGTH["pde_ov2p5_pct"]) * scale

    return {
        "overvoltage_V":                  target_ov,
        "pde_pct":                        pde,
        "gain":                           gain,
        "dark_current_typ_uA":            dc_typ,
        "dark_current_max_uA":            dc_max,
        "afterpulsing_pct":               ap,
        "dark_count_rate_kHz_per_mm2":    dcr,
        "crosstalk_pct":                  crosstalk,
        "pde_vs_wavelength_nm":           wl,
        "pde_vs_wavelength_pct":          pde_spec,
        "microcell_recharge_tau_ns":      SIPM_60035_PERFORMANCE["microcell_recharge_tau_ns"],
        "pde_interpolation_note": (
            "Spectral PDE scaled from the 2.5 V OV spectrum using the ratio "
            "PDE(target_ov)/PDE(2.5V) taken from the PDE-vs-OV curve (Fig. 2). "
            "Crosstalk inverted from the PDE-vs-crosstalk curve (Fig. 3). "
            "All other scalars linearly interpolated between datasheet anchor points."
        ),
    }


SIPM_60035_AT_4V = interpolate_at_ov(4.0)


def vbd_errors_at_ov(target_ov):
    """
    Absolute 1-sigma errors on SiPM performance parameters due to VBD_ERR_V
    uncertainty in the breakdown voltage (equivalently, the overvoltage).

    All OV-dependent quantities are differentiated numerically using central
    finite differences on the same interpolation curves used in interpolate_at_ov.

    Returns a dict with keys:
      pde_err_pct          – error on peak PDE (percentage points)
      gain_err             – error on gain (same units as gain)
      afterpulsing_err_pct – error on afterpulsing (percentage points)
      crosstalk_err_pct    – error on crosstalk (percentage points)
    """
    h    = 0.01   # V step for finite difference (much smaller than VBD_ERR_V)
    ov_lo = target_ov - h
    ov_hi = target_ov + h

    ov_pts   = np.array(SIPM_60035_PDE_VS_OV["overvoltage_V"])
    pde_pts  = np.array(SIPM_60035_PDE_VS_OV["pde_pct"])
    gain_ov  = np.array(SIPM_60035_GAIN_VS_OV["overvoltage_V"])
    gain_pts = np.array(SIPM_60035_GAIN_VS_OV["gain"])
    ov_2pt   = np.array(SIPM_60035_PERFORMANCE["overvoltage_V"])
    ap_pts   = np.array(SIPM_60035_PERFORMANCE["afterpulsing_pct"])
    ct_pde   = np.array(SIPM_60035_PDE_VS_CROSSTALK["pde_pct"])
    ct_pts   = np.array(SIPM_60035_PDE_VS_CROSSTALK["crosstalk_pct"])

    dpde  = (np.interp(ov_hi, ov_pts, pde_pts)  - np.interp(ov_lo, ov_pts, pde_pts))  / (2 * h)
    dgain = (np.interp(ov_hi, gain_ov, gain_pts) - np.interp(ov_lo, gain_ov, gain_pts)) / (2 * h)
    dap   = (np.interp(ov_hi, ov_2pt, ap_pts)   - np.interp(ov_lo, ov_2pt, ap_pts))   / (2 * h)

    # Crosstalk is parameterised via PDE(OV): CT = CT(PDE(OV))
    dct = (np.interp(np.interp(ov_hi, ov_pts, pde_pts), ct_pde, ct_pts) -
           np.interp(np.interp(ov_lo, ov_pts, pde_pts), ct_pde, ct_pts)) / (2 * h)

    return {
        "pde_err_pct":          abs(dpde)  * VBD_ERR_V,
        "gain_err":             abs(dgain) * VBD_ERR_V,
        "afterpulsing_err_pct": abs(dap)   * VBD_ERR_V,
        "crosstalk_err_pct":    abs(dct)   * VBD_ERR_V,
    }


# ---------------------------------------------------------------------------
# Spectrally-weighted effective responsivity and PDE  (inner products)
#
# L(λ) is the normalised laser Gaussian: ∫ L dλ = 1  (units: nm⁻¹)
# Inner product: <Q, L> = ∫ Q(λ) L(λ) dλ
# Because L is already normalised this equals the laser-weighted average of Q.
# ---------------------------------------------------------------------------

# Fine wavelength grid spanning ±5σ around 405 nm
_wl = np.linspace(405.0 - 5 * LASER_SIGMA_NM, 405.0 + 5 * LASER_SIGMA_NM, 2000)
_L  = laser_spectrum(_wl)    # normalised: np.trapz(_L, _wl) == 1

# Interpolate FDS100 responsivity onto grid
_R_interp = np.interp(_wl,
                      FDS100_RESPONSIVITY["wavelength_nm"],
                      FDS100_RESPONSIVITY["responsivity_A_per_W"])

# Interpolate SiPM PDE spectra onto grid (fraction, not percent)
_pde_2p5_interp = np.interp(_wl,
                             SIPM_60035_PDE_VS_WAVELENGTH["wavelength_nm"],
                             SIPM_60035_PDE_VS_WAVELENGTH["pde_ov2p5_pct"]) / 100.0

_pde_4v_interp  = np.interp(_wl,
                             SIPM_60035_AT_4V["pde_vs_wavelength_nm"],
                             SIPM_60035_AT_4V["pde_vs_wavelength_pct"]) / 100.0

# Inner products: <Q, L> = ∫ Q(λ) L(λ) dλ
# L normalised → result is dimensionally Q's unit (A/W or dimensionless)
EFFECTIVE_RESPONSIVITY_A_PER_W = np.trapz(_R_interp       * _L, _wl)
EFFECTIVE_PDE_2P5V             = np.trapz(_pde_2p5_interp * _L, _wl)
EFFECTIVE_PDE_4V               = np.trapz(_pde_4v_interp  * _L, _wl)

# Spectral product curves (not normalised — show the actual integrand)
_pd_product_2p5  = _R_interp       * _L   # A/W per nm, integrates to EFFECTIVE_RESPONSIVITY
_sipm_product_2p5 = _pde_2p5_interp * _L  # fraction per nm, integrates to EFFECTIVE_PDE_2P5V
_sipm_product_4v  = _pde_4v_interp  * _L  # fraction per nm, integrates to EFFECTIVE_PDE_4V

LASER_WEIGHTED = {
    "laser_mean_nm":                     405.0,
    "laser_sigma_nm":                    LASER_SIGMA_NM,
    "laser_3sigma_range_nm":             (395.0, 415.0),
    "wavelength_grid_nm":                _wl,
    "laser_spectrum":                    _L,
    "pd_responsivity_interp":            _R_interp,
    "sipm_pde_2p5v_interp":              _pde_2p5_interp,
    "sipm_pde_4v_interp":                _pde_4v_interp,
    "pd_product":                        _pd_product_2p5,
    "sipm_product_2p5v":                 _sipm_product_2p5,
    "sipm_product_4v":                   _sipm_product_4v,
    "effective_pd_responsivity_A_per_W": EFFECTIVE_RESPONSIVITY_A_PER_W,
    "effective_sipm_pde_2p5v":           EFFECTIVE_PDE_2P5V,
    "effective_sipm_pde_4v":             EFFECTIVE_PDE_4V,
}

# ---------------------------------------------------------------------------
# Measured dark currents (empirical, from nd=0 center scans)
# ---------------------------------------------------------------------------

SIPM_DARK_CURRENT_MEASURED_A = {
    2.5: 1.4335108492e-6,
    4.0: 3.3424781548e-6,
}

# ---------------------------------------------------------------------------
# VBD-propagated errors at each operating overvoltage
# ---------------------------------------------------------------------------

_vbd_2p5 = vbd_errors_at_ov(2.5)
_vbd_4p0 = vbd_errors_at_ov(4.0)

# Peak PDE at each OV (percent) — used to scale eff_pde error via chain rule:
#   eff_pde(OV) = eff_pde(2.5V) * pde_peak(OV) / pde_peak(2.5V)
#   d(eff_pde)/d(OV) = eff_pde * (d pde_peak/d OV) / pde_peak(OV)
_pde_peak_2p5 = float(np.interp(2.5, SIPM_60035_PDE_VS_OV["overvoltage_V"],
                                 SIPM_60035_PDE_VS_OV["pde_pct"]))
_pde_peak_4p0 = float(np.interp(4.0, SIPM_60035_PDE_VS_OV["overvoltage_V"],
                                 SIPM_60035_PDE_VS_OV["pde_pct"]))

# ---------------------------------------------------------------------------
# Convenience bundles: all parameters needed for IV analysis at each OV
# ---------------------------------------------------------------------------

SIPM_PARAMS = {
    2.5: {
        "overvoltage_V":              2.5,
        "gain":                       SIPM_60035_PERFORMANCE["gain"][0],          # 2.9e6
        "effective_pde":              EFFECTIVE_PDE_2P5V,
        "crosstalk":                  SIPM_60035_PERFORMANCE["crosstalk_pct"][0] / 100.0,
        "afterpulsing":               SIPM_60035_PERFORMANCE["afterpulsing_pct"][0] / 100.0,
        "dark_current_measured_A":    SIPM_DARK_CURRENT_MEASURED_A[2.5],
        "dark_current_datasheet_typ_A": SIPM_60035_PERFORMANCE["dark_current_typ_uA"][0] * 1e-6,
        # VBD-propagated 1-sigma errors
        "gain_err_vbd":               _vbd_2p5["gain_err"],
        "effective_pde_err_vbd":      EFFECTIVE_PDE_2P5V * _vbd_2p5["pde_err_pct"] / _pde_peak_2p5,
        "crosstalk_err_vbd":          _vbd_2p5["crosstalk_err_pct"] / 100.0,
        "afterpulsing_err_vbd":       _vbd_2p5["afterpulsing_err_pct"] / 100.0,
    },
    4.0: {
        "overvoltage_V":              4.0,
        "gain":                       SIPM_60035_AT_4V["gain"],
        "effective_pde":              EFFECTIVE_PDE_4V,
        "crosstalk":                  SIPM_60035_AT_4V["crosstalk_pct"] / 100.0,
        "afterpulsing":               SIPM_60035_AT_4V["afterpulsing_pct"] / 100.0,
        "dark_current_measured_A":    SIPM_DARK_CURRENT_MEASURED_A[4.0],
        "dark_current_datasheet_typ_A": SIPM_60035_AT_4V["dark_current_typ_uA"] * 1e-6,
        # VBD-propagated 1-sigma errors
        "gain_err_vbd":               _vbd_4p0["gain_err"],
        "effective_pde_err_vbd":      EFFECTIVE_PDE_4V * _vbd_4p0["pde_err_pct"] / _pde_peak_4p0,
        "crosstalk_err_vbd":          _vbd_4p0["crosstalk_err_pct"] / 100.0,
        "afterpulsing_err_vbd":       _vbd_4p0["afterpulsing_err_pct"] / 100.0,
    },
}

PD_PARAMS = {
    "effective_responsivity_A_per_W": EFFECTIVE_RESPONSIVITY_A_PER_W,
    "dark_current_A":                 FDS100_RESPONSIVITY["dark_current_at_0V_A"],
    "pd_fraction":                    BEAMSPLITTER["pd_fraction"],
    "sipm_fraction":                  BEAMSPLITTER["sipm_fraction"],
}


if __name__ == "__main__":
    PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'parse_datasheets')
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=== onsemi MicroFJ-60035-TSV ===")
    for k, v in SIPM_60035_GENERAL.items():
        print(f"  {k}: {v}")
    print()
    print("  Performance at 2.5 V / 6.0 V OV:")
    for k, v in SIPM_60035_PERFORMANCE.items():
        print(f"  {k}: {v}")

    print("\n=== Siglent SDM3045X - DC current accuracy ===")
    for I in [1e-6, 10e-6, 100e-6, 1e-3]:
        acc = siglent_current_accuracy(I)
        print(f"  I = {I*1e6:.1f} uA  accuracy = +/-{acc*1e9:.2f} nA  ({acc/I*100:.3f}%)")

    print("\n=== Keithley 6487 - DC current accuracy ===")
    for I in [1e-9, 10e-9, 1e-6, 10e-6]:
        acc = keithley_current_accuracy(I)
        print(f"  I = {I*1e9:.1f} nA  accuracy = +/-{acc*1e12:.2f} pA  ({acc/I*100:.3f}%)")

    print("\n=== Thorlabs FDS100 ===")
    print(f"  Responsivity at 405 nm: {FDS100_RESPONSIVITY['responsivity_at_405nm_A_per_W']} A/W")
    print(f"  Dark current at 0 V:    {FDS100_RESPONSIVITY['dark_current_at_0V_A']} A")

    print("\n=== D405-20 laser diode ===")
    print(f"  Wavelength: {LASER_D405['wavelength_min_nm']}-{LASER_D405['wavelength_max_nm']} nm "
          f"(typ {LASER_D405['wavelength_typ_nm']} nm)")

    print("\n=== MicroFJ-60035-TSV interpolated at 4.0 V OV ===")
    d = SIPM_60035_AT_4V
    print(f"  PDE at 420 nm:         {d['pde_pct']:.1f} %")
    print(f"  Gain:                  {d['gain']:.3e}")
    print(f"  Dark current (typ):    {d['dark_current_typ_uA']:.2f} uA")
    print(f"  Dark current (max):    {d['dark_current_max_uA']:.2f} uA")
    print(f"  Crosstalk:             {d['crosstalk_pct']:.1f} %")
    print(f"  Afterpulsing:          {d['afterpulsing_pct']:.2f} %")
    print(f"  DCR:                   {d['dark_count_rate_kHz_per_mm2']:.0f} kHz/mm2")
    print(f"  PDE at 405 nm (spec):  {float(np.interp(405, d['pde_vs_wavelength_nm'], d['pde_vs_wavelength_pct'])):.1f} %")

    print(f"\n=== VBD error propagation (VBD_ERR_V = {VBD_ERR_V} V) ===")
    for ov, sp in SIPM_PARAMS.items():
        print(f"  OV = {ov} V:")
        print(f"    gain:          {sp['gain']:.3e}  +/- {sp['gain_err_vbd']:.2e}  "
              f"({sp['gain_err_vbd']/sp['gain']*100:.1f}%)")
        print(f"    eff PDE:       {sp['effective_pde']*100:.2f}%  +/- {sp['effective_pde_err_vbd']*100:.2f}pp  "
              f"({sp['effective_pde_err_vbd']/sp['effective_pde']*100:.1f}%)")
        print(f"    crosstalk:     {sp['crosstalk']*100:.2f}%  +/- {sp['crosstalk_err_vbd']*100:.2f}pp  "
              f"({sp['crosstalk_err_vbd']/sp['crosstalk']*100:.1f}%)")
        print(f"    afterpulsing:  {sp['afterpulsing']*100:.3f}%  +/- {sp['afterpulsing_err_vbd']*100:.3f}pp  "
              f"({sp['afterpulsing_err_vbd']/sp['afterpulsing']*100:.1f}%)")

    print("\n=== Laser-weighted effective quantities ===")
    lw = LASER_WEIGHTED
    print(f"  Laser: Gaussian, mean={lw['laser_mean_nm']} nm, sigma={lw['laser_sigma_nm']:.2f} nm")
    print(f"  Effective PD responsivity:  {lw['effective_pd_responsivity_A_per_W']:.5f} A/W")
    print(f"  Effective SiPM PDE (2.5 V): {lw['effective_sipm_pde_2p5v']*100:.2f} %")
    print(f"  Effective SiPM PDE (4.0 V): {lw['effective_sipm_pde_4v']*100:.2f} %")

    import matplotlib.pyplot as plt

    wl  = lw["wavelength_grid_nm"]
    L   = lw["laser_spectrum"]
    L_n = L / L.max()   # normalise to 1 for overlay

    # ---- Figure 1: SiPM PDE vs laser spectrum --------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    fig.suptitle("SiPM PDE and laser spectrum")

    ax = axes[0]
    ax.plot(wl, lw["sipm_pde_2p5v_interp"] * 100, color="steelblue",  label="PDE 2.5 V OV")
    ax.plot(wl, lw["sipm_pde_4v_interp"]   * 100, color="darkorange", label="PDE 4.0 V OV")
    ax.set_ylabel("PDE (%)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax = axes[1]
    ax.plot(wl, L_n, color="mediumpurple", label="Laser spectrum (norm.)")
    ax.set_ylabel("Laser intensity (a.u.)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax = axes[2]
    ax.plot(wl, lw["sipm_product_2p5v"] * 100, color="steelblue",
            label=f"PDE x L(lambda), 2.5 V  [integral = {lw['effective_sipm_pde_2p5v']*100:.2f}%]")
    ax.plot(wl, lw["sipm_product_4v"]   * 100, color="darkorange",
            label=f"PDE x L(lambda), 4.0 V  [integral = {lw['effective_sipm_pde_4v']*100:.2f}%]")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("PDE x L(lambda) (% / nm)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sipm_pde_laser.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: PD responsivity vs laser spectrum -------------------------
    fig2, axes2 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    fig2.suptitle("Photodiode responsivity and laser spectrum")

    ax = axes2[0]
    ax.plot(wl, lw["pd_responsivity_interp"], color="forestgreen", label="FDS100 responsivity")
    ax.set_ylabel("Responsivity (A/W)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax = axes2[1]
    ax.plot(wl, L_n, color="mediumpurple", label="Laser spectrum (norm.)")
    ax.set_ylabel("Laser intensity (a.u.)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax = axes2[2]
    ax.plot(wl, lw["pd_product"], color="forestgreen",
            label=f"R x L(lambda)  [integral = {lw['effective_pd_responsivity_A_per_W']*1000:.4f} mA/W/nm * nm]")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("R x L(lambda) (A/W / nm)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, "pd_responsivity_laser.png"), dpi=150)
    plt.close(fig2)

    # ---- Figure 3: OV-dependent SiPM properties (6-panel) -------------------
    _ov_fine   = np.linspace(1.0, 6.0, 300)
    _ov_curve  = np.array(SIPM_60035_PDE_VS_OV["overvoltage_V"])
    _pde_curve = np.array(SIPM_60035_PDE_VS_OV["pde_pct"])
    _gain_ov   = np.array(SIPM_60035_GAIN_VS_OV["overvoltage_V"])
    _gain_pts  = np.array(SIPM_60035_GAIN_VS_OV["gain"])
    _ov_2pt    = np.array(SIPM_60035_PERFORMANCE["overvoltage_V"])
    _ct_pde    = np.array(SIPM_60035_PDE_VS_CROSSTALK["pde_pct"])
    _ct_pts    = np.array(SIPM_60035_PDE_VS_CROSSTALK["crosstalk_pct"])
    _ap_pts    = np.array(SIPM_60035_PERFORMANCE["afterpulsing_pct"])
    _dcr_pts   = np.array(SIPM_60035_PERFORMANCE["dark_count_rate_kHz_per_mm2"])

    _pde_fine  = np.interp(_ov_fine, _ov_curve, _pde_curve)
    _gain_fine = np.interp(_ov_fine, _gain_ov, _gain_pts)
    _ct_fine   = np.interp(_pde_fine, _ct_pde, _ct_pts)
    _ap_fine   = np.interp(_ov_fine, _ov_2pt, _ap_pts)
    _dcr_fine  = np.interp(_ov_fine, _ov_2pt, _dcr_pts)

    _pde_at_2p5_peak = float(np.interp(2.5, _ov_curve, _pde_curve))
    _eff_pde_fine = []
    for _ov_i in _ov_fine:
        _scale   = float(np.interp(_ov_i, _ov_curve, _pde_curve)) / _pde_at_2p5_peak
        _pde_spec = np.array(SIPM_60035_PDE_VS_WAVELENGTH["pde_ov2p5_pct"]) * _scale / 100.0
        _pde_i    = np.interp(_wl, SIPM_60035_PDE_VS_WAVELENGTH["wavelength_nm"], _pde_spec)
        _eff_pde_fine.append(np.trapz(_pde_i * _L, _wl))
    _eff_pde_fine = np.array(_eff_pde_fine) * 100.0   # percent

    _op_styles = {
        2.5: dict(color='steelblue',  marker='o'),
        4.0: dict(color='darkorange', marker='s'),
    }
    _vbd_errs = {2.5: _vbd_2p5, 4.0: _vbd_4p0}

    fig3, axes3 = plt.subplots(3, 2, figsize=(12, 11))
    fig3.suptitle("onsemi MicroFJ-60035-TSV: OV-dependent properties\n"
                  f"(error bars = VBD uncertainty ±{VBD_ERR_V} V)", fontsize=11)

    # [0,0] PDE vs OV
    ax = axes3[0, 0]
    ax.plot(_ov_fine, _pde_fine, 'k-', linewidth=1.5, label='Peak PDE at 420 nm')
    for ov_val, sp in SIPM_PARAMS.items():
        vbd = _vbd_errs[ov_val]
        pde_pk = float(np.interp(ov_val, _ov_curve, _pde_curve))
        st = _op_styles[ov_val]
        ax.errorbar(ov_val, pde_pk, yerr=vbd["pde_err_pct"],
                    fmt=st['marker'], color=st['color'], markersize=8,
                    capsize=4, elinewidth=1.5,
                    label=f"{ov_val} V: {pde_pk:.0f} ± {vbd['pde_err_pct']:.1f} %")
    ax.set_xlabel("Overvoltage (V)")
    ax.set_ylabel("Peak PDE at 420 nm (%)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # [0,1] Gain vs OV
    ax = axes3[0, 1]
    ax.plot(_ov_fine, _gain_fine / 1e6, 'k-', linewidth=1.5, label='Gain (datasheet)')
    for ov_val, sp in SIPM_PARAMS.items():
        st = _op_styles[ov_val]
        ax.errorbar(ov_val, sp["gain"] / 1e6, yerr=sp["gain_err_vbd"] / 1e6,
                    fmt=st['marker'], color=st['color'], markersize=8,
                    capsize=4, elinewidth=1.5,
                    label=f"{ov_val} V: {sp['gain']:.2e} ± {sp['gain_err_vbd']:.1e}")
    ax.set_xlabel("Overvoltage (V)")
    ax.set_ylabel("Gain (×10⁶)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # [1,0] Crosstalk vs OV
    ax = axes3[1, 0]
    ax.plot(_ov_fine, _ct_fine, 'k-', linewidth=1.5, label='Crosstalk (via PDE-CT curve)')
    for ov_val, sp in SIPM_PARAMS.items():
        vbd = _vbd_errs[ov_val]
        st = _op_styles[ov_val]
        ax.errorbar(ov_val, sp["crosstalk"] * 100, yerr=vbd["crosstalk_err_pct"],
                    fmt=st['marker'], color=st['color'], markersize=8,
                    capsize=4, elinewidth=1.5,
                    label=f"{ov_val} V: {sp['crosstalk']*100:.1f} ± {vbd['crosstalk_err_pct']:.2f} %")
    ax.set_xlabel("Overvoltage (V)")
    ax.set_ylabel("Optical crosstalk (%)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # [1,1] Afterpulsing vs OV
    ax = axes3[1, 1]
    ax.plot(_ov_fine, _ap_fine, 'k-', linewidth=1.5,
            label='Afterpulsing (linear interp., 2 datasheet points)')
    ax.scatter([2.5, 6.0], list(_ap_pts), s=50, color='k', zorder=5)
    for ov_val, sp in SIPM_PARAMS.items():
        vbd = _vbd_errs[ov_val]
        st = _op_styles[ov_val]
        ax.errorbar(ov_val, sp["afterpulsing"] * 100, yerr=vbd["afterpulsing_err_pct"],
                    fmt=st['marker'], color=st['color'], markersize=8,
                    capsize=4, elinewidth=1.5,
                    label=f"{ov_val} V: {sp['afterpulsing']*100:.2f} ± {vbd['afterpulsing_err_pct']:.3f} %")
    ax.set_xlabel("Overvoltage (V)")
    ax.set_ylabel("Afterpulsing (%)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # [2,0] DCR vs OV
    ax = axes3[2, 0]
    ax.plot(_ov_fine, _dcr_fine, 'k-', linewidth=1.5,
            label='DCR (linear interp., 2 datasheet points)')
    ax.scatter([2.5, 6.0], list(_dcr_pts), s=50, color='k', zorder=5,
               label='Datasheet anchor points')
    ax.set_xlabel("Overvoltage (V)")
    ax.set_ylabel("DCR (kHz / mm²)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # [2,1] Laser-weighted effective PDE vs OV
    ax = axes3[2, 1]
    ax.plot(_ov_fine, _eff_pde_fine, 'k-', linewidth=1.5,
            label='Effective PDE (405 nm laser-weighted)')
    for ov_val, sp in SIPM_PARAMS.items():
        eff_err_pct = sp["effective_pde_err_vbd"] * 100
        st = _op_styles[ov_val]
        ax.errorbar(ov_val, sp["effective_pde"] * 100, yerr=eff_err_pct,
                    fmt=st['marker'], color=st['color'], markersize=8,
                    capsize=4, elinewidth=1.5,
                    label=f"{ov_val} V: {sp['effective_pde']*100:.2f} ± {eff_err_pct:.2f} %")
    ax.set_xlabel("Overvoltage (V)")
    ax.set_ylabel("Laser-weighted effective PDE (%)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    fig3.tight_layout()
    fig3.savefig(os.path.join(PLOT_DIR, "sipm_ov_properties.png"), dpi=150)
    plt.close(fig3)

    # ---- Figure 4: Spectral responsivities ----------------------------------
    # Upper panel: FDS100 calibrated responsivity with per-band uncertainty shading
    # Lower panel: SiPM PDE spectra at 2.5 V and 4.0 V OV
    # Vertical dashed line at 405 nm (laser wavelength) on both panels.

    _cal_wl = FDS100_RESPONSIVITY["wavelength_nm"]
    _cal_R  = FDS100_RESPONSIVITY["responsivity_A_per_W"]

    # Build 1-sigma relative uncertainty array (k=2 → divide by 2)
    _cal_rel_err = np.where(_cal_wl <= 470, 2.9 / 2 / 100,
                   np.where(_cal_wl <= 990, 2.7 / 2 / 100,
                                            3.1 / 2 / 100))
    _cal_R_hi = _cal_R * (1 + _cal_rel_err)
    _cal_R_lo = _cal_R * (1 - _cal_rel_err)

    _sipm_wl    = np.array(SIPM_60035_PDE_VS_WAVELENGTH["wavelength_nm"])
    _sipm_2p5   = np.array(SIPM_60035_PDE_VS_WAVELENGTH["pde_ov2p5_pct"])
    _sipm_4v    = np.array(SIPM_60035_AT_4V["pde_vs_wavelength_pct"])

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig4.suptitle("Spectral responsivities\n"
                  "FDS100-CAL (cert. 25031152700, S/N 250131214, 31.01.2025) "
                  "and onsemi MicroFJ-60035-TSV PDE", fontsize=9)

    _laser_2sig = 2 * LASER_SIGMA_NM   # ±2σ half-width ≈ ±6.67 nm

    # --- upper panel: FDS100 responsivity ---
    ax4a.plot(_cal_wl, _cal_R, color='forestgreen', linewidth=1.8,
              label='FDS100-CAL responsivity (cal.)')
    ax4a.fill_between(_cal_wl, _cal_R_lo, _cal_R_hi,
                      color='forestgreen', alpha=0.25,
                      label='±1σ calibration uncertainty')
    ax4a.axvline(405, color='mediumpurple', linewidth=1.2, linestyle='--',
                 label='Laser 405 nm')
    ax4a.axvspan(405 - _laser_2sig, 405 + _laser_2sig,
                 color='mediumpurple', alpha=0.12,
                 label=f'Laser ±2σ  ({405 - _laser_2sig:.1f}–{405 + _laser_2sig:.1f} nm)')
    ax4a.set_ylabel('Responsivity (A / W)')
    ax4a.legend(fontsize=8)
    ax4a.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # --- lower panel: SiPM PDE ---
    ax4b.plot(_sipm_wl, _sipm_2p5, color='steelblue',  linewidth=1.8,
              label='SiPM PDE  2.5 V OV (datasheet Fig. 1)')
    ax4b.plot(_sipm_wl, _sipm_4v,  color='darkorange', linewidth=1.8,
              label='SiPM PDE  4.0 V OV (scaled from 2.5 V)')
    ax4b.axvline(405, color='mediumpurple', linewidth=1.2, linestyle='--',
                 label='Laser 405 nm')
    ax4b.axvspan(405 - _laser_2sig, 405 + _laser_2sig,
                 color='mediumpurple', alpha=0.12,
                 label=f'Laser ±2σ  ({405 - _laser_2sig:.1f}–{405 + _laser_2sig:.1f} nm)')
    ax4b.set_xlabel('Wavelength (nm)')
    ax4b.set_ylabel('PDE (%)')
    ax4b.legend(fontsize=8)
    ax4b.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    fig4.tight_layout()
    fig4.savefig(os.path.join(PLOT_DIR, "spectral_responsivities.png"), dpi=150)
    plt.close(fig4)

    print(f"\nPlots saved to: {PLOT_DIR}")
