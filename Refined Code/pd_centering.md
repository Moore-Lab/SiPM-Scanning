# pd_centering.py — Photodiode Centering Scans

## Overview

Performs line scans using **only the Keithley electrometer** to measure photodiode current (no Siglent / no SiPM). Used to align the photodiode to the beam centre independently of the SiPM.

## Dependencies

| Package        | Purpose                      |
|----------------|------------------------------|
| `int_arduino`  | Arduino motor control        |
| `int_keithley` | Keithley electrometer        |
| `numpy`        | Array operations             |
| `os`           | File/directory management    |

## Functions

### `measure(x_mm, y_mm, csv=...)`

Reads Keithley current (3 measurements, no delay) and appends to CSV.

**CSV columns:**
```
x, y, pd_current, pd_std, pd_stderr, pd_time
```

### `xline_scan(x, y1, y2, ds, func, csv=...)`

Scans Y positions at fixed X.

### `yline_scan(y, x1, x2, ds, func, csv=...)`

Scans X positions at fixed Y.

### `center()`

Returns a fixed estimate `[3.5, 3.5]` for the centre. Intended as a placeholder for a proper fitting routine.

## Standalone Execution

```bash
python pd_centering.py
```

1. Initializes Arduino and Keithley.
2. Sets Keithley voltage to 0 V (no SiPM bias — PD only).
3. Runs X and Y line scans across 0–4 mm at 0.1 mm steps.
4. Returns stage home.

## Output Files

| File Pattern                         | Description |
|--------------------------------------|-------------|
| `data/pd_line_scan_<ts>.csv`        | PD centering scan data |

## Notes

- The 0–4 mm scan range is smaller than the 0–10 mm range used for SiPM scans, reflecting the smaller active area of the photodiode.
- Uses global `ser` and `keithley` objects.
- Voltage is set to 0 V because the photodiode operates in photovoltaic mode (no reverse bias needed for centering).
