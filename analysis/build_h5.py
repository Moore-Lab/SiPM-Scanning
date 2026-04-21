import h5py
import numpy as np
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_datasheets as ds

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'measurements.h5')

FOLDERS = {
    'OVfive': {'over_voltage': 2.5, 'bias_voltage': 27.3},
    'OVfour': {'over_voltage': 4.0, 'bias_voltage': 28.8},
}

COLUMNS = ['x', 'y', 'sipm_current', 'sipm_std', 'sipm_stderr',
           'sipm_time', 'pd_current', 'pd_std', 'pd_stderr', 'pd_time']

PATTERN = re.compile(r'meas_([^_]+)_([^_]+)_(center_scan|xline_scan|yline_scan)_(\d+)\.csv')

with h5py.File(OUTPUT_FILE, 'w') as f:
    for folder, meta in FOLDERS.items():
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f'Skipping missing folder: {folder_path}')
            continue

        ov_grp = f.create_group(folder)
        ov_grp.attrs['over_voltage'] = meta['over_voltage']
        ov_grp.attrs['bias_voltage'] = meta['bias_voltage']
        ov_grp.attrs['vbd_err_V']    = ds.VBD_ERR_V

        sp = ds.SIPM_PARAMS[meta['over_voltage']]
        ov_grp.attrs['gain_err_vbd']          = sp['gain_err_vbd']
        ov_grp.attrs['effective_pde_err_vbd'] = sp['effective_pde_err_vbd']
        ov_grp.attrs['crosstalk_err_vbd']     = sp['crosstalk_err_vbd']
        ov_grp.attrs['afterpulsing_err_vbd']  = sp['afterpulsing_err_vbd']

        runs = {}
        for fname in os.listdir(folder_path):
            m = PATTERN.match(fname)
            if not m:
                continue
            nd_str, bias_str, scan_type, ts_str = m.groups()
            key = (float(nd_str), int(ts_str))
            runs.setdefault(key, {})[scan_type] = os.path.join(folder_path, fname)

        for (nd, timestamp), scans in sorted(runs.items()):
            nd_grp = ov_grp.create_group(f'nd_{nd}_t{timestamp}')
            nd_grp.attrs['nd'] = nd
            nd_grp.attrs['timestamp'] = timestamp
            nd_grp.attrs['over_voltage'] = meta['over_voltage']
            nd_grp.attrs['bias_voltage'] = meta['bias_voltage']

            for scan_type, fpath in scans.items():
                raw = np.loadtxt(fpath, delimiter=',', skiprows=1)
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                ds = nd_grp.create_dataset(scan_type, data=raw)
                ds.attrs['columns'] = COLUMNS
                ds.attrs['scan_type'] = scan_type
                ds.attrs['nd'] = nd
                ds.attrs['timestamp'] = timestamp
                ds.attrs['over_voltage'] = meta['over_voltage']
                ds.attrs['bias_voltage'] = meta['bias_voltage']

            print(f'  {folder}/nd_{nd}_t{timestamp}: {list(scans.keys())}')

print(f'\nSaved: {OUTPUT_FILE}')
