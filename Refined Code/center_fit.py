import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

# ---------------------------------------------------------------------------
# Function: double_edge_profile
# Description:
#   Models a beam profile with two edges (rising and falling) using error functions.
#   The function is offset + amplitude scaled difference of two erf curves,
#   representing the gradual transition edges of the beam intensity.
#
# Parameters:
#   pos     : array-like
#       Positions at which to evaluate the profile (e.g., spatial coordinates).
#   offset  : float
#       Baseline offset (background current level).
#   amplitude : float
#       Peak amplitude of the beam profile above the offset.
#   center1 : float
#       Position of the first edge (rising edge center).
#   width1  : float
#       Width (standard deviation) of the first edge transition.
#   center2 : float
#       Position of the second edge (falling edge center).
#   width2  : float
#       Width (standard deviation) of the second edge transition.
#
# Returns:
#   Array of profile values at each position.
# ---------------------------------------------------------------------------
def double_edge_profile(pos, offset, amplitude, center1, width1, center2, width2):
    # Calculate the profile as offset + amplitude * (erf rising - erf falling)
    return offset + 0.5 * amplitude * (
        erf((pos - center1) / (np.sqrt(2) * width1)) - 
        erf((pos - center2) / (np.sqrt(2) * width2))
    )

# ---------------------------------------------------------------------------
# Function: load_scan
# Description:
#   Loads scan data from a CSV file, filters it along a fixed axis value within tolerance,
#   and returns position and inverted current arrays along the other axis.
#
# Parameters:
#   csv_file : str
#       Path to the CSV file containing scan data.
#   axis : str, default 'x'
#       Axis to keep fixed ('x' or 'y').
#   fixed_val : float, default 5.0
#       Fixed coordinate value on the chosen axis to select data from.
#   tol : float, default 1e-6
#       Tolerance used to match fixed_val (to handle floating-point errors).
#
# Returns:
#   pos : numpy array
#       Positions along the scan axis (perpendicular to fixed axis).
#   current : numpy array
#       Measured SiPM current values, inverted (multiplied by -1) if needed.
# ---------------------------------------------------------------------------
def load_scan(csv_file, axis='x', fixed_val=5.0, tol=1e-6):
    # Load CSV data assuming headers and columns 'x', 'y', 'sipm_current'
    data = np.genfromtxt(csv_file, delimiter=",", names=True)
    
    # Select data near fixed_val along the fixed axis with given tolerance
    if axis == 'x':
        mask = np.abs(data['x'] - fixed_val) < tol
        pos = data['y'][mask]  # Position array along Y-axis for x fixed
    else:
        mask = np.abs(data['y'] - fixed_val) < tol
        pos = data['x'][mask]  # Position array along X-axis for y fixed
    
    # Extract SiPM current and invert if measurement polarity requires it
    current = data['sipm_current'][mask]
    
    return pos, -1 * current  # Invert current if sensor polarity is reversed

# ---------------------------------------------------------------------------
# Function: analyze_beam
# Description:
#   Performs analysis on the beam scan data along one axis:
#     - Loads the scan data
#     - Cleans invalid points (NaNs)
#     - Sorts data for consistent fitting
#     - Provides initial parameter guesses for double-edge fit
#     - Fits the double_edge_profile function to data using nonlinear least squares
#     - Calculates beam Full Width at Half Maximum (FWHM) as distance between edges
#     - Calculates centroid (center of mass) of the profile
#
# Parameters:
#   csv_file : str
#       Path to the CSV file containing scan data.
#   axis : str, default 'x'
#       Axis that is fixed in the scan ('x' or 'y').
#   fixed_val : float, default 5.0
#       Fixed coordinate value along the fixed axis.
#
# Returns:
#   pos : numpy array
#       Positions along the scan axis.
#   current : numpy array
#       Corresponding SiPM current values.
#   popt : array-like
#       Optimized parameters for the double-edge profile function.
#   fwhm : float
#       Full width at half maximum of the beam (distance between edges).
#   centroid : float
#       Center of mass of the beam profile weighted by current.
# ---------------------------------------------------------------------------
def analyze_beam(csv_file, axis='x', fixed_val=5.0):
    pos, current = load_scan(csv_file, axis, fixed_val)

    # Remove any NaN values that could interfere with fitting
    mask = ~np.isnan(current)
    pos = pos[mask]
    current = current[mask]

    # Require sufficient points for a meaningful fit
    if len(pos) < 6:
        raise ValueError(f"Too few data points for axis {axis}. Check fixed_val or increase tolerance.")

    # Sort position and current arrays in ascending position order for smooth fitting and plotting
    idx = np.argsort(pos)
    pos = pos[idx]
    current = current[idx]

    # Initial parameter guesses for the fit:
    offset = np.min(current)              # baseline offset near minimum current
    amplitude = np.max(current) - offset # approximate peak amplitude above offset
    grad = np.gradient(current)           # numerical derivative of current vs position
    
    # Rough estimate of edge centers from max and min gradient (rising and falling edges)
    center1 = pos[np.argmax(grad)]
    center2 = pos[np.argmin(grad)]
    
    # Initial widths set to a fraction of distance between edges
    width1 = width2 = (center2 - center1) / 10
    
    p0 = [offset, amplitude, center1, width1, center2, width2]

    # Perform nonlinear least squares curve fit of double_edge_profile to the data
    popt, _ = curve_fit(double_edge_profile, pos, current, p0=p0)

    # Calculate FWHM as absolute distance between two fitted edge centers
    fwhm = abs(popt[4] - popt[2])

    # Calculate centroid as weighted mean position (center of mass)
    centroid = np.sum(pos * current) / np.sum(current)

    return pos, current, popt, fwhm, centroid

# ---------------------------------------------------------------------------
# Function: analyze_and_plot
# Description:
#   Performs analysis and plots for two orthogonal beam scans (X and Y).
#   For each axis:
#     - Runs analyze_beam to fit and extract beam parameters
#     - Generates smooth fitted curve for visualization
#     - Plots data points, fit curve, edge positions, and centroid
#     - Prints fitted parameters and metrics to console
#
# Parameters:
#   x_file : str
#       Path to CSV file for X-scan data (Y fixed).
#   y_file : str
#       Path to CSV file for Y-scan data (X fixed).
# ---------------------------------------------------------------------------
def analyze_and_plot(x_file, y_file):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # For 'X' plot, use x_file and axis='x'
    # For 'Y' plot, use y_file and axis='y'
    file_axis_pairs = [(x_file, 'x'), (y_file, 'y')]
    centroids = []

    for ax, (file, axis_fixed), axis_label in zip(axs, file_axis_pairs, ['X', 'Y']):
        pos, current, popt, fwhm, centroid = analyze_beam(file, axis=axis_fixed, fixed_val=5.0)
        centroids.append(centroid)

        fit_pos = np.linspace(min(pos), max(pos), 500)
        fit_current = double_edge_profile(fit_pos, *popt)

        ax.plot(pos, current, '.', label='Data')
        ax.plot(fit_pos, fit_current, '-', color='orange', label='Fit')
        ax.axvline(popt[2], color='gray', linestyle='--', label='Edge 1')
        ax.axvline(popt[4], color='gray', linestyle='--', label='Edge 2')
        ax.axvline(centroid, color='orange', linestyle='--', label='Centroid')

        ax.set_title(f'{axis_label} Beam Profile (fixed {axis_label.lower()} = 5 mm)')
        ax.set_xlabel(f'{"Y" if axis_label=="X" else "X"}-position [mm]')
        ax.set_ylabel('SiPM Current [A]')
        ax.grid(True)
        ax.legend()

        print(f'\n[{axis_label}-scan @ fixed {"y" if axis_label=="X" else "x"} = 5.0 mm]')
        print(f'  Edge 1 Center: {popt[2]:.3f} mm')
        print(f'  Edge 2 Center: {popt[4]:.3f} mm')
        print(f'  FWHM: {fwhm:.3f} mm')
        print(f'  Centroid: {centroid:.3f} mm')
        print(f'  Min Current (offset): {popt[0]:.2e} A')
        print(f'  Max Current (offset + amplitude): {(popt[0] + popt[1]):.2e} A')

    plt.tight_layout()
    #plt.show()
    return tuple(centroids)  # (centroid_x, centroid_y)

def find_center(xcsv, ycsv):
    center = analyze_and_plot(xcsv, ycsv)
    print(f"Center of beam profile: {center}")
    return center

# ---------------------------------------------------------------------------
# Main guard: run example analysis if script is executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    analyze_and_plot(
        x_file=r"C:\Users\ddaya\OneDrive - Yale University\Darroch Research\SiPM-Scanning\Refined Code\data\4.5_cal_xline_scan_1752613149.csv",
        y_file=r"C:\Users\ddaya\OneDrive - Yale University\Darroch Research\SiPM-Scanning\Refined Code\data\4.5_cal_yline_scan_1752613149.csv"
    )
    #find_center(        xcsv=r"C:\Users\ddaya\OneDrive - Yale University\Darroch Research\Refined Code\data\x_scan_1751303823.csv",
        #ycsv=r"C:\Users\ddaya\OneDrive - Yale University\Darroch Research\Refined Code\data\y_scan_1751303823.csv"
    #)
