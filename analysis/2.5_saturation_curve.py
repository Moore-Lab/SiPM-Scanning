# For each csv file of the center scans, it will take the PD current as the x coordinate and the SiPM current as the y coordinate

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
from scipy import special
import scipy.constants as const
from scipy.optimize import curve_fit
import darroch_error

param = [1]  # Initial guess for the parameters k and reset time

def Nfired_func(n,x_stdv,y_stdv,reset=1):
    # I have to average radii among all the trials
    k = 0.3759
    RC=50E-9
    tau=reset*RC
    a = (35E-6)**2*tau
    A = (2*np.pi*x_stdv*y_stdv/a)
    return A*(np.euler_gamma+np.log(k*n/A)-special.expi(-k*n/A))

pd_current = []
sipm_current = []


x_rad = []
y_rad = []
# Go through every csv file that starts with 'data/meas_' and ends with '_center_scan.csv'
for csv_file in os.listdir("data/OVfive"):
    if csv_file.startswith("meas_") and "_center_scan" in csv_file:
        # Load the data from the csv file
        data = np.loadtxt(os.path.join("data/OVfive", csv_file), delimiter=",", skiprows=1)
        pd_current.append(data[6])  # Assuming PD current is in the seventh column
        sipm_current.append(data[2])  # Assuming SiPM current is in the third column

        # Here you can process the data as needed
        # For example, you could fit a model to the data or compute some statistics
    # How do I only analyze csv files with ND greater than 2.5
    elif csv_file.startswith("meas_") and ("xline_scan" in csv_file or "yline_scan" in csv_file):
        # Extract ND value from filename
        try:
            nd_str = csv_file.split("_")[1]
            nd_val = float(nd_str)
        except (IndexError, ValueError):
            continue  # Skip files with unexpected naming
        # Process the file only if ND value is greater than 2.5 because before that radius is insignificant
        if nd_val > 2.5:
            if "xline_scan" in csv_file:
                center, pos, current, current_err, p0 = darroch_error.profile(os.path.join("data/OVfive", csv_file), 1)
                pos_line, fitted_curve, fit_param, popt, residuals = darroch_error.fit(pos, current, current_err, p0)
                rad = popt[4]
                x_rad.append(rad)
            elif "yline_scan" in csv_file:
                center, pos, current, current_err, p0 = darroch_error.profile(os.path.join("data/OVfive", csv_file), 0)
                pos_line, fitted_curve, fit_param, popt, residuals = darroch_error.fit(pos, current, current_err, p0)
                rad = popt[4]
                y_rad.append(rad)
x_rad = np.mean(x_rad)
y_rad = np.mean(y_rad)
x_stdv = x_rad/2 *10**-3
y_stdv = y_rad/2 *10**-3
print(f"Average x radius: {x_rad}, Average y radius: {y_rad}")

# Get rid of dark current
pd_current = np.array(pd_current) #- .0032E-9  # Subtracting a small value to account for dark current
sipm_current = np.array(sipm_current) - 1.4335108492e-06  # Subtracting a small value to account for dark current

# I have to consider the crosstalk and afterpulsing of the SiPM
c = .08
a = .0075
'''
y = np.linspace(0.05,0.15,num=10000)
ymin = y[np.argmin(np.abs(c - y*np.exp(-y)))]
print(ymin)
print(-np.log(1-c))
c = ymin
'''
sipm_current = sipm_current / (1 + c + a)  # Correct for crosstalk and afterpulsing

#   Conversion factor to go from PD current to Rate of photons (cps)
r = .07475  # Responsitivity for 405 nm estimated from my specific data sheet
j = (const.h * const.c)/(405E-9)    # Joules per photon at 405 nm
factor = r * j
bs = 0.09024390244 # Because of beam splitter
# Convert the PD_current to Rate of photons (cps)
photons = np.array(pd_current) * 1/factor * bs  # Convert from Amps
filtered_indices = photons >= 1e8
photons = photons[filtered_indices]

# Convert the SiPM_current to Rate of fired cells (cps)
gain = 2.9E6  # Gain of the SiPM
charge = 1.6E-19  # Charge of an electron in Coulombs
avalanches = np.array(sipm_current) * 1/(gain * charge)  # Convert from Amps
avalanches = avalanches[filtered_indices]

popt, pcov = curve_fit(lambda n, reset: Nfired_func(n, x_stdv, y_stdv, reset), photons, avalanches, p0=param)
print("The reset time is (%.2e +/- %.2e) ns"%(popt[0]*50E-9, np.sqrt(pcov[0][0])*50E-9))

# Calculate effective PDE from data (at low photon rates before saturation)
effective_pde_data = avalanches / photons
# Sort by photon rate to identify leftmost (lowest photon rate) points
sorted_indices = np.argsort(photons)
pde_estimate = np.mean(effective_pde_data[sorted_indices[:5]])  # Average of first 5 leftmost points
print(f"Estimated PDE from data (low flux): {pde_estimate:.4f}")

y_error = abs(avalanches * .0025 + .04*10**-6)  # Error in the siglent measurements
# Graph residuals
residuals = (avalanches - Nfired_func(photons, x_stdv, y_stdv, *popt))/(y_error)

def chi2_calc(res):
    X2 = np.sum(res**2)/len(res)
    return X2

chi2 = chi2_calc(residuals)
#print(chi2)

# Generate smooth curves
photons_smooth = np.logspace(np.log10(photons.min()), np.log10(photons.max()), 500)
effective_pde_model = Nfired_func(photons_smooth, x_stdv, y_stdv, *popt) / photons_smooth

'''
figure, axis = plt.subplots(2, 1, figsize=(10, 11), sharex=True, constrained_layout=True)

# Plot 1: Saturation curve (avalanches vs photons)
axis[0].errorbar(photons, avalanches, yerr=y_error, fmt='o', label='Data')
axis[0].plot(photons_smooth, Nfired_func(photons_smooth, x_stdv, y_stdv, *popt), ls='--', label='Model')
axis[0].set_xlabel('Rate of photons [cps]')
axis[0].set_ylabel('Rate of fired cells [cps]')
axis[0].set_xscale('log')
axis[0].set_yscale('log')
axis[0].set_title('SiPM Saturation Curve')
axis[0].legend()
axis[0].grid(True, alpha=0.3)

# Plot 2: Residuals
axis[1].plot(photons, residuals, marker='o', ls='None', label=f'Residuals (Chi2={chi2:.2f})')
axis[1].set_xscale('log')
axis[1].set_xlabel('Rate of photons [cps]')
axis[1].set_ylabel('Residuals')
axis[1].set_title('Residuals of SiPM Saturation Curve Fit')
axis[1].legend()
axis[1].grid(True, alpha=0.3)
'''

figure, axis = plt.subplots(1, 1, figsize=(6.2, 4.4))
axis.errorbar(photons, avalanches, yerr=y_error, fmt='o', ms=4, label='Data')
axis.plot(photons_smooth, Nfired_func(photons_smooth, x_stdv, y_stdv, *popt), ls='--', lw=1.5, label='Model')
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_xlabel('Rate of photons [cps]', labelpad=2)
axis.set_ylabel('Rate of fired cells [cps]', labelpad=2)
axis.set_title('SiPM Saturation Curve', pad=2)
axis.grid(True, alpha=0.3)
axis.legend(frameon=False)
figure.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.95)

'''
# Plot 3: Effective PDE (avalanches/photons vs photons)
axis[2].plot(photons, effective_pde_data, marker='o', ls='None', label='Data', markersize=6)
axis[2].axhline(pde_estimate, color='r', linestyle=':', alpha=0.7, linewidth=2, label=f'PDE estimate: {pde_estimate:.4f}')
axis[2].set_xscale('log')
axis[2].set_yscale('log')
axis[2].set_xlabel('Rate of photons [cps]')
axis[2].set_ylabel('Effective PDE (N_fired / N_photons)')
axis[2].set_title('Effective PDE vs Photon Rate')
axis[2].legend()
axis[2].grid(True, alpha=0.3)
'''

plt.show()

'''
figure, axis = plt.subplots(1, 1, figsize=(8, 6))
axis.plot(pd_current, sipm_current, 'o')
axis.set_xlabel('Rate of photons [cps]')
axis.set_ylabel('Rate of fired cells [cps]')
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_title('SiPM Saturation Curve')
#axis.legend()

# Plot Lucas's fit solved for a gaussian beam
#placeholder for stdv of the gaussian beam (radius)
stdv = 10**-3
# The repeated factor
rf = 2*np.pi*(stdv)**2/((35*10**-6)**2)
gamma = 0
#This PDE does not include crosstalk or afterpulsing
pde = .38
xspan = np.logspace(1,15,1000)

fit = rf*(gamma + np.log(pde * xspan / rf) - expi(-pde * xspan / rf))
axis.plot(xspan, fit, 'r-', label='Fit')

plt.show()
'''
