# For each csv file of the center scans, it will take the PD current as the x coordinate and the SiPM current as the y coordinate

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
from scipy import special
import scipy.constants as const
from scipy.optimize import curve_fit
import darroch_error

param = [50E-9]  # Initial guess for the parameters k and reset time

def Nfired_func(n,x_stdv,y_stdv,reset=50E-9):
    # I have to average radii among all the trials
    #s = 2.3E-3/4
    k = .3745
    a = (35E-6)**2*reset
    A = (2*np.pi*x_stdv*y_stdv/a)
    return A*(np.euler_gamma+np.log(k*n/A)-special.expi(-k*n/A))

pd_current = []
sipm_current = []


x_rad = []
y_rad = []
# Go through every csv file that starts with 'data/meas_' and ends with '_center_scan.csv'
for csv_file in os.listdir("data/first"):
    if csv_file.startswith("meas_") and "_center_scan" in csv_file:
        # Load the data from the csv file
        data = np.loadtxt(os.path.join("data/first", csv_file), delimiter=",", skiprows=1)
        pd_current.append(data[6])  # Assuming PD current is in the seventh column
        sipm_current.append(-1 * data[2])  # Assuming SiPM current is in the third column

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
                center, pos, current, current_err, p0 = darroch_error.profile(os.path.join("data/first", csv_file), 1)
                pos_line, fitted_curve, fit_param, popt, residuals = darroch_error.fit(pos, current, current_err, p0)
                rad = popt[4]
                x_rad.append(rad)
            elif "yline_scan" in csv_file:
                center, pos, current, current_err, p0 = darroch_error.profile(os.path.join("data/first", csv_file), 0)
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
sipm_current = np.array(sipm_current) - .9E-6  # Subtracting a small value to account for dark current

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

# Convert the SiPM_current to Rate of fired cells (cps)
gain = 2.9E6  # Gain of the SiPM
charge = 1.6E-19  # Charge of an electron in Coulombs
avalanches = np.array(sipm_current) * 1/(gain * charge)  # Convert from Amps

popt, pcov = curve_fit(lambda n, reset: Nfired_func(n, x_stdv, y_stdv, reset), photons, avalanches, p0=param)
print(popt)

y_error = avalanches * .0025 + .04*10**-6  # Error in the siglent measurements
# Graph residuals
residuals = (avalanches - Nfired_func(photons, x_stdv, y_stdv, *popt))/(y_error)

def chi2_calc(res):
    X2 = np.sum(res**2)/len(res)
    return X2

chi2 = chi2_calc(residuals)
#print(chi2)

figure, axis = plt.subplots(2, 1, figsize=(10, 8))
#axis2 = axis.twiny()
axis[0].errorbar(photons,avalanches,yerr=y_error,fmt='o',label='Data')
#axis.plot(pd_current,Nfired_func(pd_current),ls='-',label='Fit')
axis[0].plot(photons,Nfired_func(photons, x_stdv, y_stdv, *popt), ls='--', label='Model')
#axis.plot(rate_ph,rate_meas,'s',ls='-',label='Measurement')
#axis2.plot(rate_pwr,rate_eff,'o',ls='-',label='Power')
axis[0].set_xlabel('Rate of photons [cps]')
#axis2.set_xlabel('Power [mW]')
axis[0].set_ylabel('Rate of fired cells [cps]')
axis[0].set_xscale('log')
#axis2.set_xscale('log')
axis[0].set_yscale('log')
axis[0].set_title('SiPM Saturation Curve')
axis[0].legend()

axis[1].plot(photons, residuals, marker='o', ls='None', label=f'Residuals (Chi2={chi2:.2f})')
axis[1].set_xscale('log')
axis[1].set_xlabel('Rate of photons [cps]')
axis[1].set_ylabel('Residuals [cps]')
axis[1].set_title('Residuals of SiPM Saturation Curve Fit')
axis[1].legend()

figure.tight_layout()
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
