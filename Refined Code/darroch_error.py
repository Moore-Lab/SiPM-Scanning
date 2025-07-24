import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import special


def joined_erf(x, *p):
    # p = [offset, amplitude, center1, center2, width]
    # Use center1 and width1 for x < transition, center2 and width2 for x >= transition
    transition = (p[2] + p[3]) / 2
    result = np.zeros_like(x)
    mask = x < transition
    result[mask] = p[1]/2 * (1 - special.erf(np.sqrt(2)*(p[2] - x[mask])/p[4])) + p[0]
    result[~mask] = p[1]/2 * (1 - special.erf(np.sqrt(2)*(x[~mask] - p[3])/p[4])) + p[0]
    return result

# These are the guesses for the parameters
def profile(csv, col):
    csv = np.loadtxt(csv, delimiter=",", skiprows=1)
    pos = csv[:, col]
    current = -1 * csv[:, 2]
    #current_err = csv[:, 4]
    # Trying a current error of the 600uA of the siglent
    current_err = current * .0025 + .04*10**-6
    # Take the mean of the 10% lowest current values to give the offset
    offset = np.mean(np.sort(current)[:int(len(current)*0.1)])
    # Take the mean of the 25% highest current values and subtract the lowest to give the amplitude
    maximum = np.mean(np.sort(current)[-int(len(current)*0.25):])
    amplitude = maximum - offset
    fwhm = amplitude / 2
    # use np.where to find the indices where the current is greater than the offset + fwhm
    indices = np.where(current >= offset + fwhm)[0]
    # Take the first and last indices where the current is greater than the offset + fwhm
    if len(indices) > 0:
        first_index = indices[0]
        last_index = indices[-1]
        # difference between them to find the width of the detector
        fwhm_width = pos[last_index] - pos[first_index]
    else:
        #print("No indices found where current is greater than offset + fwhm.")
        fwhm_width = None
    # find the two points closest to amplitude * (1-1/e^2) --> this give the standard deviation/beam radius
    stdev = amplitude * (1 - 1/np.e**2)
    # find the indice closest to each of the first/last indices where the current is greater than the standard deviation
    stdev_first_index = np.where(current >= offset + stdev)[0][0] if np.any(current >= offset + stdev) else None
    stdev_last_index = np.where(current >= offset + stdev)[0][-1] if np.any(current >= offset + stdev) else None
    # finding the average beam radius
    beam_radius = ((pos[stdev_first_index] - pos[first_index]) + np.abs(pos[stdev_last_index] - pos[last_index])) / 2 if stdev_first_index is not None and stdev_last_index is not None else None
    center = (pos[first_index] + pos[last_index]) / 2
    p0 = [offset, amplitude, pos[first_index], pos[last_index], beam_radius]
    return center, pos, current, current_err, p0

def fit(pos, current, current_err, p0):
    # Plot the joined erf fit
    try:
        popt, _ = curve_fit(joined_erf, pos, current, p0=p0)
        #print(f"Fitted parameters: {popt}")
    except Exception as e:
        print(f"Error fitting joined erf: {e}")
    # Plot the fitted curve
    pos_line = np.linspace(pos.min(), pos.max(), 1000)
    fitted_curve = joined_erf(pos_line, *popt)
    # Make a string of the fit parameters for the legend
    fit_param = "Intensity = %.2e A"%popt[1] + "\nDevice Width = %.1f mm"%(popt[3]-popt[2]) + "\nDevice Center = %.1f mm"%(popt[3]/2+popt[2]/2) + "\nBeam Radius = %.2f mm"%popt[4]

    def residual_calc(x, y, y_err, param, func):
        res = (y - func(x, *param))/y_err
        return res
    
    residuals = residual_calc(pos, current, current_err, popt, joined_erf)
    
    def chi2_calc(res):
        X2 = np.sum(res**2)/len(res)
        return X2
    
    chi2 = chi2_calc(residuals)
    #print(chi2)

    fit_param = "Intensity = %.2e A"%popt[1] + "\nDevice Width = %.1f mm"%(popt[3]-popt[2]) + "\nDevice Center = %.1f mm"%(popt[3]/2+popt[2]/2) + "\nBeam Radius = %.2f mm"%popt[4] + "\nChi2 = %.2f"%chi2


    # Trying to take the derivative to get the gaussian didn't work
    '''
    # How come these aren't coming out as gaussian? --> this will give me the standard deviation
    x_deriv = np.gradient(current1, y_pos)
    # smooth the data with a low pass filter
    x_deriv = scipy.signal.savgol_filter(x_deriv, 8, 3)
    # filter with a boxcar
    #x_deriv = scipy.signal.convolve(np.gradient(current1, y_pos), np.ones(5)/5, mode='same')
    #y_deriv = np.gradient(current2, x_pos)
    '''

    return pos_line, fitted_curve, fit_param, popt, residuals

'''
def topbeam(x,*p):
    return p[1]/2*(1-special.erf(np.sqrt(2)*(p[2]-x)/p[3]))+p[0]
def botbeam(x,*p):
    return p[1]/2*(1-special.erf(np.sqrt(2)*(x-p[2])/p[3]))+p[0]
def beam(x,*p):
    tran = (p[2]+p[4])/2
    beamintensity = []
    for X in x:
        if X < tran:
            beamintensity.append(topbeam(X,p[0],p[1],p[2],p[3]))
        else:
            beamintensity.append(botbeam(X,p[0],p[1],p[4],p[3]))
    return np.array(beamintensity)
'''
r'''
def profile_old(i,data):
    # Check to see what profile is in the data
    if len(np.unique(data.Xproc)) > 1:
        posdata = data.Xproc
        prof = 'x'
    if len(np.unique(data.Yproc)) > 1:
        posdata = data.Yproc
        prof = 'y'
    # Find the parameters for the ERF fit
    # Locate the three regions about the two inflection points (sensor edges)
    i0 = np.where(np.gradient(data.Iproc[str(i)],posdata)==max(np.gradient(data.Iproc[str(i)],posdata)))[0][0]
    i1 = np.where(np.gradient(data.Iproc[str(i)],posdata)==min(np.gradient(data.Iproc[str(i)],posdata)))[0][0]
    # Find the location of the edges
    s0 = posdata[i0]
    s1 = posdata[i1]
    #print(s0,s1)
    # Locate the index corresponding to the center of the beam
    i2 = np.where(posdata >= s0+(s1-s0)/2)[0][0]
    # Estimate the max power and offset
    I0 = np.mean([x for x in data.Iproc[str(i)][i0:i1] if str(x) != 'nan']) # Max
    I1 = np.mean([x for x in data.Iproc[str(i)][:i0] if str(x) != 'nan']) # Offset
    I2 = np.mean([x for x in data.Iproc[str(i)][i1:] if str(x) != 'nan']) # Offset
    # Estimate the indices corresponding the beam radius
    i3 = np.where(data.Iproc[str(i)] >= I1)[0][0]
    i4 = np.where(data.Iproc[str(i)] >= I0)[0][0]
    i5 = np.where(data.Iproc[str(i)] >= I0)[0][-1]
    i6 = np.where(data.Iproc[str(i)] >= I2)[0][-1]
    # Estimate the beam radius
    w0 = (posdata[i4]-posdata[i3])/2
    w1 = (posdata[i6]-posdata[i5])/2
    # Find the first and last crossing of the FWHM
    FWHM = np.where(data.Iproc[str(i)] >= max(data.Iproc[str(i)])/2)[0]
    x0 = posdata[FWHM[0]]
    x1 = posdata[FWHM[-1]]
    # Find the centroid of the device
    #devI = data.Iproc[FWHM]
    #devIerr = data.Ieproc[FWHM]
    #devpos = posdata[FWHM]
    #devx = np.dot(devpos,devI)/np.sum(devI)
    # Find the leakage current and the beam intensity
    l0 = min([x for x in data.Iproc[str(i)] if str(x) != 'nan'])
    a0 = max([x for x in data.Iproc[str(i)] if str(x) != 'nan'])# - l0
    # Find the width of the beam
    # Find where the beam intensity is 1 + 1/e^2 of the minimum
    d0 = posdata[np.where(data.Iproc[str(i)] >= l0+a0/np.e**2)[0][0]]
    d1 = posdata[np.where(data.Iproc[str(i)] >= l0+a0/np.e**2)[0][-1]]
    r0 = np.abs(x0-d0)
    r1 = np.abs(x1-d1)
    # Organise the best-guess parameters into an array
    P0 = [l0,a0,x0,r0]#[I0,s0,w0,I1]
    P1 = [l0,a0,x1,r1]#[I0,s1,w1,I2]
    P = [l0,a0,x0,r0,x1]#,r1]
    # Split the data into sections corresponding to each side of the sensor and remove nan
    toppos, topsignal, toperror, botpos, botsignal, boterror = [],[],[],[],[],[]
    for j in range(len(data.Iproc[str(i)][:i2])):
        if str(data.Iproc[str(i)][:i2][j]) != 'nan':
            toppos.append(posdata[:i2][j])
            topsignal.append(data.Iproc[str(i)][:i2][j])
            toperror.append(data.Ieproc[str(i)][:i2][j])
    for j in range(len(data.Iproc[str(i)][i2:])):
        if str(data.Iproc[str(i)][i2:][j]) != 'nan':
            botpos.append(posdata[i2:][j])
            botsignal.append(data.Iproc[str(i)][i2:][j])
            boterror.append(data.Ieproc[str(i)][i2:][j])
    toppos, topsignal, toperror, botpos, botsignal, boterror = np.array(toppos), np.array(topsignal), np.array(toperror), np.array(botpos), np.array(botsignal), np.array(boterror)
    boterror = np.sqrt(boterror**2 + (botsignal*1E-2)**2)
    # Combine the data into a single array
    posdata_ = np.concatenate((toppos,botpos))
    signal_ = np.concatenate((topsignal,botsignal))
    err_ = np.concatenate((toperror,boterror))
    print(signal_,err_)
    # Fit the data
    top_popt, top_pcov = curve_fit(topbeam,toppos,topsignal,p0=P0)#,sigma=toperror)
    bot_popt, bot_pcov = curve_fit(botbeam,botpos,botsignal,p0=P1)#,sigma=boterror)
    popt, pcov = curve_fit(beam,posdata_,signal_,p0=P)#,sigma=err_)
    def quickplot():
        figure, axis = plt.subplots(2,1)#,dpi=200,figsize=(12,12))
        axis[0].errorbar(toppos,topsignal,yerr=toperror,fmt='.')
        axis[0].errorbar(botpos,botsignal,yerr=boterror,fmt='.')
        axis[0].plot(toppos,topbeam(toppos,*top_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm'%(P0[1],P0[2],P0[3]))
        axis[0].plot(botpos,botbeam(botpos,*bot_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm'%(P1[1],P1[2],P1[3]))
        axis[1].errorbar(posdata_,signal_,yerr=err_,fmt='.')
        axis[1].plot(posdata_,beam(posdata_,*popt),label='P0 = %.2e A\nx0 = %.1f mm\nx1 = %.1f mm\nw0 = %.2f mm'%(popt[1],popt[2],popt[4],popt[3]))
        plt.show()
    #quickplot()
    # Calculate the residuals
    topres = (topsignal-topbeam(toppos,*top_popt))/toperror
    botres = (botsignal-botbeam(botpos,*bot_popt))/boterror
    res = (signal_-beam(posdata_,*popt))/err_
    # Calculate the chi2
    top_X2 = np.sum(topres**2)/(len(topsignal)-len(P0))
    bot_X2 = np.sum(botres**2)/(len(botsignal)-len(P1))
    X2 = np.sum(res**2)/(len(signal_)-len(P))
    print('IV Channel: Reduced chi-squared = %.2f' % X2)
    while X2 > 1:
        err_ = err_*1.1
        popt, pcov = curve_fit(beam,posdata_,signal_,sigma=err_,p0=P)
        # What is the reduced chi-squared value for the fit?
        res = (signal_-beam(posdata_,*popt))/err_
        X2 = np.sum(res**2)/(len(signal_)-len(P))
    print('IV Channel: Reduced chi-squared = %.2f' % X2)
    # Calculate the device edges and location using the fit
    FWHM = np.where(beam(posdata_,*popt)>(popt[0]+popt[1]/2))[0]
    x0 = posdata_[FWHM[0]]
    x1 = posdata_[FWHM[-1]]
    dev0 = np.dot(signal_[FWHM], posdata_[FWHM])/np.sum(signal_[FWHM])
    print('Device centroid: %.2f mm' % dev0)
    print('Device edges: %.2f mm, %.2f mm' % (x0,x1))
    print('Device width: %.2f mm' % (x1-x0))
    def quickplot():
        # Plot the data and fit
        figure, axis = plt.subplots(2,1)#,dpi=200,figsize=(12,12))
        axis[1].errorbar(posdata_,signal_,yerr=err_,fmt='.')
        #axis[1].plot(posdata_,beam(posdata_,*popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm\nx1 = %.1f mm\nw1 = %.2f mm'%(popt[0],popt[2],popt[3],popt[4],popt[5]))
        axis[1].plot(posdata_,beam(posdata_,*popt),label='P0 = %.2e A\nx0 = %.1f mm\nx1 = %.1f mm\nw0 = %.2f mm\ndev0 = %.2f mm'%(popt[0],x0,x1,popt[3],dev0))
        #axis[1].plot(toppos,topbeam(toppos,*top_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm'%(top_popt[1],top_popt[2],top_popt[3]))
        #axis[1].plot(botpos,botbeam(botpos,*bot_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm'%(bot_popt[1],bot_popt[2],bot_popt[3]))
        # Plot the residuals from the fit
        axis[0].plot(posdata_,res,'o',label=r'$\chi^2$ = '+'%.2f'%X2)
        #axis[0].plot(toppos,topres,'o',label=r'$\chi^2$ = '+'%.2f'%top_X2)
        #axis[0].plot(botpos,botres,'o',label=r'$\chi^2$ = '+'%.2f'%bot_X2)
        # Plot the best-guess parameters
        #axis[1].plot(posdata_,beam(posdata_,*P))
        #axis[1].plot(toppos,topbeam(toppos,*P0))
        #axis[1].plot(botpos,botbeam(botpos,*P1))
        #axis[0].plot(posdata,np.gradient(data.Iproc,posdata))
        # Format the plots
        for ax in axis:
            if prof == 'x':
                ax.set_xlabel('X-position [mm]')
            if prof == 'y':
                ax.set_xlabel('Y-position [mm]')
            ax.set_xlim(posdata[0],posdata[-1])
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)
        axis[0].set_ylabel('Normalised Residuals')
        axis[1].set_ylabel('PD Current [pA]')
        axis[1].legend()
        plt.tight_layout()
        plt.show()
        #figure.savefig(os.path.join(PLOTS,'%s_XYI.svg'))
    quickplot()
    def quickplot():
        # Plot the data and fit
        figure, axis = plt.subplots(2,1)#,dpi=200,figsize=(12,12))
        axis[1].errorbar(posdata,data.Iproc,yerr=data.Ieproc,fmt='.')
        axis[1].plot(toppos,topbeam(toppos,*top_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm'%(top_popt[1],top_popt[2],top_popt[3]))
        axis[1].plot(botpos,botbeam(botpos,*bot_popt),label='P0 = %.2e A\nx0 = %.1f mm\nw0 = %.2f mm'%(bot_popt[1],bot_popt[2],bot_popt[3]))
        # Plot the residuals from the fit
        axis[0].plot(toppos,topres,'o',label=r'$\chi^2$ = '+'%.2f'%top_X2)
        axis[0].plot(botpos,botres,'o',label=r'$\chi^2$ = '+'%.2f'%bot_X2)
        # Plot the best-guess parameters
        #axis[1].plot(toppos,topbeam(toppos,P0))
        #axis[1].plot(botpos,botbeam(botpos,P1))
        #axis[0].plot(posdata,np.gradient(data.Iproc,posdata))
        # Format the plots
        for ax in axis:
            if prof == 'x':
                ax.set_xlabel('X-position [mm]')
            if prof == 'y':
                ax.set_xlabel('Y-position [mm]')
            ax.set_xlim(posdata[0],posdata[-1])
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)
        axis[0].set_ylabel('Normalised Residuals')
        axis[1].set_ylabel('PD Current [pA]')
        axis[1].legend()
        plt.show()
        #figure.savefig(os.path.join(PLOTS,'%s_XYI.svg'))
    #quickplot()
'''

def plot(x_pos, x_current, x_current_err, x_pos_line, x_fitted_curve, x_fit_param, x_residuals,
         y_pos, y_current, y_current_err, y_pos_line, y_fitted_curve, y_fit_param, y_residuals):
    figure, axis = plt.subplots(2, 2, sharex=True, figsize=(12, 7))

    axis[0, 0].errorbar(x_pos, x_current, yerr=x_current_err, fmt='o', color='red')
    axis[0, 0].plot(x_pos_line, x_fitted_curve, '-', color='green', label=x_fit_param)
    axis[0, 0].set_xlabel("Y Position (mm)")
    axis[0, 0].set_ylabel("Current (A)")
    axis[0, 0].set_title('X Profile (fixed x)')
    axis[0, 0].legend()

    axis[1, 0].errorbar(x_pos, x_residuals, yerr = np.ones(len(x_pos)), fmt = '.')

    axis[0, 1].errorbar(y_pos, y_current, yerr=y_current_err, fmt='o', color='red')
    axis[0, 1].plot(y_pos_line, y_fitted_curve, '-', color='green', label=y_fit_param)
    axis[0, 1].set_xlabel("X Position (mm)")
    axis[0, 1].set_ylabel("Current (A)")
    axis[0, 1].set_title('Y Profile (fixed y)')
    axis[0, 1].legend()

    axis[1, 1].errorbar(y_pos, y_residuals, yerr = np.ones(len(y_pos)), fmt = '.')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x_csv = r'C:\Users\ddaya\OneDrive - Yale University\Darroch Research\SiPM-Scanning\Refined Code\data\first\meas_2.7_27.4_xline_scan_1753113067.csv'
    y_csv = r'C:\Users\ddaya\OneDrive - Yale University\Darroch Research\SiPM-Scanning\Refined Code\data\first\meas_2.7_27.4_yline_scan_1753113067.csv'
    x_center, x_pos, x_current, x_current_err, x_p0 = profile(x_csv, 1)
    y_center, y_pos, y_current, y_current_err, y_p0 = profile(y_csv, 0)
    x_pos_line, x_fitted_curve, x_fit_param, x_popt, x_residuals = fit(x_pos, x_current, x_current_err, x_p0)
    y_pos_line, y_fitted_curve, y_fit_param, y_popt, y_residuals = fit(y_pos, y_current, y_current_err, y_p0)
    plot(x_pos, x_current, x_current_err, x_pos_line, x_fitted_curve, x_fit_param, x_residuals,
         y_pos, y_current, y_current_err, y_pos_line, y_fitted_curve, y_fit_param, y_residuals)
