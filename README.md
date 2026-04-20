# SiPM-Scanning
Python codes to control stepper motors for a linear stage, picoammeter, and a digital multimeter to scan a SiPM to measure its saturation behavior along with a calibrated PD.

Look in the "Arduino-Stepper-Motors" repository for the arduino firware that I use here to control the stepper motors.

int_arduino
- Initializing a serial connection with arduino to control stepper motors

int_keithley
- Initializing a serial connection with Keithley 6487 (picoammeter)
- In my setup, I used this to bias the SiPM and measure the current of the calibrated photodiode
- Uses SCPI

int_siglent
- Initializing a serial connection with Siglent SDM3045X (digital multimeter)
- In my setup, I used this to measure the current of the SiPM
- Uses SCPI

raster_scan
- This will do a raster scan of the SiPM, taking both current measurements
- You can change the increments in which the linear stage moves in

twod_scan
- This will do do two line scans along roughly the center of the SiPM
- This will integrate the gaussian profile of the laser to give you a error function

darroch_error
- From the csv data obtained in twod_scan, this code with find the exact center of the SiPM
- It will also give you the error function with the FWHM, which will give you the dimensions of the SiPM

center_scan
- Using the center given in center_fit, this code will take a measurment at the center of the SiPM

calibration
- This is an initial full script to calibrate your setup
- Very similar to measurement, but can help you to decide the range of ND filters you can use

measurement
- This is an official script, where the raster can also be taken with every ND filter
