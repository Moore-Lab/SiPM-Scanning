# SiPM-Scanning
Python codes to control stepper motors for a linear stage, picoammeter, and a digital multimeter to scan a SiPM to measure its saturation behavior along with a calibrated PD.

Look in the "Arduino-Stepper-Motors" repository for the arduino firware that I use here to control the stepper motors.

int_arduino
- Initializing a serial connection with arduino to control stepper motors

int_keithley
- Initializing a serial connection with Keithley 6487 (picoammeter)
- In my setup, I used this to bias the SiPM and measure the current of the calibrated photodiode

int_siglent
- Initializing a serial connection with Siglent (digital multimeter)
- In my setup, I used this to measure the current of the SiPM
