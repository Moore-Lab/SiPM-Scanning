import time
import numpy as np
import time
import os

import int_arduino
import int_keithley
import int_siglent
import twod_scan
import raster_scan
import center_scan
import center_fit

if __name__ == "__main__":
    nd = 2
    try:
        ser = int_arduino.main()  # Initialize Arduino connection --> shoould work that it's only once (one homing)
        keithley = int_keithley.initialize_keithley()
        siglent = int_siglent.initialize_siglent()

        int_keithley.set_voltage(keithley, -27.4)  # Set voltage to -27.9 V
        timestamp = int(time.time())  # Create a single timestamp to label csv files

        csv_raster=f"data/meas_{nd}_raster_scan_{timestamp}.csv"
        raster_scan.snake_scan(0, 7, 0, 7, .1, raster_scan.measure, ser, keithley, siglent, csv_raster)  
        
        xcsv=f"data/meas_{nd}_xline_scan_{timestamp}.csv"
        ycsv=f"data/meas_{nd}_yline_scan_{timestamp}.csv"
        twod_scan.xline_scan(3.5, 0, 7, .1, twod_scan.measure, ser, keithley, siglent, xcsv)
        twod_scan.yline_scan(3.5, 0, 7, .1, twod_scan.measure, ser, keithley, siglent, ycsv)

        csv_center=f"data/meas_{nd}_center_scan_{timestamp}.csv"
        center = center_fit.find_center(xcsv, ycsv) # Function that will tell me the center of the SiPM
        center_scan.center_scan(center[1], center[0], center_scan.measure, ser, keithley, siglent, csv_center) #reverse center because it gives it in reverse
        raster_scan.flush(ser)   # Return to home position when done
    except KeyboardInterrupt:
        print("\n Scan aborted by user.")
    finally:
        ser.close()         # Close the serial connection
        print("Serial connection closed.")