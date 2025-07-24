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
import darroch_error

if __name__ == "__main__":
    nd = 5.5
    bias = 27.4
    try:
        ser = int_arduino.main()  # Initialize Arduino connection --> shoould work that it's only once (one homing)
        keithley = int_keithley.initialize_keithley()
        siglent = int_siglent.initialize_siglent()

        # int_keithley.set_voltage(keithley, -27.4)  --> manually doing this with a seperate supply
        timestamp = int(time.time())  # Create a single timestamp to label csv files

        #csv_raster=f"data/first/meas_{nd}_{bias}_raster_scan_{timestamp}.csv"
        #raster_scan.snake_scan(0, 10, 0, 10, .1, raster_scan.measure, ser, keithley, siglent, csv_raster)  
        
        xcsv=f"data/first/meas_{nd}_{bias}_xline_scan_{timestamp}.csv"
        ycsv=f"data/first/meas_{nd}_{bias}_yline_scan_{timestamp}.csv"
        twod_scan.xline_scan(5, 0, 10, .1, twod_scan.measure, ser, keithley, siglent, xcsv)
        twod_scan.yline_scan(5, 0, 10, .1, twod_scan.measure, ser, keithley, siglent, ycsv)

        csv_center=f"data/first/meas_{nd}_{bias}_center_scan_{timestamp}.csv"
        #center = center_fit.find_center(xcsv, ycsv) # Function that will tell me the center of the SiPM
        x_center, x_pos, x_current, x_current_err, x_p0 = darroch_error.profile(xcsv, 1)
        y_center, y_pos, y_current, y_current_err, y_p0 = darroch_error.profile(ycsv, 0)
        center_scan.center_scan(y_center, x_center, center_scan.measure, ser, keithley, siglent, csv_center) #reverse center because it gives it in reverse
        raster_scan.flush(ser)   # Return to home position when done
    except KeyboardInterrupt:
        print("\n Scan aborted by user.")
    finally:
        ser.close()         # Close the serial connection
        print("Serial connection closed.")