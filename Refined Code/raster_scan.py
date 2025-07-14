import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

import int_arduino
import int_keithley
import int_siglent

#import logging

timestamp = int(time.time())

'''
# Set up logging to a file
logging.basicConfig(
    filename=f'scan_log_{timestamp}.txt',      # Log file name
    level=logging.INFO,           # Log level
    format='%(asctime)s %(levelname)s: %(message)s'
)
'''

# --------------------------------------------------------------------
# FUNCTION: Send move command to Arduino and wait for "OK"
# --------------------------------------------------------------------
def move_to(x_mm, y_mm, ser):
    """
    Sends a move command in millimeters to the Arduino and waits until movement is complete.
    """
    command = f"move {round(x_mm, 4)} {round(y_mm, 4)}\n"
    ser.write(command.encode())  # Send move command over serial
    print(f"Sent: {command.strip()}")
    #logging.info(f"Moved to ({round(x_mm, 1)}, {round(y_mm, 1)})...")

    # Wait for Arduino to confirm it finished moving
    while True:
        line = ser.readline().decode().strip()
        if line == "OK":
            print(f"Arrived at position ({x_mm}, {y_mm})")
            break
        elif line.startswith("POS"):
            print(f"Arduino actual position: {line[4:]}")
        elif line == "ERR":
            raise RuntimeError(f"Arduino reported error at ({x_mm}, {y_mm})")
        elif line:
            print(f"Arduino says: {line}")  # Other optional debug messages

# --------------------------------------------------------------------
# FUNCTION: Simulate taking a measurement (can be replaced later) --> Changed
# --------------------------------------------------------------------

def measure(x_mm, y_mm, keithley, siglent, csv=f"data/center_{int(time.time())}.csv"):
    # Perform an IV measurement at (x, y) using the siglent.
    print(f"Measuring at ({x_mm}, {y_mm})...")

    #data = int_siglent.precise_current(siglent, 1, 1) --> forgot I am trying to also take PD measurements
    
    # Should run both instruments in parallel
    with ThreadPoolExecutor() as executor:
        future_sipm = executor.submit(int_siglent.precise_current, siglent, 5, 0.5)
        future_pd = executor.submit(int_keithley.measure_current, keithley, 5, 1) #--> until I actually hook up PD
        sipm_data = future_sipm.result()
        pd_data = future_pd.result() #--> below is just a placeholder for PD data until I hook it up
        #pd_data = [0,0,0,time.time()]
    data = np.concatenate((sipm_data, pd_data), axis=0)
     
    # Save to a single file with appended data
    if not os.path.exists("data"):
        os.makedirs("data")

    # Add column names if file does not exist
    file_exists = os.path.isfile(csv)
    with open(csv, 'a') as f:
        if not file_exists:
            f.write("x,y,sipm_current,sipm_std,sipm_stderr,sipm_time,pd_current,pd_std,pd_stderr,pd_time\n")
        s_c, s_std, s_stderr, s_t, p_c, p_std, p_stderr, p_t = data
        f.write(f"{x_mm},{y_mm},{s_c},{s_std},{s_stderr},{s_t},{p_c},{p_std},{p_stderr},{p_t}\n")

    print(f"Appended IV data at ({x_mm}, {y_mm}) to {csv}")

# --------------------------------------------------------------------
# FUNCTION: Snake-style scan across SiPM
# --------------------------------------------------------------------

def snake_scan(x1, x2, y1, y2, ds, func, ser, keithley, siglent, csv=f"data/raster_scan_{timestamp}.csv"):


    direction = 1  # Initial direction: left-to-right

    for y_idx in np.append(np.arange(y1, y2, ds), y2):  # Loop through the array with the last value always in (regardless of divisibility)

        # Determine scan order for current row (left-to-right or right-to-left)
        x_range = np.append(np.arange(x1, x2, ds),x2) if direction == 1 else np.append(np.arange(x2, x1, -ds), x1)

        for x_idx in x_range:
            try:
                move_to(x_idx, y_idx, ser)
                func(x_idx, y_idx, keithley, siglent, csv)
            except Exception as e:
                print(f"Error at ({x_idx:.1f}, {y_idx:.1f}): {e}")
                # Write a row of zeros to mark the failure
                with open(csv, 'a') as f:
                    f.write(f"{x_idx:.1f},{y_idx:.1f},0,0,0,{time.time()},0,0,0,{time.time()}\n")
        
        direction *= -1  # Reverse direction for next row

    print("Finished full scan of SiPM.")

# --------------------------------------------------------------------
# FUNCTION: Return stage to (0, 0) after scanning
# --------------------------------------------------------------------
def flush(ser):
    """
    Sends a 'flush' command to Arduino to return motors to limit switch position.
    """
    ser.write(b"flush\n")
    print("Sent: flush")

    while True:
        line = ser.readline().decode().strip()
        if line == "OK":
            print("Returned to limit switch")
            break
        elif line:
            print(f"Arduino: {line}")  # Other debug messages

# --------------------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------------------
# customize the ranges and increments

if __name__ == "__main__":
    try:
        ser = int_arduino.main()  # Initialize Arduino connection
        keithley = int_keithley.initialize_keithley()
        siglent = int_siglent.initialize_siglent()

        int_keithley.set_voltage(keithley, -27.4)  # Set voltage to -27.9 V

        csv=f"data/raster_scan_{int(time.time())}.csv"
        snake_scan(0, 10, 0, 10, .1, measure, ser, keithley, siglent, csv)        # Start scanning the full sensor
        flush(ser)   # Return to home position when done
    except KeyboardInterrupt:
        print("\n Scan aborted by user.")
    finally:
        ser.close()         # Close the serial connection
        print("Serial connection closed.")
