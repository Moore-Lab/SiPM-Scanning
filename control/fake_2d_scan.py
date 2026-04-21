# measuring the SiPM's current with the keithley instead of the siglent

import time
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

import int_arduino
import int_keithley

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

def measure(x_mm, y_mm, keithley, csv=f"data/twod_scan_{int(time.time())}.csv"):
    # Perform an IV measurement at (x, y) using the siglent.
    print(f"Measuring at ({x_mm}, {y_mm})...")
    
    # Should run both instruments in parallel
    with ThreadPoolExecutor() as executor:
        future_pd = executor.submit(int_keithley.measure_current, keithley, 3, 1) #--> until I actually hook up PD
        pd_data = future_pd.result() #--> below is just a placeholder for PD data until I hook it up
    data = pd_data
     
    # Save to a single file with appended data
    if not os.path.exists("data"):
        os.makedirs("data")

    # Add column names if file does not exist
    file_exists = os.path.isfile(csv)
    with open(csv, 'a') as f:
        if not file_exists:
            f.write("x,y,sipm_current,sipm_std,sipm_stderr,sipm_time\n")
        s_c, s_std, s_stderr, s_t = data
        f.write(f"{x_mm},{y_mm},{s_c},{s_std},{s_stderr},{s_t}\n")

    print(f"Appended IV data at ({x_mm}, {y_mm}) to {csv}")

# --------------------------------------------------------------------
# FUNCTION: Line scans across SiPM
# --------------------------------------------------------------------

def xline_scan(x, y1, y2, ds, func, ser, keithley, csv=f"data/line_scan_{int(time.time())}.csv"):

    for y_idx in np.append(np.arange(y1, y2, ds), y2):  # Loop through the array with the last value always in (regardless of divisibility)

        move_to(x, y_idx, ser) # Move to this (x, y) position
        func(x, y_idx, keithley, csv)

    print("Finished x line scan of SiPM.")

def yline_scan(y, x1, x2, ds, func, ser, keithley, csv=f"data/line_scan_{int(time.time())}.csv"):

    for x_idx in np.append(np.arange(x1, x2, ds), x2):  # Loop through the array with the last value always in (regardless of divisibility)

        move_to(x_idx, y, ser) # Move to this (x, y) position
        func(x_idx, y, keithley, csv)

    print("Finished y line scan of SiPM.")


# --------------------------------------------------------------------
# FUNCTION: Return stage to (0, 0) after scanning
# --------------------------------------------------------------------
def flush():
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

        int_keithley.set_voltage(keithley, -27.4)  # Set voltage to -27.9 V

        xcsv=f"data/x_scan_{int(time.time())}.csv"
        ycsv=f"data/y_scan_{int(time.time())}.csv"
        xline_scan(5, 0, 10, .1, measure, ser, keithley, xcsv)
        yline_scan(5, 0, 10, .1, measure, ser, keithley, ycsv)
        flush()   # Return to home position when done
    except KeyboardInterrupt:
        print("\n Scan aborted by user.")
    finally:
        ser.close()         # Close the serial connection
        print("Serial connection closed.")
