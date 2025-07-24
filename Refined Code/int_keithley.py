import pyvisa
import time
import numpy as np
import os

def initialize_keithley(resource_name='ASRL4::INSTR'):
    rm = pyvisa.ResourceManager()
    keithley = rm.open_resource(resource_name=resource_name, read_termination='\r', write_termination="\n", timeout=60000)

    # Set voltage source mode
    keithley.write("*RST")                          # Reset the instrument
    keithley.write("SYST:ZCH OFF")               # Turn off zero check
    keithley.write("SENS:FUNC 'CURR'")           # Set to measure current
    
    return keithley

def test_connection(keithley):
    try:
        print("Testing connection...")
        response = keithley.query("*IDN?")
        print("Instrument connected:", response)
        return True
    except Exception as e:
        print("Connection failed:", e)
        return False
'''
def set_voltage(keithley, voltage=-27.9):
    # Set voltage source mode
    keithley.write("*RST")                          # Reset the instrument
    keithley.write("SYST:ZCH OFF")               # Turn off zero check
    keithley.write("SOUR:FUNC VOLT")            # Source voltage
    keithley.write("SOUR:VOLT:MODE FIX")         # Set voltage mode
    keithley.write("SOUR:VOLT:RANG 50")         # Set voltage range
    keithley.write("SOUR:VOLT:ILIM 2.5e-3")   # Set current limit to 2.5 mA
    keithley.write(f"SOUR:VOLT {voltage}")              # Set desired negative voltage (e.g., -5 V)

    # Enable the output
    keithley.write("SOUR:VOLT:STAT ON")
'''
    
def set_current(keithley):
    try:
        keithley.write("SENS:FUNC 'CURR'")
    except Exception as e:
        print("Failed to set function to current:", e)
        return None

# Function to measure the current of the PD, which will have no bias
def measure_current(keithley, n_measurements=5, delay=0.5):
    """
    While sourcing voltage, measure current using the instrument's statistics engine.
    Returns: np.array([mean, std, stderr, timestamp])
    """
    # Prepare buffer and measurement settings
    keithley.write("FORM:ELEM READ,VSO,TIME")
    keithley.write("SYST:ZCH OFF") # Just in case, turning off zero check again
    keithley.write(f"TRIG:COUN {n_measurements}")  # Set trigger count
    keithley.write(f"TRAC:POIN {n_measurements}")  # Set buffer size
    keithley.write("TRAC:FEED SENS")             # Store raw input readings in buffer
    keithley.write("TRAC:FEED:CONT NEXT")  

    # Trigger readings
    keithley.write("INIT")
    time.sleep(delay)  # Wait for the instrument to fill the buffer

    msmt_time = time.time()

    # Get mean from the instrument's statistic engine
    keithley.write("CALC3:FORM MEAN")
    mean_val = float(keithley.query("CALC3:DATA?"))

    # Get standard deviation from the instrument's statistic engine
    keithley.write("CALC3:FORM SDEV")
    std_val = float(keithley.query("CALC3:DATA?"))

    stderr = std_val / np.sqrt(n_measurements)

    print(f"Current: {mean_val:.4e} A, Std: {std_val:.4e} A, StdErr: {stderr:.4e} A, Time: {msmt_time:.2f}")

    return np.array([mean_val, std_val, stderr, msmt_time])


def get_voltage(keithley, n_measurements=1):
    """
    Get the voltage reading from the Keithley.
    Returns: np.array([mean_voltage, std_voltage, stderr_voltage, timestamp])
    """
    voltages = []
    for _ in range(n_measurements):
        voltage = float(keithley.query("MEAS:VOLT?"))
        voltages.append(voltage)
        time.sleep(0.1)  # Small delay to allow for stable readings
    
    voltages = np.array(voltages)
    mean_voltage = np.mean(voltages)
    std_voltage = np.std(voltages)
    stderr_voltage = std_voltage / np.sqrt(n_measurements)
    msmt_time = time.time()
    
    print(f"Voltage: {mean_voltage:.4f} V, Std: {std_voltage:.4f} V, StdErr: {stderr_voltage:.4f} V, Time: {msmt_time:.2f}")
    
    return np.array([mean_voltage, std_voltage, stderr_voltage, msmt_time])

def precise_iv(inst, start_volt, stop_volt, step_volt, n_measurements=5):
    """
    Perform an IV (current-voltage) sweep with repeated current measurements at each voltage step.
    Uses the instrument's internal buffer, then calculates the mean and standard error as (std / √n).
    """
    nsteps = int((stop_volt - start_volt) / step_volt) + 1
    voltages = np.linspace(start_volt, stop_volt, nsteps)
    
    iv_data = []
    for volt in voltages:
        inst.write("*RST")
        inst.write("FORM:ELEM READ,VSO,TIME")
        inst.write(f"TRIG:COUN {n_measurements}") # Set trigger count
        inst.write(f"TRAC:POIN {n_measurements}") # Set buffer size
        inst.write("TRAC:FEED SENS")             # Store raw input readings in buffer
        inst.write("TRAC:FEED:CONT NEXT")        
        inst.write("SYST:ZCH OFF")               # Turn off zero check
        inst.write("SOUR:FUNC VOLT")
        inst.write("SOUR:VOLT:MODE FIX")
        inst.write("SOUR:VOLT:RANG 50")          # Ensure the range covers your target voltage
        inst.write("SENS:FUNC 'CURR'")
        inst.write("SOUR:VOLT:ILIM 2.5e-3") # Set current limit to 25 mA if working below 10 V


        
        # Apply voltage
        inst.write(f"SOUR:VOLT:LEV {volt:.4f}")
        inst.write("SOUR:VOLT:STAT ON")
        
        # Trigger readings
        inst.write("INIT")
        time.sleep(0.5)  # Wait for the instrument to fill the buffer

        #data = inst.read()
        #dvals = data.split(",")
        #msmt_time = float(dvals[2])
        msmt_time = time.time()#float(inst.query("CALC3:DATA?")) # Get the time of the last measurement

        # Get mean from the instrument's statistic engine
        inst.write("CALC3:FORM MEAN")             # Select mean statistic
        mean_val = float(inst.query("CALC3:DATA?"))

        # Get standard deviation from the instrument's statistic engine
        inst.write("CALC3:FORM SDEV")             # Select standard deviation statistic
        sdev_val = float(inst.query("CALC3:DATA?"))
        
        stderr = sdev_val / np.sqrt(n_measurements)

        inst.write("SOUR:VOLT:STAT OFF")
        
        iv_data.append([volt, mean_val, stderr, msmt_time])
        print(f"Voltage: {volt:.4f} V, Current: {mean_val:.4e} A, StdErr: {stderr:.4e} A, Time: {msmt_time:.2f}")

    #inst.write("SYST:LOC")
    return np.array(iv_data)

def save_data(data, filename="data/iv_data.csv"):
    """
    Save the IV data to a CSV file.
    """
    if not os.path.exists("data"):
        os.makedirs("data")
    np.savetxt(filename, data, delimiter=",", header="Voltage (V), Current (A), StdErr (A), Timestamp (s)", comments="")
    print(f"Data saved to {filename}")

# allows the script to be run directly if I want it to
if __name__ == "__main__":
    keithley = initialize_keithley()
    if test_connection(keithley):
        #data = precise_iv(keithley, -27.9, -27.9, 1, n_measurements=10)
        #print("Precise IV Sweep:", data)
        # Create a filename based on the current time
        # Get a unix timetimestamp
        #ts = time.time()
        #filename = "data/iv_data_%i.csv"%ts
        #save_data(data, "data/iv_data.csv")
        set_voltage(keithley, -27.4)  # Set voltage to -27.9 V
        #set_current(keithley)
        #print(keithley.query("SENS:FUNC?"))  # Should print 'CURR'
        measure_current(keithley, 5, 0.5) # Measure current 10 times with 1 second delay