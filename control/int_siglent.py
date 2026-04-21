import pyvisa
import time
import numpy as np

def initialize_siglent(resource_name="USB0::0xF4EC::0x1205::SDM34HBQ801881::INSTR"):
    rm = pyvisa.ResourceManager()
    siglent = rm.open_resource(resource_name, timeout=10000)
    
    # Identification
    print("Connected to:", siglent.query("*IDN?").strip())
    
    # Optional: reset and clear
    siglent.write("*RST")  # Reset the DMM
    siglent.write("*CLS")  # Clear status
    siglent.write("SYST:AZER ON")  # autozero on

    # Set measurement mode to DC current
    siglent.write('FUNC "CURR:DC"')      # Set function to DC current
    siglent.write("CURR:DC:RANG: AUTO ON")  # Changed from auto to 6mA range

    return siglent

def precise_current(inst, n_measurements=1, delay=1.0):
    """
    Take n_measurements DC current readings from the Siglent, return mean, std, stderr, and timestamp.
    Returns: np.array([mean, std, stderr, timestamp])
    """
    currents = []
    for _ in range(n_measurements):
        current = float(inst.query("READ?"))
        currents.append(current)
        #inst.write("*CLS") # trying this line and the one below to see if it helps ragged edges
        #inst.write("SYST:AZER ON")
        #time.sleep(delay)
    currents = np.array(currents)
    mean_val = np.mean(currents)
    std_val = np.std(currents)
    stderr = std_val / np.sqrt(n_measurements)
    msmt_time = time.time()
    print(f"Current: {mean_val:.4e} A, Std: {std_val:.4e} A, StdErr: {stderr:.4e} A, Time: {msmt_time:.2f}")
    return np.array([mean_val, std_val, stderr, msmt_time])

def main():
    # Replace with the VISA resource name for your device (use rm.list_resources() to find it)
    # Could be something like 'USB0::0xF4EC::0xEE38::SDM34xxxxxxx::INSTR' or 'TCPIP0::192.168.1.123::INSTR'
    # Run pyvisa.ResourceManager().list_resources() to find it and it will change if using LAN connection over USB
    resource_name = "USB0::0xF4EC::0x1205::SDM34HBQ801881::INSTR"

    try:
        siglent = initialize_siglent(resource_name)
        precise_current(siglent, 10, 1.0)
    except Exception as e:
        print("Error communicating with instrument:", e)

if __name__ == "__main__":
    main()