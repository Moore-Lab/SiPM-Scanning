import serial
import time

def main(com="COM3"):

    PORT = com                # Update this to match your Arduino port (e.g. '/dev/ttyACM0' on Linux)
    BAUDRATE = 9600              # Must match the Arduino Serial.begin() rate
    DELAY_SECONDS = 1            # Time to "measure" at each point (in seconds)

    # --------------------------------------------------------------------
    # CONNECT TO ARDUINO
    # --------------------------------------------------------------------

    try:
        # Establish serial connection to Arduino
        ser = serial.Serial(PORT, BAUDRATE, timeout=2)
        time.sleep(2)  # Wait for Arduino to finish resetting
        print(f"Connected to Arduino on port {PORT}")
    except serial.SerialException:
        raise RuntimeError(f"Could not connect to Arduino on {PORT}. Check your cable and port.")
    
    return ser

if __name__ == "__main__":
    ser = main()