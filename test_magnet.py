import pigpio
import time
import sys

def main():
    magnet_pin = 17

    # Connect to the local Pi's pigpio daemon
    pi = pigpio.pi()
    
    if not pi.connected:
        print("Failed to connect to pigpio daemon.")
        print("Did you start it on the host with 'sudo pigpiod'?")
        sys.exit(1)

    print("Connected to pigpio daemon.")
    
    # Configure pin as an output
    pi.set_mode(magnet_pin, pigpio.OUTPUT)

    try:
        print("Starting electromagnet test on mosfet. Press Ctrl+C to quit.")
        while True:
            print("Electromagnet ON")
            pi.write(magnet_pin, 1)
            time.sleep(2)
            
            print("Electromagnet OFF")
            pi.write(magnet_pin, 0)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        # Turn off the magnet on exit
        pi.write(magnet_pin, 0)
        pi.stop()
        print("pigpio connection closed.")

if __name__ == "__main__":
    main()