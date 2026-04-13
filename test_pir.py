import pigpio
import time
import sys

def main():
    pir_pin = 24

    # Connect to the local Pi's pigpio daemon
    pi = pigpio.pi()
    
    if not pi.connected:
        print("Failed to connect to pigpio daemon.")
        print("Did you start it on the host with 'sudo pigpiod'?")
        sys.exit(1)

    print("Connected to pigpio daemon.")
    
    # Configure pin as an input
    pi.set_mode(pir_pin, pigpio.INPUT)

    try:
        print("Starting PIR sensor test. Press Ctrl+C to quit.")
        while True:
            motion = pi.read(pir_pin)
            if motion:
                print("Motion detected!")
            else:
                print("No motion.")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        pi.stop()
        print("pigpio connection closed.")

if __name__ == "__main__":
    main()