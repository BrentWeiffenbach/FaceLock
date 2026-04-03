import pigpio
import time
import sys

def main():
    button_pin = 16

    # Connect to the local Pi's pigpio daemon
    pi = pigpio.pi()
    
    if not pi.connected:
        print("Failed to connect to pigpio daemon.")
        print("Did you start it on the host with 'sudo pigpiod'?")
        sys.exit(1)

    print("Connected to pigpio daemon.")
    
    # Configure pin as an input and enable the internal pull-up resistor
    pi.set_mode(button_pin, pigpio.INPUT)
    pi.set_pull_up_down(button_pin, pigpio.PUD_UP)

    try:
        print("Starting button test. Press Ctrl+C to quit.")
        last_state = pi.read(button_pin)
        
        while True:
            current_state = pi.read(button_pin)
            if current_state != last_state:
                # With a pull-up resistor, a reading of 0 means the button is pressed (connected to ground)
                if current_state == 0:
                    print("Button pressed!")
                else:
                    print("Button released!")
                last_state = current_state
            
            # Short sleep to slightly debounce
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        pi.stop()
        print("pigpio connection closed.")

if __name__ == "__main__":
    main()