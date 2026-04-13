import pigpio
import time
import sys

def main():
    servo_pin = 23

    # Connect to the local Pi's pigpio daemon
    pi = pigpio.pi()
    
    if not pi.connected:
        print("Failed to connect to pigpio daemon.")
        print("Did you start it on the host with 'sudo pigpiod'?")
        sys.exit(1)

    print("Connected to pigpio daemon.")
    
    # In pigpio, servo pulsewidths are in microseconds:
    # 500  = 0 degrees (approx)
    # 1500 = 90 degrees (center)
    # 2500 = 180 degrees (approx)
    # 0    = pulses off

    def set_angle(pulsewidth):
        print(f"Setting pulsewidth: {pulsewidth} µs")
        pi.set_servo_pulsewidth(servo_pin, pulsewidth)
        time.sleep(1)

    try:
        print("Starting servo sweep test. Press Ctrl+C to quit.")
        set_angle(500)   # ~0 degrees
        # set_angle(2500)  # ~180 degrees

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        # Turn off servo pulses
        pi.set_servo_pulsewidth(servo_pin, 0)
        pi.stop()
        print("pigpio connection closed.")

if __name__ == "__main__":
    main()
