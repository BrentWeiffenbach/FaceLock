import pigpio
import time
import sys
import threading

def main():
    # GPIO mapping
    ELECTROMAGNET_GPIO = 17
    PIR_SENSOR_GPIO = 24
    LOWER_ARM_SERVO_GPIO = 27
    UPPER_ARM_SERVO_GPIO = 22
    DEADLOCK_MICROSERVO_GPIO = 23
    SERVO_PINS = [LOWER_ARM_SERVO_GPIO, UPPER_ARM_SERVO_GPIO, DEADLOCK_MICROSERVO_GPIO]

    POSITION_A_PULSEWIDTH = 500
    POSITION_B_PULSEWIDTH = 1500
    STEP_DELAY_SECONDS = 1.0


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

    def set_angle(pulsewidth, pin):
        pi.set_servo_pulsewidth(pin, pulsewidth)

    def set_all_servos(pulsewidth):
        for pin in SERVO_PINS:
            set_angle(pulsewidth, pin)
        print(f"All servos set to {pulsewidth} µs")

    def pir_monitor(stop_event):
        last_state = None
        while not stop_event.is_set():
            motion = pi.read(PIR_SENSOR_GPIO)
            if motion != last_state:
                if motion:
                    print("PIR: Motion detected!")
                else:
                    print("PIR: No motion.")
                last_state = motion
            time.sleep(0.1)

    stop_event = threading.Event()
    pir_thread = None

    try:
        print("Starting synchronized servo alternation test. Press Ctrl+C to quit.")
        print("PIR monitoring is running in a separate thread.")

        pir_thread = threading.Thread(target=pir_monitor, args=(stop_event,), daemon=True)
        pir_thread.start()

        while True:
            pi.write(ELECTROMAGNET_GPIO, 1)
            print("Electromagnet ON - moving all servos to Position A")
            set_all_servos(POSITION_A_PULSEWIDTH)
            time.sleep(STEP_DELAY_SECONDS)

            print("Electromagnet OFF - moving all servos to Position B")
            pi.write(ELECTROMAGNET_GPIO, 0)
            set_all_servos(POSITION_B_PULSEWIDTH)
            time.sleep(STEP_DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        stop_event.set()
        # Turn off everything
        pi.write(ELECTROMAGNET_GPIO, 0)
        pi.set_servo_pulsewidth(LOWER_ARM_SERVO_GPIO, 0)
        pi.set_servo_pulsewidth(UPPER_ARM_SERVO_GPIO, 0)
        pi.set_servo_pulsewidth(DEADLOCK_MICROSERVO_GPIO, 0)
        pi.stop()
        print("pigpio connection closed.")

if __name__ == "__main__":
    main()
