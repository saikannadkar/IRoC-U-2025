import cv2
import numpy as np
import time
import random
import base64
import threading
import os
import logging
import unittest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_HEIGHT = 5
MIN_TEMP = -20
MARS_GRAVITY = 3.71
SOLAR_CHARGING_RATE = 5
WIND_THRESHOLD = 20
SUNLIGHT_AVAILABLE = True
DUST_STORM_THRESHOLD = 25
MIN_BATTERY_LEVEL = 40
CRITICAL_BATTERY_LEVEL = 50
PROPELLER_SPEED = 2500
DRONE_LEG_HEIGHT = 0.6
LIDAR_HEIGHT = 0.6
PRE_FLIGHT_CHECK_INTERVAL = 180
FLIGHT_DATA_FILE = "flight_data.txt"

class MarsDrone:
    def _init_(self):
        self.altitude = 0
        self.position = [0, 0]
        self.home_position = [0, 0]
        self.is_flying = False
        self.battery = 100
        self.temperature = random.randint(-80, 20)
        self.wind_speed = random.randint(0, 30)
        self.heater_on = False
        self.is_charging = False
        self.communication_online = True
        self.propeller_speed = PROPELLER_SPEED
        self.autonomous_mode = True
        self.manual_mode = False
        self.flight_data = []
        self.flight_start_time = None
        self.velocity = [0, 0]
        self.camera = cv2.VideoCapture(0)
        if not self.camera or not self.camera.isOpened():
            logging.error("Camera failed to initialize.")
            self.camera = None
        else:
            logging.info("Camera initialized successfully.")
        logging.info(f"Lidar sensor set at {LIDAR_HEIGHT}m above ground.")
        self.radio_transmitter = True
        self.lock = threading.Lock()
        logging.info("Radio transmitter initialized for communication.")

    def pre_flight_check(self):
        logging.info("Running pre-flight check...")
        if self.battery < MIN_BATTERY_LEVEL:
            logging.error("Battery too low for flight!")
            return False
        logging.info("Battery OK")
        logging.info("Camera OK" if self.camera else "Camera failure")
        logging.info("Communication OK" if self.radio_transmitter else "Communication failure")
        logging.info("All systems normal. Ready for takeoff!")
        return True

    def regulate_temperature(self):
        self.heater_on = self.temperature < MIN_TEMP
        logging.info("Heater activated." if self.heater_on else "Heater deactivated.")

    def takeoff(self, altitude=4):
        if not self.is_flying and self.pre_flight_check():
            self.altitude = min(altitude, MAX_HEIGHT)
            self.is_flying = True
            self.flight_start_time = time.time()
            logging.info(f"Drone taking off to {self.altitude} meters.")
            logging.info(f"Propeller speed set to {self.propeller_speed} RPM.")
            self.regulate_temperature()
            try:
                threading.Thread(target=self.perform_periodic_checks, daemon=True).start()
            except Exception as e:
                logging.error(f"Error starting periodic checks thread: {e}")
        else:
            logging.error("Takeoff aborted or drone already flying.")

    def perform_periodic_checks(self):
        while self.is_flying:
            time.sleep(PRE_FLIGHT_CHECK_INTERVAL)
            try:
                with self.lock:
                    logging.info("Performing in-flight pre-checks...")
                    self.pre_flight_check()
                    self.monitor_battery()
                    self.check_communication()
            except Exception as e:
                logging.error(f"Error during periodic checks: {e}")

    def find_safe_landing_spot(self):
        logging.info("AI-based terrain analysis in progress...")
        time.sleep(3)
        return True

    def land(self):
        if self.is_flying:
            if self.find_safe_landing_spot():
                logging.info("Hovering for 20 seconds before landing...")
                time.sleep(20)
                logging.info("Safe spot found. Landing...")
            else:
                logging.warning("No safe landing spot! Returning to home position...")
                self.position = self.home_position
            self.altitude = 0
            self.is_flying = False
            logging.info(f"Drone landed successfully. Flight time: {time.time() - self.flight_start_time:.2f} seconds.")
            self.save_flight_data()
        else:
            logging.error("Drone is already on the ground.")

    def charge_battery(self):
        if not self.is_flying:
            logging.info("Charging...")
            while self.battery < 100:
                time.sleep(1)
                self.battery = min(self.battery + SOLAR_CHARGING_RATE, 100)
            logging.info("Battery fully charged!")
        else:
            logging.error("Cannot charge while flying!")

    def monitor_battery(self):
        if self.battery < CRITICAL_BATTERY_LEVEL:
            logging.warning("Battery critical! Searching for safe landing...")
            if not self.find_safe_landing_spot():
                logging.warning("No safe landing spot! Returning to home position...")
                self.position = self.home_position
            self.land()
        else:
            logging.info("Battery at safe level.")

    def check_communication(self):
        if not self.communication_online:
            logging.warning("Communication lost! Returning to home position...")
            self.position = self.home_position
            self.land()

    def save_flight_data(self):
        with open(FLIGHT_DATA_FILE, "a") as file:
            file.write(str(self.flight_data) + "\n")
        logging.info("Flight data stored temporarily.")
        self.transmit_data()

    def transmit_data(self, data=None):
        if self.radio_transmitter:
            if data:
                logging.info(f"Transmitting data over radio: {data[:50]}...")
            else:
                logging.info("Transmitting flight data over radio...")
        else:
            logging.error("Radio transmitter failed.")
        time.sleep(2)
        try:
            os.remove(FLIGHT_DATA_FILE)
            logging.info("Flight data transmitted and deleted.")
        except Exception as e:
            logging.error(f"Error deleting flight data file: {e}")

    def capture_and_transmit_image(self):
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                img_str = base64.b64encode(buffer).decode('utf-8')
                self.transmit_data(img_str)
                logging.info("Image transmitted successfully. Deleting temporary data.")
                self.flight_data.clear()
            else:
                logging.error("Failed to capture image.")
        else:
            logging.error("No camera available.")

    def shutdown(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        logging.info("Drone system shutting down.")

    def handle_command(self, command):
        if command == "takeoff":
            self.takeoff()
        elif command == "land":
            self.land()
        elif command == "charge":
            self.charge_battery()
        elif command == "capture":
            self.capture_and_transmit_image()
        elif command == "shutdown":
            self.shutdown()
        else:
            logging.error("Unknown command.")

class TestMarsDrone(unittest.TestCase):
    def setUp(self):
        self.drone = MarsDrone()

    def test_pre_flight_check(self):
        self.drone.battery = 50
        self.assertTrue(self.drone.pre_flight_check())
        self.drone.battery = 30
        self.assertFalse(self.drone.pre_flight_check())

    def test_takeoff_and_land(self):
        self.drone.battery = 100
        self.assertFalse(self.drone.is_flying)
        self.drone.takeoff(4)
        self.assertTrue(self.drone.is_flying)
        self.drone.land()
        self.assertFalse(self.drone.is_flying)

    def test_battery_monitoring(self):
        self.drone.battery = 45
        self.drone.is_flying = True
        self.drone.flight_start_time = time.time() - 100
        self.drone.monitor_battery()
        self.assertFalse(self.drone.is_flying)

if _name_ == '_main_':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        drone = MarsDrone()
        while True:
            try:
                command = input("Enter command (takeoff, land, charge, capture, shutdown): ").strip().lower()
                if command == "shutdown":
                    drone.shutdown()
                    break
                else:
                    drone.handle_command(command)
            except Exception as e:
                logging.error(f"Error in command interface: {e}")