# backend/child_monitor/core.py

import os
import time
from datetime import datetime

import cv2

from . import shared_state
from .detection import YoloV5PersonDetector
from .settings import ALERT_DELAY, RESOLUTION


def stop_server():
    import time

    time.sleep(1)
    os._exit(0)


def _log_with_timestamp(message: str):
    """
    Adds a timestamp to log messages
        :param message: Mensagem a ser registrada
        :return: None
    """
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {message}")


class ChildMonitor:
    """
    Child Monitoring Core Class

    This class implements the core logic of the SafeZone monitoring system.

    It is responsible for:
    - Selecting the camera input and allowing the user to define a monitoring
      zone via mouse input.
    - Detecting people in real-time using a YOLOv5 model.
    - Determining whether detected individuals are inside or outside the
      safe zone.
    - Managing alerts when a person leaves the zone for a defined duration.
    - Streaming the latest video frame to a shared state for external access.
    - Sending detection and alert data to a frontend via WebSocket.
    - Running the monitoring loop in a separate thread for non-blocking,
      real-time operation.
    - Handling resource cleanup and logging errors gracefully.

    The class is designed for multi-threaded environments and supports
    integration with FastAPI-based frontends.
    It initializes with default settings such as video resolution
    and alert delay time.
    """

    def __init__(self):
        """
        Initializes the ChildMonitor instance
            :return: None
        """
        self.cap = None
        self.monitoring = False
        self.person_outside_zone = False
        self.outside_start_time = None
        self.alert_triggered = False
        self.RESOLUTION = RESOLUTION
        self.ALERT_DELAY = ALERT_DELAY
        self.zone_coords = None
        self.selecting_zone = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.temp_frame = None
        self.detector = YoloV5PersonDetector()

    def find_available_camera(self):
        """
        Find an available camera
            :return: Index of the first available camera or None if none found
        """
        _log_with_timestamp("🔍 Searching for available cameras...")

        # List of indices to test
        camera_indices = [0, 1, 2, 3, 4, 5]

        for index in camera_indices:
            _log_with_timestamp(f"🔍 Testing camera index {index}...")
            cap = cv2.VideoCapture(index)

            if cap.isOpened():
                # Try to read a frame to confirm it's working
                ret, frame = cap.read()
                if ret:
                    _log_with_timestamp(f"✅ CCamera found at index {index}")
                    cap.release()
                    return index
                else:
                    _log_with_timestamp(
                        f"⚠️ CCamera {index} opened but can't read frames"
                    )
                    cap.release()
            else:
                _log_with_timestamp(f"❌ Could not open CCamera {index}")

        _log_with_timestamp("❌ No camera found")
        return None

    def select_zone_with_mouse(self, camera_index=None):
        """
        Allows the user to select the monitoring zone
            :param camera_index: Optional camera index to use
            :return: True if zone selected, False if canceled or error
        """
        if camera_index is None:
            camera_index = self.find_available_camera()
            if camera_index is None:
                return False

        _log_with_timestamp(
            f"📹 Starting camera {camera_index} for zone selection..."
        )
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            _log_with_timestamp("❌ Error: Could not open camera")
            return False

        # Configure resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUTION[1])

        _log_with_timestamp("📍 Click and drag to select the monitoring zone")
        cv2.namedWindow("Select Zone", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Select Zone", self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                _log_with_timestamp("❌ Error capturing frame")
                break

            # Resize frame if necessary
            frame = cv2.resize(frame, self.RESOLUTION)
            self.temp_frame = frame.copy()

            # Draw zone if selected
            if self.zone_coords:
                cv2.rectangle(
                    frame,
                    self.zone_coords[0],
                    self.zone_coords[1],
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Zone selected - Press ENTER to confirm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Click and drag to select zone",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Draw preview of selection
            if self.selecting_zone and self.ix >= 0 and self.iy >= 0:
                cv2.rectangle(
                    frame,
                    (self.ix, self.iy),
                    (self.fx, self.fy),
                    (255, 0, 0),
                    2,
                )

            cv2.imshow("Select Zone", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.zone_coords:  # ENTER
                cv2.destroyWindow("Select Zone")
                _log_with_timestamp(f"✅ Zone confirmed: {self.zone_coords}")
                return True
            elif key == 27:  # ESC
                cv2.destroyWindow("Select Zone")
                _log_with_timestamp("❌ Selection canceled")
                return False

    def mouse_callback(self, event, x, y, flags, param=None):
        """
        Callback for zone selection with mouse
            :param event: Mouse event
            :param x: Mouse X position
            :param y: Mouse Y position
            :param flags: Mouse flags
            :param param: Optional parameter required by OpenCV
            :return: None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting_zone = True
            self.ix, self.iy = x, y
            _log_with_timestamp(f"🖱️ Start zone selection at ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting_zone:
                self.fx, self.fy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting_zone = False
            self.fx, self.fy = x, y

            # Validate minimum zone size
            if abs(self.fx - self.ix) > 30 and abs(self.fy - self.iy) > 30:
                x1, y1 = min(self.ix, self.fx), min(self.iy, self.fy)
                x2, y2 = max(self.ix, self.fx), max(self.iy, self.fy)
                self.zone_coords = ((x1, y1), (x2, y2))
                _log_with_timestamp(f"📐 Zone selected: {self.zone_coords}")
            else:
                _log_with_timestamp("⚠️ Zone too small, please try again")
                self.zone_coords = None

    def run_monitoring(self, should_exit=None):
        """
        Main Monitoring Loop
            :param should_exit: Optional threading.Event to signal exit
            :return: True if monitoring started successfully, False if error
        """
        if not self.cap or not self.cap.isOpened():
            _log_with_timestamp("❌ Camera not initialized")
            return False

        if not self.zone_coords:
            _log_with_timestamp("❌ Zone not selected")
            return False

        self.monitoring = True
        _log_with_timestamp("🚀 Monitoring started!")

        cv2.namedWindow("SafeZone - Safety You Can See", cv2.WINDOW_AUTOSIZE)

        try:
            while (
                self.monitoring
                and (should_exit is None or not should_exit.is_set())
            ):
                ret, frame = self.cap.read()
                if not ret:
                    _log_with_timestamp("❌ Error capturing frame")
                    break

                # Resize frame
                frame = cv2.resize(frame, self.RESOLUTION)

                # Update shared frame for streaming
                with shared_state.frame_lock:
                    shared_state.latest_frame = frame.copy()

                # Check for people in the zone
                if self.zone_coords:
                    self.check_person_in_zone(frame)

                if self.alert_triggered:
                    cv2.putText(
                        frame,
                        "ALERT ACTIVE !!!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("SafeZone - Safety You Can See", frame)

                # Check if window was closed
                try:
                    if (
                        cv2.getWindowProperty(
                            "SafeZone - Safety You Can See",
                            cv2.WND_PROP_VISIBLE,
                        )
                        < 1
                    ):
                        _log_with_timestamp("🗂️ Window closed by user")
                        if should_exit:
                            should_exit.set()
                        break
                except cv2.error:
                    _log_with_timestamp("❌ Window was destroyed")
                    if should_exit:
                        should_exit.set()
                    break

                # Keyboard control
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _log_with_timestamp("⌨️ Key 'q' pressed, exiting...")
                    break

        except KeyboardInterrupt:
            _log_with_timestamp("⚠️ Monitoring interrupted by user")
        finally:
            self.cleanup()
        return True

    def check_person_in_zone(self, frame):
        """
        Check if there are people in the zone and manage alerts
            :param frame: Current video frame
            :return: None
        """
        try:
            # Detect persons in the frame
            persons = self.detector.detect_person_in_frame(frame)
            persons_with_zone_info = []
            for person in persons:
                x, y, w, h = person
                person_in_zone = self.is_person_in_zone(person)
                color = (0, 255, 0) if person_in_zone else (0, 0, 255)

                # Add zone information
                persons_with_zone_info.append(
                    {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "in_zone": person_in_zone,
                    }
                )

                # Draw rectangle around person
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Add status text
                status_text = (
                    "Inside Safe Zone"
                    if person_in_zone
                    else "Outside Safe Zone"
                )
                cv2.putText(
                    frame,
                    status_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            # Send detection data via WebSocket
            self.send_detection_data_to_frontend(persons_with_zone_info)

            person_in_zone = any(
                self.is_person_in_zone(person) for person in persons
            )

            current_time = time.time()

            if not person_in_zone:
                # Person outside the zone
                if not self.person_outside_zone:
                    self.person_outside_zone = True
                    self.outside_start_time = current_time
                # Check if the delay time has passed
                elif (
                    self.outside_start_time
                    and (
                        current_time - self.outside_start_time
                        >= self.ALERT_DELAY
                    )
                    and not self.alert_triggered
                ):
                    self.alert_triggered = True
                    delay = self.ALERT_DELAY
                    _log_with_timestamp(
                        f"🚨 ALARM ACTIVE! Person outside the safe zone for "
                        f"{delay}s+"
                    )
                    # Send alert via WebSocket (if available)
                    self.send_websocket_alert(
                        "start",
                        {
                            "message": (
                                ("Alarm started! Person outside the safe zone")
                            ),
                            "timestamp": current_time,
                        },
                    )
            else:
                # Person returned to the zone
                if self.person_outside_zone:
                    duration = (
                        current_time - self.outside_start_time
                        if self.outside_start_time
                        else 0
                    )
                    # Log only if outside for more time than ALERT_DELAY
                    if duration >= self.ALERT_DELAY:
                        _log_with_timestamp(
                            f"✅ Person returned to the safe zone "
                            f"(was outside for {duration:.1f}s)"
                        )

                    self.person_outside_zone = False
                    self.outside_start_time = None

                    if self.alert_triggered:
                        _log_with_timestamp(
                            "🔕 Alarm canceled! Person returned to safe zone"
                        )
                        self.alert_triggered = False

                        # Send cancellation via WebSocket
                        self.send_websocket_alert(
                            "stop",
                            {
                                "message": (
                                    (
                                        (
                                            "Alarm stopped! Person returned "
                                            "to safe zone"
                                        )
                                    )
                                ),
                                "timestamp": current_time,
                                "duration": duration,
                            },
                        )
            # Draw Safe Zone with color based on alarm status
            #   Green = Safe
            #   Yellow = person outside but no alarm yet
            #   Red = Alarm active
            if self.alert_triggered:
                zone_color = (0, 0, 255)  # Red - Alarm active
                zone_text = "ALARM ACTIVE !!!"
            elif self.person_outside_zone:
                zone_color = (0, 255, 255)  # Yellow - person outside
                zone_text = "Monitored Zone"
            else:
                zone_color = (0, 255, 0)  # Green - all normal
                zone_text = "Safe Zone"

            if self.zone_coords:
                cv2.rectangle(
                    frame,
                    self.zone_coords[0],
                    self.zone_coords[1],
                    zone_color,
                    3,
                )
                cv2.putText(
                    frame,
                    zone_text,
                    (self.zone_coords[0][0], self.zone_coords[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    zone_color,
                    2,
                )
        except Exception as e:
            _log_with_timestamp(f"❌ Error in detection: {e}")

    def send_websocket_alert(self, alert_type, data):
        """
        Sends an alert via WebSocket to the main server
            :param alert_type: Type of alert ("start" or "stop")
            :param data: Additional data to send with the alert
            :return: None
        """
        for attempt in range(3):  # 3 attempts to send the alert
            try:
                import requests

                response = requests.post(
                    "http://localhost:8000/send_alert",
                    json={"type": alert_type, **data},
                    timeout=3,
                )
                if response.status_code == 200:
                    return  # Sucess
            except Exception as e:
                _log_with_timestamp(f"⚠️ Attempt {attempt+1} failed: {e}")
                time.sleep(0.5)

        _log_with_timestamp(f"❌ Failed to send alert {alert_type}")

    def send_detection_data_to_frontend(self, persons_with_zone_info):
        """
        Sends detection data to the frontend via WebSocket
            :param persons_with_zone_info: List of detected
                                           persons with zone info
            :return: None
        """
        try:
            import requests

            detection_data = {
                "type": "detection",
                "persons": persons_with_zone_info,
                "zone": (
                    {
                      "x1": self.zone_coords[0][0] if self.zone_coords else 0,
                      "y1": self.zone_coords[0][1] if self.zone_coords else 0,
                      "x2": self.zone_coords[1][0] if self.zone_coords else 0,
                      "y2": self.zone_coords[1][1] if self.zone_coords else 0,
                    }
                    if self.zone_coords
                    else None
                ),
                "resolution": {
                    "width": self.RESOLUTION[0],
                    "height": self.RESOLUTION[1],
                },
                "alert_active": self.alert_triggered,
                "person_outside_zone": self.person_outside_zone,
                "timestamp": time.time(),
            }
            # Send to the main server endpoint
            requests.post(
                "http://localhost:8000/send_detection",
                json=detection_data,
                timeout=0.1,
            )
        except Exception:
            pass  # Ignore connection errors

    def is_person_in_zone(self, person):
        """
        Checks if a person is within the defined zone
            :param person: Tuple with (x, y, width, height) of the person
            :return: True if person is in the zone, False otherwise
        """
        if not self.zone_coords:
            return True

        x, y, w, h = person
        person_center_x = x + w // 2
        person_center_y = y + h // 2

        zone_x1, zone_y1 = self.zone_coords[0]
        zone_x2, zone_y2 = self.zone_coords[1]

        return (
            zone_x1 <= person_center_x <= zone_x2
            and zone_y1 <= person_center_y <= zone_y2
        )

    def cleanup(self):
        """
        Cleans up resources
            :return: None
        """
        self.monitoring = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        _log_with_timestamp("🧹 System cleanup finished")
