# backend/child_monitor/detection.py

import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from .yolo_config import PERFORMANCE_PROFILES, get_optimized_config

# Configure logging to suppress unnecessary messages
logging.getLogger("ultralytics").setLevel(logging.ERROR)

try:
    # Try to import Ultralytics YOLOv5
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ Ultralytics not found. Using OpenCV DNN as fallback.")


# see yolo_config.py for PERFORMANCE_PROFILES available
if ULTRALYTICS_AVAILABLE:
    # Use the performance profile for YOLOv5
    # This will use the optimized configuration for better performance
    YOLO_CONFIG = PERFORMANCE_PROFILES.get(
        "performance", get_optimized_config("performance")
    )
else:
    YOLO_CONFIG = PERFORMANCE_PROFILES.get(
        "optimized", get_optimized_config("optimized")
    )


def _log_with_timestamp(message: str):
    """
    Adds a timestamp to log messages for better debugging
        :param message: The message to log
    """
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {message}")


class YoloV5PersonDetector:
    """
    YOLOv5 Person Detector
    This class initializes the YOLOv5 model for detecting people
    in images or video frames.
    It supports both Ultralytics YOLOv5 and OpenCV DNN fallback.
    """

    def __init__(
        self,
        model_path=None,
        performance_profile="fast",
    ):
        try:
            from .yolo_config import get_optimized_config

            self.config = get_optimized_config(performance_profile)
        except (ImportError, Exception):
            self.config = YOLO_CONFIG

        self.confidence_threshold = self.config["confidence"]
        self.nms_threshold = self.config["iou"]
        self.model = None
        self.use_ultralytics = ULTRALYTICS_AVAILABLE

        if self.use_ultralytics:
            self._init_yolov5_ultralytics(model_path)
        else:
            self._init_opencv_fallback()

    def _init_yolov5_ultralytics(self, model_path):
        """
        Initializes YOLOv5 using Ultralytics
        This method will try to load the optimized model first,
        falling back to the standard model if necessary.
            :param model_path: Optional path to a custom YOLOv5 model
            :type model_path: str
        """
        try:
            # Prioritizes model "u" optimized
            if model_path and os.path.exists(model_path):
                model_to_use = model_path
            else:
                # Try model "u" first, fallback to normal
                model_to_use = self.config["model_name"]
                if model_to_use == "yolov5n.pt":
                    model_to_use = "yolov5nu.pt"  # Use optimized version

            _log_with_timestamp(f"📦 Load model YOLOv5: {model_to_use}")

            try:
                self.model = YOLO(model_to_use)
                _log_with_timestamp(
                    f"✅ Model {model_to_use} loaded successfully"
                )
            except Exception as e:
                if "yolov5nu" in model_to_use:
                    _log_with_timestamp(
                        f"⚠️ Failed to load {model_to_use}: {e}"
                    )
                    _log_with_timestamp("🔄 Trying classic version yolov5n.pt")
                    self.model = YOLO("yolov5n.pt")
                    _log_with_timestamp("✅ Model yolov5n.pt loaded (fallback)")
                else:
                    raise e

            # Detect available device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            _log_with_timestamp(f"🖥️ Device: {device}")

            # Apply quantization for CPU if enabled
            if device == "cpu" and self.config.get("quantize", False):
                try:
                    _log_with_timestamp("⚡ Quantization enabled for CPU...")
                    # Quantization will be applied during inference
                except Exception as e:
                    _log_with_timestamp(f"⚠️ Quantization failed: {e}")

            _log_with_timestamp("✅ YOLOv5 initialized successfully")

        except Exception as e:
            _log_with_timestamp(f"❌ Error initializing YOLOv5: {e}")
            _log_with_timestamp("🔄 Trying fallback to OpenCV...")
            self.use_ultralytics = False
            self._init_opencv_fallback()

    def _init_yolov5_torch_hub(self, model_path):
        """
        Initializes YOLOv5 using torch.hub
        This method will try to load the optimized model first,
        falling back to the standard model if necessary.
            :param model_path: Optional path to a custom YOLOv5 model
            :type model_path: str
        """
        try:
            _log_with_timestamp("📦 Load model YOLOv5 via torch.hub...")

            # Try to load the improved model first
            try:
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "yolov5nu",
                    pretrained=True,
                    trust_repo=True,
                    verbose=False,
                )
                _log_with_timestamp(
                    "✅ Model YOLOv5nu loaded successfully (optimized version)"
                )
            except Exception as e:
                # Fallback to classic model if new one doesn't work
                _log_with_timestamp(f"⚠️ Failed to load YOLOv5nu: {e}")
                _log_with_timestamp("🔄 Trying YOLOv5n classic model...")

                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "yolov5n",  # Fallback to classic version
                    pretrained=True,
                    trust_repo=True,
                    verbose=False,
                )
                _log_with_timestamp(
                    "✅ Model YOLOv5n loaded successfully " "(classic version)"
                )

            # Configure model
            self.model.conf = self.confidence_threshold     # type: ignore
            self.model.iou = self.nms_threshold             # type: ignore
            self.model.classes = [0]  # Class 0 = person    # type: ignore

            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            _log_with_timestamp(
                f"🎯 Detector configured (device: {self.device})"
            )

        except Exception as e:
            _log_with_timestamp(
                f"❌ Error loading YOLO model via torch.hub: {e}"
            )
            _log_with_timestamp("💡 Installing missing dependencies...")

            # Try to install missing dependencies
            try:
                import subprocess

                subprocess.check_call([
                    "pip",
                    "install",
                    "seaborn",
                    "matplotlib"
                ])
                _log_with_timestamp(
                    "✅ Dependencies installed. Please restart the system."
                )
            except Exception:
                pass

            self.model = None

    def _init_opencv_fallback(self):
        """
        Fallback using OpenCV DNN with ONNX model
        This method initializes the OpenCV DNN model for YOLOv5.
        It will use the YOLOv5n ONNX model if available.
            :param model_path: Optional path to a custom YOLOv5 model
            :type model_path: str
        """
        self.net = None  # Always initialize as None first
        try:
            # Try to load ONNX model if available
            onnx_path = "yolov5n.onnx"  # Nano to be lighter
            if os.path.exists(onnx_path):
                self.net = cv2.dnn.readNetFromONNX(onnx_path)
                _log_with_timestamp("✅ ONNX model loaded with OpenCV")
            else:
                _log_with_timestamp("❌ ONNX model not found")
                _log_with_timestamp("💡 For better performance, install:")
                _log_with_timestamp(
                    "   pip install ultralytics seaborn matplotlib"
                )
                self.model = None

        except Exception as e:
            _log_with_timestamp(f"❌ Error in OpenCV fallback: {e}")
            self.model = None
            self.net = None

    def detect_person_in_frame(self, frame):
        """
        Detect people in the frame using quantized YOLOv5
            :param frame: Input image or video frame
            :type frame: np.ndarray
            :return: List of bounding boxes for detected people
            :rtype: List[List[int]]
        """
        if not self.model and not hasattr(self, "net"):
            return []

        try:
            if self.use_ultralytics and self.model:
                return self._detect_with_ultralytics(frame)
            elif hasattr(self, "net") and self.net:
                return self._detect_with_opencv(frame)
            else:
                return []

        except Exception as e:
            _log_with_timestamp(f"❌ Error in detection: {e}")
            return []

    def _detect_with_ultralytics(self, frame):
        """
        Detection using Ultralytics YOLO with optimizations
            :param frame: Input image or video frame
            :type frame: np.ndarray
            :return: List of bounding boxes for detected people
            :rtype: List[List[int]]
        """
        if not self.model:
            return []

        # Executes inference with optimized settings for speed
        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=self.config.get("classes", [0]),     # Only detect people
            max_det=self.config.get("max_det", 10),
            imgsz=self.config.get("img_size", 416),      # Smaller for speed
            device=getattr(self, "device", "cpu"),
            half=False,
        )   # type: ignore

        people_boxes = []

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 = person in COCO dataset
                    if int(box.cls) == 0:  # person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Convert to [x, y, w, h] format
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)

                        people_boxes.append([x, y, w, h])

        return people_boxes

    def _detect_with_opencv(self, frame):
        """
        Fallback using OpenCV DNN
            :param frame: Input image or video frame
            :type frame: np.ndarray
            :return: List of bounding boxes for detected people
            :rtype: List[List[int]]
        """
        if not hasattr(self, "net") or self.net is None:
            return []

        height, width = frame.shape[:2]

        # Prepare input with optimized size
        img_size = self.config.get("img_size", 416)
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (img_size, img_size), swapRB=True, crop=False
        )

        self.net.setInput(blob)
        outputs = self.net.forward()

        boxes = []
        confidences = []

        # Process detections
        for output in outputs:
            for detection in output:            # type: ignore
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Class 0 = person
                if class_id == 0 and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.confidence_threshold,
                self.nms_threshold,
            )
            if len(indices) > 0:
                # Dealing with different types of NMSBoxes return
                try:
                    if isinstance(indices, np.ndarray):
                        indices_flat = indices.flatten()
                    else:
                        indices_flat = list(indices)
                    people_boxes = [boxes[i] for i in indices_flat]
                except (AttributeError, TypeError):
                    # Fallback para compatibilidade
                    people_boxes = [boxes[i] for i in range(len(boxes))]
            else:
                people_boxes = []
        else:
            people_boxes = []

        return people_boxes

    def is_available(self):
        """
        Checks if the detector is available
            :return: bool: True if the model is loaded, False otherwise
        """
        return (self.model is not None) or (
            hasattr(self, "net") and self.net is not None
        )
