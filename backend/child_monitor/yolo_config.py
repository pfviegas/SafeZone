# backend/child_monitor/yolo_config.py

import os

# Performance-optimized YOLOv5 configuration
YOLO_CONFIG = {
    # Model to use (yolov5n is the lightest and fastest)
    "model_name": "yolov5n.pt",
    # Input size (smaller = faster, larger = more accurate)
    "img_size": 416,  # Reduced from 640 for better performance
    # Confidence threshold for detections
    "confidence": 0.6,  # Increased to reduce false positives
    # IoU threshold for Non-Maximum Suppression
    "iou": 0.45,
    # Device (auto detects GPU/CPU)
    "device": "cpu",  # Will be changed automatically if GPU is available
    # Quantization settings for CPU
    "quantize": True,
    # Classes to detect (0 = person in COCO dataset)
    "classes": [0],  # Only people
    # Use half precision (FP16) if GPU is available
    "half": False,  # Will be activated automatically for GPU
    # Maximum number of detections per image
    "max_det": 10,
    # Cache settings
    "cache_dir": os.path.join(os.path.dirname(__file__), ".yolo_cache"),
}

# Specific configurations for different scenarios
PERFORMANCE_PROFILES = {
    "fast": {
        "model_name": "yolov5n.pt",     # Nano - lightweight and fast
        "img_size": 320,
        "confidence": 0.6,
    },
    "optimized": {                      # Configuration for performance
        "model_name": "yolov5n.pt",
        "img_size": 416,
        "confidence": 0.6,
    },
    "performance": {
        "model_name": "yolov5nu.pt",    # Ultralytics optimized
        "img_size": 416,                # Smaller for faster inference
        "confidence": 0.6,
    },
    "balanced": {
        "model_name": "yolov5s.pt",     # Small - balanced
        "img_size": 640,
        "confidence": 0.5,
    },
    "accurate": {
        "model_name": "yolov5m.pt",     # Medium - more accurate
        "img_size": 640,
        "confidence": 0.4,
    },
}


def get_optimized_config(profile="balanced"):
    """
    Returns optimized configuration based on the chosen profile
        :param profile: "fast", "balanced", or "accurate"
        :return: dict: Optimized configuration
    """
    config = YOLO_CONFIG.copy()

    if profile in PERFORMANCE_PROFILES:
        config.update(PERFORMANCE_PROFILES[profile])

    return config
