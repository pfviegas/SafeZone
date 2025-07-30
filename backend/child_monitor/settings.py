# backend/child_monitor/settings.py

# Global settings for SafeZone application

import os

RESOLUTION = (1024, 768)     # webcam resolution
ALERT_DELAY = 2            # seconds before alert
ALARM_WAV_PATH = os.path.join(
    os.path.dirname(__file__),
    'retro-game-emergency-alarm.wav'
)
