# backend/child_monitor/shared_state.py

import threading
import time
from typing import Optional

# Global shared state for the SafeZone monitoring system
alert_timer_started = False
alert_start_time = 0
running = True

# Strict alarm control
_alarm_lock = threading.Lock()
_alarm_active = False
_alarm_start_time: Optional[float] = None
_last_alarm_log_time: Optional[float] = None


def is_alarm_active() -> bool:
    """
    Checks if the alarm is currently active.
        Returns True if the alarm is active, False otherwise.
    """
    with _alarm_lock:
        return _alarm_active


def start_alarm() -> bool:
    """
    Starts the alarm if it is not already active.
        Returns True if the alarm was started, False if it was already active.
    """
    global _alarm_active, _alarm_start_time

    with _alarm_lock:
        if _alarm_active:
            return False

        _alarm_active = True
        _alarm_start_time = time.time()
        return True


def stop_alarm() -> tuple[bool, float]:
    """
    Stops the alarm if it is active.
        Returns (success: bool, duration: float).
        success=True if the alarm was stopped, False if it was not active.
    """
    global _alarm_active, _alarm_start_time, _last_alarm_log_time

    with _alarm_lock:
        if not _alarm_active or _alarm_start_time is None:
            return False, 0.0
        current_time = time.time()
        duration = current_time - _alarm_start_time

        # Avoid spam - only allow log if enough time has passed since the last
        if (
            _last_alarm_log_time
            and (current_time - _last_alarm_log_time) < 0.5
        ):
            return False, 0.0  # Too recent, ignore

        _alarm_active = False
        _alarm_start_time = None
        _last_alarm_log_time = current_time

        return True, duration


def reset_alarm_state():
    """
    Resets the alarm state to inactive and clears timestamps.
    """
    global _alarm_active, _alarm_start_time, _last_alarm_log_time

    with _alarm_lock:
        _alarm_active = False
        _alarm_start_time = None
        _last_alarm_log_time = None


def get_alarm_duration() -> float:
    """
    Returns the duration in seconds, or 0.0 if the alarm is not active.
    """
    with _alarm_lock:
        if _alarm_active and _alarm_start_time:
            return time.time() - _alarm_start_time
        return 0.0


should_exit = threading.Event()
latest_frame = None
frame_lock = threading.Lock()
