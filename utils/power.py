# utils/power.py
from __future__ import annotations

def get_power_state() -> str:
    """
    Returns: "plugged" | "battery" | "unknown"
    """
    try:
        import psutil  # type: ignore
        batt = psutil.sensors_battery()
        if batt is None:
            return "unknown"
        return "plugged" if batt.power_plugged else "battery"
    except Exception:
        return "unknown"
