"""Pinch's firmware, assembled from its parts: core.py (the loop, stops, saving, status) and senses.py
(the sensor threads and how results are applied)."""

from robot.firmware.core import FirmwareCore
from robot.firmware.senses import FirmwareSenses


class PinchFirmware(FirmwareSenses, FirmwareCore):
    pass
