"""
Stop manager: resolves GPS coordinates to named bus stops.

Loads a JSON stop list and returns the nearest stop within a proximity radius.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

GpsCoord = tuple[float, float]   # (latitude, longitude)


@dataclass
class BusStop:
    stop_id: str
    name: str
    lat: float
    lon: float


class StopManager:
    """
    Maps GPS coordinates to bus stop IDs.

    stops.json format:
    [
      {"stop_id": "S01", "name": "市役所前", "lat": 35.0, "lon": 135.0},
      ...
    ]
    """

    def __init__(
        self,
        stop_list_path: str,
        proximity_radius_m: float = 30.0,
    ) -> None:
        self.stop_list_path = stop_list_path
        self.proximity_radius_m = proximity_radius_m
        self._stops: list[BusStop] = []

    # ------------------------------------------------------------------
    def load_stops(self) -> None:
        with open(self.stop_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._stops = [
            BusStop(
                stop_id=s["stop_id"],
                name=s["name"],
                lat=float(s["lat"]),
                lon=float(s["lon"]),
            )
            for s in data
        ]
        logger.info("Loaded %d bus stops from %s", len(self._stops), self.stop_list_path)

    # ------------------------------------------------------------------
    def get_current_stop(self, gps_coord: GpsCoord) -> Optional[str]:
        """
        Return stop_id of the nearest stop within proximity_radius_m,
        or None if no stop is within range.
        """
        if not self._stops:
            return None

        lat, lon = gps_coord
        nearest_id: Optional[str] = None
        nearest_dist = float("inf")

        for stop in self._stops:
            dist = self._haversine_m(lat, lon, stop.lat, stop.lon)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = stop.stop_id

        if nearest_dist <= self.proximity_radius_m:
            return nearest_id
        return None

    def is_at_stop(self, gps_coord: GpsCoord) -> bool:
        return self.get_current_stop(gps_coord) is not None

    def get_stop_by_id(self, stop_id: str) -> Optional[BusStop]:
        for stop in self._stops:
            if stop.stop_id == stop_id:
                return stop
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine great-circle distance in metres."""
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
