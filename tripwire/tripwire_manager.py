"""
Virtual tripwire manager.

Each bus door has two virtual lines (outer and inner).
A boarding event is recorded when a passenger's centroid crosses:
    outer_line (inward)  →  inner_line (inward)   within `sequence_timeout_frames`

An alighting event is the reverse:
    inner_line (outward) →  outer_line (outward)

Crossing detection uses the signed cross product of the line vector
with the centroid motion vector.  The direction of crossing is confirmed
by a dot-product check against the door's inward normal vector.

All coordinates are in NORMALIZED space [0, 1] to be resolution-agnostic.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

Point = tuple[float, float]   # (x_norm, y_norm)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CrossingType(Enum):
    BOARD = auto()
    ALIGHT = auto()


@dataclass
class CrossingEvent:
    track_id: int
    door_id: str
    camera_id: str
    crossing_type: CrossingType
    frame_id: int
    position: Point   # centroid at crossing moment


@dataclass
class TripwireLine:
    line_id: str
    door_id: str
    p1: Point
    p2: Point
    inward_normal: Point    # unit vector pointing INTO the bus interior


@dataclass
class DoorConfig:
    door_id: str
    outer: TripwireLine
    inner: TripwireLine


# ---------------------------------------------------------------------------
# Per-track crossing state machine
# ---------------------------------------------------------------------------

class SequenceState(Enum):
    IDLE = auto()
    OUTER_CROSSED_INWARD = auto()   # waiting for inner inward  → BOARD
    INNER_CROSSED_OUTWARD = auto()  # waiting for outer outward → ALIGHT


@dataclass
class TrackDoorState:
    sequence: SequenceState = SequenceState.IDLE
    sequence_start_frame: int = 0


# ---------------------------------------------------------------------------
# TripwireManager
# ---------------------------------------------------------------------------

class TripwireManager:
    """
    Manages virtual tripwire lines for all doors across all cameras.

    Usage:
        manager = TripwireManager(config_path, sequence_timeout_frames=75,
                                  direction_dot_thresh=0.3)
        manager.load_config()

        events = manager.update(
            camera_id="front",
            frame_id=100,
            track_positions={1: (0.4, 0.5), 2: (0.6, 0.7)},  # track_id → centroid_norm
            prev_positions={1: (0.38, 0.5), 2: (0.58, 0.7)},
        )
    """

    def __init__(
        self,
        config_path: str,
        sequence_timeout_frames: int = 75,
        direction_dot_thresh: float = 0.3,
    ) -> None:
        self.config_path = config_path
        self.sequence_timeout_frames = sequence_timeout_frames
        self.direction_dot_thresh = direction_dot_thresh

        # camera_id → list[DoorConfig]
        self._doors: dict[str, list[DoorConfig]] = {}
        # (camera_id, door_id, track_id) → TrackDoorState
        self._track_states: dict[tuple[str, str, int], TrackDoorState] = {}

    # ------------------------------------------------------------------
    def load_config(self) -> None:
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cameras: dict = cfg.get("cameras", {})
        for cam_id, cam_data in cameras.items():
            door_configs: list[DoorConfig] = []
            for door in cam_data.get("doors", []):
                outer_data = door["lines"]["outer"]
                inner_data = door["lines"]["inner"]

                outer = TripwireLine(
                    line_id=outer_data["line_id"],
                    door_id=door["door_id"],
                    p1=tuple(outer_data["p1"]),
                    p2=tuple(outer_data["p2"]),
                    inward_normal=tuple(outer_data["inward_normal"]),
                )
                inner = TripwireLine(
                    line_id=inner_data["line_id"],
                    door_id=door["door_id"],
                    p1=tuple(inner_data["p1"]),
                    p2=tuple(inner_data["p2"]),
                    inward_normal=tuple(inner_data["inward_normal"]),
                )
                door_configs.append(
                    DoorConfig(door_id=door["door_id"], outer=outer, inner=inner)
                )
            self._doors[cam_id] = door_configs
            logger.info("Loaded %d door(s) for camera '%s'", len(door_configs), cam_id)

    # ------------------------------------------------------------------
    def update(
        self,
        camera_id: str,
        frame_id: int,
        track_positions: dict[int, Point],
        prev_positions: dict[int, Point],
    ) -> list[CrossingEvent]:
        """
        Check all tracks for line crossings and return events.

        Args:
            camera_id       which camera these tracks come from
            frame_id        current frame index
            track_positions track_id → current centroid (normalized)
            prev_positions  track_id → previous centroid (normalized)
        """
        events: list[CrossingEvent] = []
        doors = self._doors.get(camera_id, [])

        for door in doors:
            for track_id, curr_pos in track_positions.items():
                prev_pos = prev_positions.get(track_id)
                if prev_pos is None:
                    continue  # no previous position, skip

                state_key = (camera_id, door.door_id, track_id)
                state = self._track_states.setdefault(state_key, TrackDoorState())

                # Timeout: reset incomplete sequences
                if (
                    state.sequence != SequenceState.IDLE
                    and (frame_id - state.sequence_start_frame) > self.sequence_timeout_frames
                ):
                    logger.debug(
                        "Tripwire sequence timeout: cam=%s door=%s track=%d",
                        camera_id, door.door_id, track_id,
                    )
                    state.sequence = SequenceState.IDLE

                # Check outer line crossing
                outer_dir = self._crossing_direction(door.outer, prev_pos, curr_pos)
                # Check inner line crossing
                inner_dir = self._crossing_direction(door.inner, prev_pos, curr_pos)

                event = self._advance_state_machine(
                    state, track_id, door, camera_id, frame_id,
                    curr_pos, outer_dir, inner_dir
                )
                if event:
                    events.append(event)

        self._cleanup_old_states(frame_id)
        return events

    # ------------------------------------------------------------------
    def remove_track(self, track_id: int) -> None:
        """Remove all state machine entries for a track_id."""
        to_remove = [k for k in self._track_states if k[2] == track_id]
        for k in to_remove:
            del self._track_states[k]

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _advance_state_machine(
        self,
        state: TrackDoorState,
        track_id: int,
        door: DoorConfig,
        camera_id: str,
        frame_id: int,
        curr_pos: Point,
        outer_dir: Optional[str],   # "inward" | "outward" | None
        inner_dir: Optional[str],
    ) -> Optional[CrossingEvent]:
        """
        Advance the two-line sequence state machine and emit an event
        when a complete boarding or alighting sequence is detected.
        """
        if state.sequence == SequenceState.IDLE:
            if outer_dir == "inward":
                state.sequence = SequenceState.OUTER_CROSSED_INWARD
                state.sequence_start_frame = frame_id
            elif inner_dir == "outward":
                state.sequence = SequenceState.INNER_CROSSED_OUTWARD
                state.sequence_start_frame = frame_id

        elif state.sequence == SequenceState.OUTER_CROSSED_INWARD:
            if inner_dir == "inward":
                # BOARDING complete
                state.sequence = SequenceState.IDLE
                return CrossingEvent(
                    track_id=track_id,
                    door_id=door.door_id,
                    camera_id=camera_id,
                    crossing_type=CrossingType.BOARD,
                    frame_id=frame_id,
                    position=curr_pos,
                )
            elif outer_dir == "outward":
                # Reversed out — abort
                state.sequence = SequenceState.IDLE

        elif state.sequence == SequenceState.INNER_CROSSED_OUTWARD:
            if outer_dir == "outward":
                # ALIGHTING complete
                state.sequence = SequenceState.IDLE
                return CrossingEvent(
                    track_id=track_id,
                    door_id=door.door_id,
                    camera_id=camera_id,
                    crossing_type=CrossingType.ALIGHT,
                    frame_id=frame_id,
                    position=curr_pos,
                )
            elif inner_dir == "inward":
                # Reversed back in — abort
                state.sequence = SequenceState.IDLE

        return None

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    def _crossing_direction(
        self,
        line: TripwireLine,
        prev_pos: Point,
        curr_pos: Point,
    ) -> Optional[str]:
        """
        Determine whether the centroid crossed `line` between prev and curr,
        and if so, whether the direction is "inward" or "outward".

        Returns "inward", "outward", or None (no crossing / rejected by
        direction filter).
        """
        sign_prev = self._signed_cross(line.p1, line.p2, prev_pos)
        sign_curr = self._signed_cross(line.p1, line.p2, curr_pos)

        # No crossing if both points are on the same side
        if sign_prev * sign_curr >= 0:
            return None

        # Motion vector
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        motion_len = math.hypot(dx, dy)
        if motion_len < 1e-9:
            return None

        # Dot product of motion vector with inward normal
        nx, ny = line.inward_normal
        dot = (dx / motion_len) * nx + (dy / motion_len) * ny

        if dot < -self.direction_dot_thresh:
            return "outward"
        elif dot > self.direction_dot_thresh:
            return "inward"
        else:
            # Motion is mostly parallel to line — ambiguous, ignore
            return None

    @staticmethod
    def _signed_cross(p1: Point, p2: Point, point: Point) -> float:
        """
        Signed cross product of (p2 - p1) × (point - p1).
        Positive = point is on the left of the directed line p1→p2.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return dx * (point[1] - p1[1]) - dy * (point[0] - p1[0])

    def _cleanup_old_states(self, current_frame: int, max_age: int = 300) -> None:
        """Periodically remove state entries for tracks not seen recently."""
        if current_frame % 150 != 0:
            return
        to_remove = [
            k for k, v in self._track_states.items()
            if v.sequence == SequenceState.IDLE
            and (current_frame - v.sequence_start_frame) > max_age
        ]
        for k in to_remove:
            del self._track_states[k]
