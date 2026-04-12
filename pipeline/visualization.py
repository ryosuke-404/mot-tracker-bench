"""
Debug visualization for the tracking pipeline.

Draws bounding boxes, track IDs, Re-ID similarity scores, tripwire lines,
and OD event annotations on frames for development and tuning.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracking.track import STrack, TrackState
from tripwire.tripwire_manager import TripwireLine, CrossingEvent, CrossingType


# Color palette (BGR)
_COLORS = {
    TrackState.Tentative: (200, 200, 0),    # cyan-ish
    TrackState.Confirmed: (0, 255, 0),       # green
    TrackState.Lost:      (0, 100, 255),     # orange
}
_BOARD_COLOR = (0, 255, 128)   # green-yellow
_ALIGHT_COLOR = (0, 64, 255)   # red-orange
_LINE_COLOR = (255, 255, 0)    # yellow


def draw_tracks(
    frame: np.ndarray,
    tracks: list[STrack],
    show_reid: bool = False,
) -> np.ndarray:
    """Draw bounding boxes and track labels."""
    out = frame.copy()

    for track in tracks:
        color = _COLORS.get(track.state, (128, 128, 128))
        x1, y1, x2, y2 = track.tlbr.astype(int)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track.track_id}"
        if show_reid and track.reid_embedding is not None:
            label += f" Re-ID"
        cv2.putText(
            out, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    return out


def draw_tripwire_lines(
    frame: np.ndarray,
    lines: list[TripwireLine],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Draw virtual tripwire lines (normalized → pixel coords)."""
    out = frame

    for line in lines:
        p1_px = (int(line.p1[0] * frame_width), int(line.p1[1] * frame_height))
        p2_px = (int(line.p2[0] * frame_width), int(line.p2[1] * frame_height))
        cv2.line(out, p1_px, p2_px, _LINE_COLOR, 2)
        cv2.putText(
            out, line.line_id, p1_px,
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _LINE_COLOR, 1, cv2.LINE_AA
        )

    return out


def draw_crossing_events(
    frame: np.ndarray,
    events: list[CrossingEvent],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Flash BOARD / ALIGHT label at the crossing position."""
    out = frame

    for event in events:
        cx = int(event.position[0] * frame_width)
        cy = int(event.position[1] * frame_height)

        if event.crossing_type == CrossingType.BOARD:
            label = f"BOARD {event.track_id}"
            color = _BOARD_COLOR
        else:
            label = f"ALIGHT {event.track_id}"
            color = _ALIGHT_COLOR

        cv2.circle(out, (cx, cy), 12, color, -1)
        cv2.putText(
            out, label, (cx + 14, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    return out


def draw_stats(
    frame: np.ndarray,
    frame_id: int,
    active_count: int,
    fps: float,
    inference_ms: float,
) -> np.ndarray:
    """Overlay performance stats in the top-left corner."""
    out = frame
    lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Det: {inference_ms:.1f}ms",
        f"On bus: {active_count}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (8, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
    return out
