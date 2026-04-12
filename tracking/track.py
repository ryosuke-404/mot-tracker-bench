"""
STrack: per-object track state used by ByteTrack.

Lifecycle:
    NEW (1 hit) → Tentative (< min_hits) → Confirmed
    Confirmed  → Lost (missed frames ≤ track_buffer)
    Lost       → Removed (missed frames > track_buffer)
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum

import numpy as np

from tracking.kalman_filter import KalmanFilter

_shared_kalman = KalmanFilter()


class TrackState(IntEnum):
    Tentative = 1
    Confirmed = 2
    Lost = 3
    Removed = 4


class STrack:
    """
    A single tracked object.

    The Kalman state is stored in the [cx, cy, a, h, vcx, vcy, va, vh] format.
    External callers work with [x1, y1, x2, y2] bounding boxes and the
    conversion helpers `tlbr_to_xyah` / `xyah_to_tlbr` handle translation.
    """

    _count = 0

    # ------------------------------------------------------------------
    @staticmethod
    def new_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id_counter() -> None:
        STrack._count = 0

    # ------------------------------------------------------------------
    def __init__(self, tlbr: np.ndarray, score: float) -> None:
        """
        Args:
            tlbr   bounding box [x1, y1, x2, y2] (pixel coords)
            score  detection confidence
        """
        self.track_id = STrack.new_id()
        self.state = TrackState.Tentative

        self._tlbr = tlbr.copy()
        self.score = score

        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self._is_activated = False

        self.hits = 0              # consecutive matched frames
        self.age = 0               # total frames since creation
        self.time_since_update = 0  # frames since last match

        # Re-ID embedding cache
        self.reid_embedding: np.ndarray | None = None
        self.embedding_history: deque[np.ndarray] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Kalman helpers
    # ------------------------------------------------------------------

    def activate(self, frame_id: int) -> None:
        """Initialize Kalman filter on first assignment."""
        self.mean, self.covariance = _shared_kalman.initiate(
            self.tlbr_to_xyah(self._tlbr)
        )
        self._is_activated = True
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def predict(self) -> None:
        """Advance Kalman prediction by one frame."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Confirmed:
            mean_state[7] = 0  # zero height velocity for uncertain tracks
        self.mean, self.covariance = _shared_kalman.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(tracks: list[STrack]) -> None:
        """Batch predict for all active tracks (in-place)."""
        if not tracks:
            return
        multi_mean = np.array([t.mean.copy() for t in tracks])
        multi_cov = np.array([t.covariance for t in tracks])

        for i, st in enumerate(tracks):
            if st.state != TrackState.Confirmed:
                multi_mean[i][7] = 0

        # Vectorized prediction
        for i, st in enumerate(tracks):
            st.mean, st.covariance = _shared_kalman.predict(
                multi_mean[i], multi_cov[i]
            )

    # ------------------------------------------------------------------
    def update(
        self,
        new_tlbr: np.ndarray,
        score: float,
        embedding: np.ndarray | None = None,
    ) -> None:
        """
        Update track with a new matched detection.
        """
        self._tlbr = new_tlbr.copy()
        self.score = score
        self.mean, self.covariance = _shared_kalman.update(
            self.mean, self.covariance, self.tlbr_to_xyah(new_tlbr)
        )

        if embedding is not None:
            self.reid_embedding = embedding
            self.embedding_history.append(embedding)

        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed
        elif self.state == TrackState.Lost:
            self.state = TrackState.Confirmed

    def increment_age(self) -> None:
        """Called each frame the track is NOT matched."""
        self.age += 1
        self.time_since_update += 1

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    def reassign_id(self, existing_id: int) -> None:
        """Override track_id when Re-ID resurrects a previously lost track."""
        self.track_id = existing_id

    # ------------------------------------------------------------------
    # Bounding box accessors
    # ------------------------------------------------------------------

    @property
    def tlbr(self) -> np.ndarray:
        """Return bounding box as [x1, y1, x2, y2]."""
        if self.mean is None:
            return self._tlbr
        return self.xyah_to_tlbr(self.mean[:4])

    @property
    def centroid(self) -> tuple[float, float]:
        """Return (cx, cy) center of the bounding box."""
        b = self.tlbr
        return float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)

    # ------------------------------------------------------------------
    # Coordinate conversion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] → [cx, cy, a, h]"""
        x1, y1, x2, y2 = tlbr
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1
        a = (x2 - x1) / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=float)

    @staticmethod
    def xyah_to_tlbr(xyah: np.ndarray) -> np.ndarray:
        """[cx, cy, a, h] → [x1, y1, x2, y2]"""
        cx, cy, a, h = xyah
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def __repr__(self) -> str:
        return (
            f"STrack(id={self.track_id}, state={self.state.name}, "
            f"hits={self.hits}, tsu={self.time_since_update})"
        )
