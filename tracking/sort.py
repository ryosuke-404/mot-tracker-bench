"""
SORT: Simple Online and Realtime Tracking
Bewley et al., 2016 — MIT License

Pure IoU-based Kalman tracking. No appearance features.
Fastest and lightest tracker — good baseline.

Key characteristics:
  - Single-stage matching (all dets together, no high/low split)
  - IoU-only cost matrix
  - Track deleted after max_age misses
  - Track confirmed after min_hits consecutive matches
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, linear_assignment


class SORTTracker:
    """
    Original SORT tracker with Kalman filter + Hungarian IoU matching.

    Drop-in replacement for ByteTracker (same update() signature).
    low_dets and frame arguments are accepted but unused.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        max_age: int = 30,
        min_hits: int = 3,
        iou_thresh: float = 0.30,
    ) -> None:
        self.track_thresh = track_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh

        self.tracked_stracks: list[STrack] = []
        self.frame_id = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    # ------------------------------------------------------------------
    def update(
        self,
        high_dets: list[Detection],
        low_dets: list[Detection] | None = None,
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        """
        SORT update: single-stage IoU matching on all detections.

        low_dets and frame are accepted for API compatibility but ignored.
        """
        self.frame_id += 1

        # Merge all dets; filter by track_thresh
        all_dets = [d for d in (high_dets + (low_dets or []))
                    if d.score >= self.track_thresh]

        # Predict Kalman state for all tracks
        STrack.multi_predict(self.tracked_stracks)

        # Build cost matrix (IoU distance)
        if self.tracked_stracks and all_dets:
            track_boxes = np.array([t.tlbr for t in self.tracked_stracks])
            det_boxes   = np.array([d.bbox for d in all_dets])
            cost = 1.0 - iou_matrix(track_boxes, det_boxes)
            matches, unmatched_t_idx, unmatched_d_idx = linear_assignment(
                cost, 1.0 - self.iou_thresh
            )
        else:
            matches        = []
            unmatched_t_idx = list(range(len(self.tracked_stracks)))
            unmatched_d_idx = list(range(len(all_dets)))

        matched_t_idx = {t for t, _ in matches}

        # Update matched tracks
        for t_idx, d_idx in matches:
            t = self.tracked_stracks[t_idx]
            t.update(all_dets[d_idx].bbox, all_dets[d_idx].score,
                     all_dets[d_idx].embedding)

        # Age unmatched tracks; remove stale ones
        still_alive: list[STrack] = []
        for i, track in enumerate(self.tracked_stracks):
            if i in matched_t_idx:
                still_alive.append(track)
            else:
                track.increment_age()
                if track.time_since_update <= self.max_age:
                    still_alive.append(track)

        # Spawn new tentative tracks for unmatched detections
        for d_idx in unmatched_d_idx:
            det = all_dets[d_idx]
            new_track = STrack(det.bbox, det.score)
            new_track.activate(self.frame_id)
            if det.embedding is not None:
                new_track.reid_embedding = det.embedding
                new_track.embedding_history.append(det.embedding)
            still_alive.append(new_track)

        self.tracked_stracks = still_alive

        # Confirm tracks with enough consecutive hits
        for t in self.tracked_stracks:
            if t.state == TrackState.Tentative and t.hits >= self.min_hits:
                t.state = TrackState.Confirmed

        return [t for t in self.tracked_stracks
                if t.state in (TrackState.Confirmed, TrackState.Tentative)]

    # ------------------------------------------------------------------
    def get_confirmed_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]

    def get_lost_tracks(self) -> list[STrack]:
        # SORT has no explicit Lost state; treat time_since_update>0 as "lost"
        return [t for t in self.tracked_stracks if t.time_since_update > 0]
