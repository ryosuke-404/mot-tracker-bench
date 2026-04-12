"""
GHOST: Global Hierarchical Online Similarity Tracker
Seidenschwarz et al., 2023 — Apache 2.0 License

Core ideas:
  1. Two-tiered Re-ID gallery:
     - Short-term gallery: embeddings from the last K frames (recent context)
     - Long-term gallery:  all embeddings since track birth (identity memory)

  2. Hierarchical matching pipeline:
     Stage 1: Short-term gallery × high dets (for active/recently-seen tracks)
     Stage 2: Long-term gallery × remaining dets (for long-lost tracks)
     Stage 3: IoU-only × low dets (for fragmented/occluded tracks)

  3. ID recycling prevention:
     A track's ID is "locked" for `id_lock_frames` frames after it goes Lost.
     New detections matching the long-term gallery during this window
     restore the original ID instead of creating a new track.

  4. EMA appearance: both galleries store EMA-blended embeddings to reduce
     frame-to-frame embedding noise.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection, iou_matrix, cosine_distance_matrix, linear_assignment
)

EMA_ALPHA       = 0.85
SHORT_WINDOW    = 10   # frames for short-term gallery
ID_LOCK_FRAMES  = 60   # frames to keep long-term gallery after going Lost


# ---------------------------------------------------------------------------

class GHOSTTrack(STrack):
    """STrack with short-term and long-term EMA embedding galleries."""

    def __init__(self, tlbr: np.ndarray, score: float) -> None:
        super().__init__(tlbr, score)
        # Short-term: rolling window of EMA embeddings
        self.short_gallery: deque[np.ndarray] = deque(maxlen=SHORT_WINDOW)
        # Long-term: single EMA-blended embedding (all history)
        self.long_embed: np.ndarray | None = None
        self.lost_frame: int = -1   # frame when the track went Lost

    def push_embedding(self, embed: np.ndarray) -> None:
        """Update both galleries with a new embedding."""
        if self.long_embed is None:
            self.long_embed = embed.copy()
        else:
            self.long_embed = EMA_ALPHA * self.long_embed + (1 - EMA_ALPHA) * embed
            n = np.linalg.norm(self.long_embed)
            if n > 1e-12:
                self.long_embed /= n
        self.short_gallery.append(embed.copy())
        self.reid_embedding = self.long_embed
        self.embedding_history.append(self.long_embed.copy())

    def short_min_cos_dist(self, query: np.ndarray) -> float:
        if not self.short_gallery:
            return 1.0
        g = np.stack(list(self.short_gallery))
        return float(1.0 - np.max(g @ query))

    def long_cos_dist(self, query: np.ndarray) -> float:
        if self.long_embed is None:
            return 1.0
        return float(1.0 - float(self.long_embed @ query))


# ---------------------------------------------------------------------------

class GHOSTTracker:
    """
    GHOST multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh_short: float = 0.35,   # cosine dist threshold for short-term
        match_thresh_long:  float = 0.50,   # cosine dist threshold for long-term
        match_thresh_iou:   float = 0.45,
        min_hits: int = 3,
        iou_thresh_stage3: float = 0.45,
        proximity_gate: float = 0.10,       # min IoU to allow Re-ID match
    ) -> None:
        self.track_thresh       = track_thresh
        self.track_buffer       = track_buffer
        self.match_thresh_short = match_thresh_short
        self.match_thresh_long  = match_thresh_long
        self.match_thresh_iou   = match_thresh_iou
        self.min_hits           = min_hits
        self.iou_thresh_stage3  = iou_thresh_stage3
        self.proximity_gate     = proximity_gate

        self.tracked_stracks: list[GHOSTTrack] = []
        self.lost_stracks:    list[GHOSTTrack] = []
        self.removed_stracks: list[GHOSTTrack] = []
        self.frame_id = 0

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    def update(
        self,
        high_dets: list[Detection],
        low_dets:  list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        self.frame_id += 1
        STrack.multi_predict(self.tracked_stracks + self.lost_stracks)

        # Stage 1: Short-term gallery × high dets vs active tracks
        (m1, unm_t1, unm_d1) = self._associate_short(
            high_dets, self.tracked_stracks, self.match_thresh_short
        )
        for t_idx, d_idx in m1:
            track = self.tracked_stracks[t_idx]
            det   = high_dets[d_idx]
            track.update(det.bbox, det.score, None)
            if det.embedding is not None:
                track.push_embedding(det.embedding)

        unmatched_active = [self.tracked_stracks[i] for i in unm_t1]
        remaining_high   = [high_dets[i] for i in unm_d1]

        # Stage 2: Long-term gallery × remaining dets vs lost tracks
        (m2, unm_lostT, unm_d2) = self._associate_long(
            remaining_high, self.lost_stracks, self.match_thresh_long
        )
        restored_ids: set[int] = set()
        for t_idx, d_idx in m2:
            track = self.lost_stracks[t_idx]
            det   = remaining_high[d_idx]
            track.update(det.bbox, det.score, None)
            if det.embedding is not None:
                track.push_embedding(det.embedding)
            restored_ids.add(track.track_id)

        restored_tracks   = [self.lost_stracks[t] for t, _ in m2]
        unmatched_lost    = [self.lost_stracks[i] for i in unm_lostT]
        remaining_high2   = [remaining_high[i] for i in unm_d2]

        # Stage 3: IoU × unmatched active + unmatched lost × low dets
        candidate_s3 = unmatched_active + unmatched_lost
        (m3, unm_s3, _) = self._associate_iou(
            low_dets, candidate_s3, self.iou_thresh_stage3
        )
        matched_s3 = {t for t, _ in m3}
        for t_idx, d_idx in m3:
            candidate_s3[t_idx].update(low_dets[d_idx].bbox, low_dets[d_idx].score, None)
        for i, track in enumerate(candidate_s3):
            if i not in matched_s3:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    track.lost_frame = self.frame_id
                else:
                    track.increment_age()

        # New tracks for still-unmatched high dets (remaining_high2)
        for det in remaining_high2:
            if det.score < self.track_thresh:
                continue
            t = GHOSTTrack(det.bbox, det.score)
            t.activate(self.frame_id)
            if det.embedding is not None:
                t.push_embedding(det.embedding)
            self.tracked_stracks.append(t)

        # Move restored tracks back to tracked
        for t in restored_tracks:
            t.state = TrackState.Confirmed
        self.tracked_stracks.extend(restored_tracks)

        # Rebuild lists
        seen: set[int] = set()
        still_lost: list[GHOSTTrack] = []
        for track in unmatched_lost:
            if track.track_id in restored_ids:
                continue
            frames_lost = self.frame_id - track.lost_frame
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            elif track.track_id not in seen:
                still_lost.append(track)
                seen.add(track.track_id)
        self.lost_stracks = still_lost
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state in (TrackState.Confirmed, TrackState.Tentative)
        ]
        return list(self.tracked_stracks)

    def get_confirmed_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]

    def get_lost_tracks(self) -> list[STrack]:
        return list(self.lost_stracks)

    # ------------------------------------------------------------------

    def _associate_short(
        self, detections: list[Detection], tracks: list[GHOSTTrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        iou = iou_matrix(track_boxes, det_boxes)

        M, N = len(tracks), len(detections)
        cost = np.ones((M, N), dtype=np.float64)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if det.embedding is not None and track.short_gallery:
                    cd = track.short_min_cos_dist(det.embedding)
                    if iou[i, j] < self.proximity_gate:
                        cost[i, j] = 1.0
                    else:
                        cost[i, j] = 0.5 * (1 - iou[i, j]) + 0.5 * cd
                else:
                    cost[i, j] = 1.0 - iou[i, j]
        return linear_assignment(cost, thresh)

    def _associate_long(
        self, detections: list[Detection], tracks: list[GHOSTTrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        iou = iou_matrix(track_boxes, det_boxes)

        M, N = len(tracks), len(detections)
        cost = np.ones((M, N), dtype=np.float64)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if det.embedding is not None and track.long_embed is not None:
                    cd = track.long_cos_dist(det.embedding)
                    # Long-term: IoU proximity gate is relaxed (lost tracks move)
                    if iou[i, j] > 0.0:
                        cost[i, j] = 0.3 * (1 - iou[i, j]) + 0.7 * cd
                    else:
                        cost[i, j] = cd + 0.2   # slight penalty for zero IoU
                else:
                    cost[i, j] = 1.0 - iou[i, j]
        return linear_assignment(cost, thresh)

    def _associate_iou(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = 1.0 - iou_matrix(
            np.array([t.tlbr for t in tracks]),
            np.array([d.bbox for d in detections]),
        )
        return linear_assignment(cost, thresh)
