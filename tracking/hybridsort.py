"""
Hybrid-SORT: Weak Cues Matter — Hybrid-SORT for Occlusion-Robust MOT
Yang et al., CVPR 2024 — MIT License

Key idea: supplement IoU with two additional "weak cues":
  1. Height cue (H): ratio of bbox heights between track and detection.
     Persons at the same depth should have similar heights.
  2. Shape cue (S): cosine similarity of (w, h) aspect vector.
     Combined into an HOC (Height-Shape-Orientation Consistency) score.

Full matching pipeline (same 2-stage as ByteTrack):
  Stage 1: high dets vs active tracks
    Cost = α * (1 - IoU) + β * HOC_cost + γ * Re-ID_cost
  Stage 2: low dets vs unmatched + lost tracks
    Cost = IoU only

The HOC score penalizes track-detection pairs where the bounding boxes
differ greatly in height or aspect ratio, which catches cases where the
IoU is non-zero but geometrically inconsistent.
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection, iou_matrix, cosine_distance_matrix, linear_assignment
)


# ---------------------------------------------------------------------------

def height_consistency_matrix(
    tracks: list[STrack], det_boxes: np.ndarray
) -> np.ndarray:
    """
    Height-ratio cost matrix (M, N).
    cost = |h_track - h_det| / max(h_track, h_det)
    0 = identical height, 1 = one is zero.
    """
    M, N = len(tracks), det_boxes.shape[0]
    cost = np.zeros((M, N), dtype=np.float64)
    for i, t in enumerate(tracks):
        b = t.tlbr
        h_t = b[3] - b[1]
        for j in range(N):
            h_d = det_boxes[j, 3] - det_boxes[j, 1]
            denom = max(h_t, h_d, 1e-6)
            cost[i, j] = abs(h_t - h_d) / denom
    return cost


def shape_consistency_matrix(
    tracks: list[STrack], det_boxes: np.ndarray
) -> np.ndarray:
    """
    Shape (aspect-ratio) cosine distance (M, N).
    Shape vector = [w, h]; cosine distance between track and detection shapes.
    """
    M, N = len(tracks), det_boxes.shape[0]
    cost = np.zeros((M, N), dtype=np.float64)
    for i, t in enumerate(tracks):
        b = t.tlbr
        w_t = max(b[2] - b[0], 1e-6)
        h_t = max(b[3] - b[1], 1e-6)
        v_t = np.array([w_t, h_t])
        n_t = np.linalg.norm(v_t)
        for j in range(N):
            w_d = max(det_boxes[j, 2] - det_boxes[j, 0], 1e-6)
            h_d = max(det_boxes[j, 3] - det_boxes[j, 1], 1e-6)
            v_d = np.array([w_d, h_d])
            n_d = np.linalg.norm(v_d)
            sim = np.dot(v_t, v_d) / (n_t * n_d + 1e-12)
            cost[i, j] = 1.0 - float(np.clip(sim, 0.0, 1.0))
    return cost


# ---------------------------------------------------------------------------

class HybridSORTTracker:
    """
    Hybrid-SORT multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.85,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.45,
        iou_weight:    float = 0.50,
        height_weight: float = 0.20,
        shape_weight:  float = 0.10,
        reid_weight:   float = 0.20,
    ) -> None:
        self.track_thresh      = track_thresh
        self.track_buffer      = track_buffer
        self.match_thresh      = match_thresh
        self.min_hits          = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.iou_weight    = iou_weight
        self.height_weight = height_weight
        self.shape_weight  = shape_weight
        self.reid_weight   = reid_weight

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []
        self.removed_stracks: list[STrack] = []
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

        # Stage 1: high dets vs active (HOC + IoU + Re-ID)
        (m1, unm_t1, unm_d1) = self._associate_hybrid(
            high_dets, self.tracked_stracks, self.match_thresh
        )
        for t_idx, d_idx in m1:
            self.tracked_stracks[t_idx].update(
                high_dets[d_idx].bbox, high_dets[d_idx].score, high_dets[d_idx].embedding
            )
        unmatched_active = [self.tracked_stracks[i] for i in unm_t1]

        # Stage 2: low dets vs unmatched + lost (IoU only)
        candidate_s2 = unmatched_active + self.lost_stracks
        (m2, unm_s2, _) = self._associate_iou(low_dets, candidate_s2, self.iou_thresh_stage2)
        matched_s2 = {t for t, _ in m2}
        for t_idx, d_idx in m2:
            candidate_s2[t_idx].update(low_dets[d_idx].bbox, low_dets[d_idx].score, None)
        for i, track in enumerate(candidate_s2):
            if i not in matched_s2:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # New tracks
        for d_idx in unm_d1:
            det = high_dets[d_idx]
            if det.score < self.track_thresh:
                continue
            t = STrack(det.bbox, det.score)
            t.activate(self.frame_id)
            if det.embedding is not None:
                t.reid_embedding = det.embedding
                t.embedding_history.append(det.embedding)
            self.tracked_stracks.append(t)

        # Rebuild
        seen_ids: set[int] = set()
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            elif track.track_id not in seen_ids:
                still_lost.append(track)
                seen_ids.add(track.track_id)
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

    def _associate_hybrid(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])

        iou_d   = 1.0 - iou_matrix(track_boxes, det_boxes)
        height_d = height_consistency_matrix(tracks, det_boxes)
        shape_d  = shape_consistency_matrix(tracks, det_boxes)

        t_embs = [t.reid_embedding for t in tracks]
        d_embs = [d.embedding      for d in detections]
        if all(e is not None for e in t_embs) and all(e is not None for e in d_embs):
            reid_d = cosine_distance_matrix(np.stack(t_embs), np.stack(d_embs))
            cost = (
                self.iou_weight    * iou_d
                + self.height_weight * height_d
                + self.shape_weight  * shape_d
                + self.reid_weight   * reid_d
            )
        else:
            w = self.iou_weight + self.height_weight + self.shape_weight
            cost = (
                (self.iou_weight / w)    * iou_d
                + (self.height_weight / w) * height_d
                + (self.shape_weight  / w) * shape_d
            )
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
