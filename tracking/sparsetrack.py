"""
SparseTrack: Multi-Object Tracking by Performing Scene Decomposition
based on Pseudo-Depth
Liu et al., 2023 — MIT License

Core ideas:
  1. Pseudo-depth ordering: estimate relative depth from bbox area
     (larger area ≈ closer to camera) or from vertical position.
     Decompose the scene into depth layers.

  2. Layer-wise matching: match detections to tracks layer-by-layer,
     starting from the nearest layer. This avoids cross-depth confusions
     where a far-away person steals the ID of a nearby one.

  3. Sub-cluster matching: within each layer, only consider spatially
     nearby track-detection pairs (sparse cost matrix), reducing compute.

  4. Same high/low two-stage structure as ByteTrack.

Depth estimation (without LiDAR):
  depth_score = bbox_area / frame_area   (larger = closer)
  Alternatively: depth_score = 1 - cy / frame_height (higher y = closer)
  We use a combination of both.
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection, iou_matrix, cosine_distance_matrix, linear_assignment
)

# Number of depth layers
N_LAYERS = 3


def pseudo_depth(bbox: np.ndarray, frame_h: float = 1440.0, frame_w: float = 2560.0) -> float:
    """
    Estimate relative depth (0 = far, 1 = near camera).
    Combines normalized bbox area and vertical position.
    """
    area  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_norm = np.clip(area / (frame_h * frame_w), 0.0, 1.0)
    cy_norm   = (bbox[1] + bbox[3]) / 2 / frame_h   # 0 = top (far), 1 = bottom (near)
    return float(0.5 * area_norm + 0.5 * cy_norm)


def assign_layer(depth: float, n: int = N_LAYERS) -> int:
    """Map depth ∈ [0,1] to a layer index 0..(n-1)."""
    return min(int(depth * n), n - 1)


# ---------------------------------------------------------------------------

class SparseTracker:
    """
    SparseTrack multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.85,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.45,
        reid_cost_weight: float = 0.35,
        n_layers: int = N_LAYERS,
        proximity_thresh: float = 0.50,   # min IoU to compute Re-ID cost at all
        frame_h: float = 1440.0,
        frame_w: float = 2560.0,
    ) -> None:
        self.track_thresh      = track_thresh
        self.track_buffer      = track_buffer
        self.match_thresh      = match_thresh
        self.min_hits          = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.reid_cost_weight  = reid_cost_weight
        self.n_layers          = n_layers
        self.proximity_thresh  = proximity_thresh
        self.frame_h           = frame_h
        self.frame_w           = frame_w

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

        if frame is not None:
            h, w = frame.shape[:2]
            self.frame_h, self.frame_w = float(h), float(w)

        STrack.multi_predict(self.tracked_stracks + self.lost_stracks)

        # Stage 1: high dets vs active tracks — layer-wise sparse matching
        (m1, unm_t1, unm_d1) = self._associate_layered(
            high_dets, self.tracked_stracks, self.match_thresh
        )
        for t_idx, d_idx in m1:
            self.tracked_stracks[t_idx].update(
                high_dets[d_idx].bbox, high_dets[d_idx].score, high_dets[d_idx].embedding
            )
        unmatched_active = [self.tracked_stracks[i] for i in unm_t1]

        # Stage 2: low dets vs unmatched + lost (standard IoU)
        candidate_s2 = unmatched_active + self.lost_stracks
        (m2, unm_s2, _) = self._associate_iou(
            low_dets, candidate_s2, self.iou_thresh_stage2
        )
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
        seen: set[int] = set()
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
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

    def _associate_layered(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        """
        Layer-wise matching: group tracks and detections by depth layer,
        match each layer independently (nearest first), then collect global
        matches and run a final global pass for cross-layer stragglers.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Assign depth layers
        t_layers = [assign_layer(
            pseudo_depth(t.tlbr, self.frame_h, self.frame_w), self.n_layers
        ) for t in tracks]
        d_layers = [assign_layer(
            pseudo_depth(d.bbox, self.frame_h, self.frame_w), self.n_layers
        ) for d in detections]

        all_matches: list[tuple[int, int]] = []
        unmatched_t = set(range(len(tracks)))
        unmatched_d = set(range(len(detections)))

        # Per-layer matching (nearest layer = highest index first)
        for layer in range(self.n_layers - 1, -1, -1):
            t_idx_layer = [i for i in unmatched_t if t_layers[i] == layer]
            d_idx_layer = [j for j in unmatched_d if d_layers[j] == layer]
            if not t_idx_layer or not d_idx_layer:
                continue
            sub_matches, sub_unm_t, sub_unm_d = self._associate_fused(
                [detections[j] for j in d_idx_layer],
                [tracks[i]     for i in t_idx_layer],
                thresh,
            )
            for local_t, local_d in sub_matches:
                global_t = t_idx_layer[local_t]
                global_d = d_idx_layer[local_d]
                all_matches.append((global_t, global_d))
                unmatched_t.discard(global_t)
                unmatched_d.discard(global_d)

        # Final cross-layer pass for remaining unmatched
        if unmatched_t and unmatched_d:
            remain_t_idx = list(unmatched_t)
            remain_d_idx = list(unmatched_d)
            sub_matches, _, _ = self._associate_fused(
                [detections[j] for j in remain_d_idx],
                [tracks[i]     for i in remain_t_idx],
                thresh,
            )
            for local_t, local_d in sub_matches:
                global_t = remain_t_idx[local_t]
                global_d = remain_d_idx[local_d]
                all_matches.append((global_t, global_d))
                unmatched_t.discard(global_t)
                unmatched_d.discard(global_d)

        return all_matches, list(unmatched_t), list(unmatched_d)

    def _associate_fused(
        self,
        detections: list[Detection],
        tracks: list[STrack],
        thresh: float,
    ) -> tuple:
        """IoU + Re-ID fused cost with proximity gate."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        iou   = iou_matrix(track_boxes, det_boxes)
        iou_d = 1.0 - iou

        t_embs = [t.reid_embedding for t in tracks]
        d_embs = [d.embedding      for d in detections]
        if all(e is not None for e in t_embs) and all(e is not None for e in d_embs):
            reid_d = cosine_distance_matrix(np.stack(t_embs), np.stack(d_embs))
            fused  = (1 - self.reid_cost_weight) * iou_d + self.reid_cost_weight * reid_d
            fused[iou < self.proximity_thresh] = 1.0
        else:
            fused = iou_d

        return linear_assignment(fused, thresh)

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
