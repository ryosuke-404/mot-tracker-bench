"""
SMILEtrack: SiMILarity LEarning for Multiple Object Tracking
Wang et al., AAAI 2023 — MIT License

Core innovations:
  1. SiamFC-inspired similarity: patch cross-correlation between track
     and detection crops as an additional matching cue (requires frame).
     When frame is None, falls back to Re-ID cosine distance.

  2. Hierarchical matching (3 stages):
     Stage A: Confirmed tracks × high dets (cosine + IoU)
     Stage B: Tentative + unmatched confirmed × high dets (IoU only)
     Stage C: All unmatched × low dets (IoU only)

  3. Strict Re-ID gate: a match is only accepted if BOTH:
       IoU > iou_gate  AND  cosine_sim > cos_gate
     This prevents appearance-only matches when spatial overlap is zero.

Note: the cross-correlation similarity (SiamFC branch) is approximated
here using the Re-ID embeddings from the feature extractor since we don't
have the FPN backbone from the original paper.
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection, iou_matrix, cosine_distance_matrix, linear_assignment
)


class SMILETracker:
    """
    SMILEtrack multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.85,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.45,
        reid_weight: float = 0.45,
        iou_gate:  float = 0.10,   # min IoU to accept a match
        cos_gate:  float = 0.30,   # min Re-ID similarity to accept a match
    ) -> None:
        self.track_thresh      = track_thresh
        self.track_buffer      = track_buffer
        self.match_thresh      = match_thresh
        self.min_hits          = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.reid_weight = reid_weight
        self.iou_gate    = iou_gate
        self.cos_gate    = cos_gate

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

        confirmed = [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]
        tentative = [t for t in self.tracked_stracks if t.state == TrackState.Tentative]

        # Stage A: confirmed × high dets (cosine + IoU with double gate)
        (mA, unm_cA, unm_dA) = self._associate_smile(
            high_dets, confirmed, self.match_thresh
        )
        for t_idx, d_idx in mA:
            confirmed[t_idx].update(
                high_dets[d_idx].bbox, high_dets[d_idx].score, high_dets[d_idx].embedding
            )

        remaining_confirmed = [confirmed[i] for i in unm_cA]
        remaining_high      = [high_dets[i] for i in unm_dA]

        # Stage B: tentative + unmatched confirmed × remaining high (IoU only)
        stage_b_tracks = remaining_confirmed + tentative
        (mB, unm_bB, unm_dB) = self._associate_iou(
            remaining_high, stage_b_tracks, self.match_thresh
        )
        for t_idx, d_idx in mB:
            stage_b_tracks[t_idx].update(
                remaining_high[d_idx].bbox, remaining_high[d_idx].score,
                remaining_high[d_idx].embedding
            )

        unmatched_active = [stage_b_tracks[i] for i in unm_bB]

        # Stage C: unmatched active + lost × low dets (IoU only)
        candidate_c = unmatched_active + self.lost_stracks
        (mC, unm_cC, _) = self._associate_iou(
            low_dets, candidate_c, self.iou_thresh_stage2
        )
        matched_c = {t for t, _ in mC}
        for t_idx, d_idx in mC:
            candidate_c[t_idx].update(
                low_dets[d_idx].bbox, low_dets[d_idx].score, None
            )
        for i, track in enumerate(candidate_c):
            if i not in matched_c:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # New tracks for still-unmatched high dets (Stage B unmatched)
        already_used_high = {unm_dA[j] for j in unm_dB}   # original indices
        for orig_idx in already_used_high:
            det = high_dets[orig_idx]
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

    def _associate_smile(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        """
        SMILEtrack joint cost with double gate:
          Only accept match if IoU > iou_gate AND cosine_sim > cos_gate.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])

        iou   = iou_matrix(track_boxes, det_boxes)     # (M, N)
        iou_d = 1.0 - iou

        t_embs = [t.reid_embedding for t in tracks]
        d_embs = [d.embedding      for d in detections]

        if all(e is not None for e in t_embs) and all(e is not None for e in d_embs):
            T = np.stack(t_embs)
            D = np.stack(d_embs)
            reid_d = cosine_distance_matrix(T, D)      # (M, N)
            cost   = (1 - self.reid_weight) * iou_d + self.reid_weight * reid_d

            # Double gate: block pairs failing either gate
            cos_sim  = 1.0 - reid_d
            bad_iou  = iou     < self.iou_gate
            bad_cos  = cos_sim < self.cos_gate
            cost[bad_iou | bad_cos] = 1.0
        else:
            cost = iou_d

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
