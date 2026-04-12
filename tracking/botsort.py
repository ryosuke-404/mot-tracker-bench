"""
BoT-SORT: Robust Associations Multi-Pedestrian Tracking.

Reference: "BoT-SORT: Robust Associations Multi-Pedestrian Tracking"
           Aharon et al., 2022  (BSD-style academic license)

Key differences from ByteTrack:
  1. Global Motion Compensation (GMC) — corrects Kalman predictions for
     camera movement (critical for bus/vehicle-mounted cameras).
  2. Re-ID embeddings are used in STAGE-1 cost matrix as a primary signal,
     not just a rescue mechanism.
  3. Fused IoU-ReID cost: combines motion (IoU) and appearance (cosine)
     with adaptive weighting.
  4. Low-score second stage identical to ByteTrack.

The same STrack / KalmanFilter classes from ByteTrack are reused.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection,
    iou_matrix,
    cosine_distance_matrix,
    linear_assignment,
)
from tracking.gmc import GMC


class BoTSORT:
    """
    BoT-SORT multi-object tracker.

    Drop-in replacement for ByteTracker with identical update() signature.
    """

    def __init__(
        self,
        track_thresh: float = 0.50,
        track_buffer: int = 150,
        match_thresh: float = 0.85,      # gate for stage-1 fused cost
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.50,
        reid_cost_weight: float = 0.55,  # higher than ByteTrack — Re-ID is primary
        gmc_method: str = "orb",         # "orb" | "ecc" | "none"
        proximity_thresh: float = 0.5,   # min IoU to even consider Re-ID cost
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.reid_cost_weight = reid_cost_weight
        self.proximity_thresh = proximity_thresh

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0

        self.gmc = GMC(method=gmc_method)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        self.gmc.reset()
        STrack.reset_id_counter()

    # ------------------------------------------------------------------
    def update(
        self,
        high_dets: list[Detection],
        low_dets: list[Detection],
        frame: np.ndarray | None = None,   # needed for GMC
    ) -> list[STrack]:
        """
        Process one frame. `frame` is the raw BGR image for GMC; pass None
        to disable camera-motion compensation this frame.
        """
        self.frame_id += 1

        # ---- Step 0: Kalman predict + GMC correction --------------------
        all_active = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_active)

        if frame is not None and self.gmc.method != "none":
            H = self.gmc.apply(frame)
            self.gmc.apply_to_tracks(H, all_active)

        # ---- Step 1: Stage-1 — high-score dets vs active tracks ---------
        active_tracks = self.tracked_stracks

        (matches_s1, unmatched_tracks_s1, unmatched_high_idx) = self._associate_fused(
            high_dets, active_tracks, self.match_thresh
        )
        for t_idx, d_idx in matches_s1:
            active_tracks[t_idx].update(
                high_dets[d_idx].bbox,
                high_dets[d_idx].score,
                high_dets[d_idx].embedding,
            )

        unmatched_active = [active_tracks[i] for i in unmatched_tracks_s1]

        # ---- Step 2: Stage-2 — low-score dets vs unmatched tracks ------
        candidate_s2 = unmatched_active + self.lost_stracks
        (matches_s2, unmatched_s2_idx, _) = self._associate_iou_only(
            low_dets, candidate_s2, self.iou_thresh_stage2
        )
        matched_s2 = {t_idx for t_idx, _ in matches_s2}
        for t_idx, d_idx in matches_s2:
            candidate_s2[t_idx].update(
                low_dets[d_idx].bbox, low_dets[d_idx].score, None
            )
        for i, track in enumerate(candidate_s2):
            if i not in matched_s2:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # ---- Step 3: New tentative tracks for unmatched high dets ------
        for d_idx in unmatched_high_idx:
            det = high_dets[d_idx]
            if det.score < self.track_thresh:
                continue
            new_track = STrack(det.bbox, det.score)
            new_track.activate(self.frame_id)
            if det.embedding is not None:
                new_track.reid_embedding = det.embedding
                new_track.embedding_history.append(det.embedding)
            self.tracked_stracks.append(new_track)

        # ---- Step 4: Rebuild lists, remove expired Lost tracks ----------
        newly_lost_ids = {candidate_s2[i].track_id for i in unmatched_s2_idx}
        still_lost: list[STrack] = []
        for track in self.lost_stracks + [
            t for t in unmatched_active if t.state == TrackState.Lost
        ]:
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            else:
                if track.track_id not in {t.track_id for t in still_lost}:
                    still_lost.append(track)
        self.lost_stracks = still_lost

        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state in (TrackState.Confirmed, TrackState.Tentative)
        ]
        return list(self.tracked_stracks)

    # ------------------------------------------------------------------
    def get_confirmed_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]

    def get_lost_tracks(self) -> list[STrack]:
        return list(self.lost_stracks)

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------

    def _associate_fused(
        self,
        detections: list[Detection],
        tracks: list[STrack],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        BoT-SORT fused cost: proximity-gated (IoU) × appearance (Re-ID).

        Cells where IoU < proximity_thresh are set to ∞ (gated out) before
        adding the Re-ID cost. This prevents appearance from overriding a
        severe spatial mismatch.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])

        iou   = iou_matrix(track_boxes, det_boxes)       # (M, N)
        iou_d = 1.0 - iou                                 # IoU distance

        # Check whether all tracks and dets have embeddings
        t_embeds = [t.reid_embedding for t in tracks]
        d_embeds = [d.embedding      for d in detections]
        all_t = all(e is not None for e in t_embeds)
        all_d = all(e is not None for e in d_embeds)

        if all_t and all_d:
            T = np.stack(t_embeds)
            D = np.stack(d_embeds)
            reid_d = cosine_distance_matrix(T, D)         # (M, N)

            # Gate: pairs with IoU < proximity_thresh → set fused cost = 1
            fused = (
                (1.0 - self.reid_cost_weight) * iou_d
                + self.reid_cost_weight * reid_d
            )
            gate = iou < self.proximity_thresh
            fused[gate] = 1.0
        else:
            fused = iou_d

        return linear_assignment(fused, thresh)

    def _associate_iou_only(
        self,
        detections: list[Detection],
        tracks: list[STrack],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        return linear_assignment(cost, thresh)
