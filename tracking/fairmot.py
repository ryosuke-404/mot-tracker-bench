"""
FairMOT-style Tracker
Wang et al., IJCV 2022 — MIT License

FairMOT's key idea: treat detection and Re-ID as a single joint task.
The detection heatmap branch and Re-ID branch share the same encoder,
so every detected center point carries a Re-ID embedding "for free".

This implementation captures the FairMOT *tracking algorithm*:
  - Single-stage matching: all detections matched with all tracks at once
    (no high/low score split, no cascade by age)
  - Joint cost: α × (1 - Re-ID similarity) + (1 - α) × (1 - IoU)
  - Re-ID embeddings from detections are used for ALL matches, not just rescues
  - Track birth/death similar to ByteTrack but without the second low-score stage

Without the actual FairMOT CenterNet backbone, we use our existing detectors
(RT-DETRv2 or YOLOX) + Re-ID extractor to supply (bbox, score, embedding)
tuples — the algorithmic matching is faithful to the paper.
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, cosine_distance_matrix, linear_assignment


class FairMOTTracker:
    """
    FairMOT-style multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).

    Key algorithmic difference from ByteTrack:
      Single-stage association with joint Re-ID + IoU cost.
      No second pass with low-score detections.
      All detections (≥ det_thresh) are treated equally.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.80,
        min_hits: int = 3,
        reid_weight: float = 0.50,      # weight of Re-ID cost in joint cost
        iou_weight:  float = 0.50,      # weight of IoU cost  in joint cost
        proximity_gate: float = 0.05,   # min IoU to consider a match at all
    ) -> None:
        assert abs(reid_weight + iou_weight - 1.0) < 1e-6, \
            "reid_weight + iou_weight must equal 1.0"

        self.track_thresh    = track_thresh
        self.track_buffer    = track_buffer
        self.match_thresh    = match_thresh
        self.min_hits        = min_hits
        self.reid_weight     = reid_weight
        self.iou_weight      = iou_weight
        self.proximity_gate  = proximity_gate

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    # ------------------------------------------------------------------
    def update(
        self,
        high_dets: list[Detection],
        low_dets:  list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        """
        FairMOT single-stage update.

        high_dets and low_dets are merged; all detections at or above
        track_thresh participate in the single-stage match.
        low_dets below track_thresh are discarded (no ByteTrack second stage).
        """
        self.frame_id += 1

        # FairMOT: single detection pool (no high/low split)
        all_dets = [d for d in (high_dets + low_dets) if d.score >= self.track_thresh]

        # Predict Kalman for all tracks
        all_tracks = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_tracks)

        # Combine active (confirmed+tentative) and lost tracks for matching
        candidate_tracks = self.tracked_stracks + self.lost_stracks

        # ---- Single-stage joint matching --------------------------------
        (matches, unmatched_t_idx,
         unmatched_d_idx) = self._associate_joint(candidate_tracks, all_dets,
                                                   self.match_thresh)

        matched_t_ids = {t_idx for t_idx, _ in matches}

        for t_idx, d_idx in matches:
            track = candidate_tracks[t_idx]
            det   = all_dets[d_idx]
            track.update(det.bbox, det.score, det.embedding)

        # Mark unmatched tracks
        for i, track in enumerate(candidate_tracks):
            if i not in matched_t_ids:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # ---- New tentative tracks for unmatched detections ---------------
        for d_idx in unmatched_d_idx:
            det = all_dets[d_idx]
            new_track = STrack(det.bbox, det.score)
            new_track.activate(self.frame_id)
            if det.embedding is not None:
                new_track.reid_embedding = det.embedding
                new_track.embedding_history.append(det.embedding)
            self.tracked_stracks.append(new_track)

        # ---- Rebuild lists -----------------------------------------------
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in candidate_tracks
                         if t.state == TrackState.Lost
                         and t not in self.tracked_stracks]):
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            elif track.track_id not in {t.track_id for t in still_lost}:
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

    def _associate_joint(
        self,
        tracks: list[STrack],
        detections: list[Detection],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        FairMOT joint cost: weighted combination of Re-ID cosine and IoU.

        Proximity gate: pairs with IoU < proximity_gate are capped at cost=1.0
        to prevent matches between spatially disjoint track-detection pairs
        even when their appearance is similar.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])

        iou   = iou_matrix(track_boxes, det_boxes)    # (M, N)
        iou_d = 1.0 - iou

        t_embeds = [t.reid_embedding for t in tracks]
        d_embeds = [d.embedding      for d in detections]

        if (all(e is not None for e in t_embeds)
                and all(e is not None for e in d_embeds)):
            T = np.stack(t_embeds)
            D = np.stack(d_embeds)
            reid_d = cosine_distance_matrix(T, D)     # (M, N)
            cost   = self.iou_weight * iou_d + self.reid_weight * reid_d
        else:
            # Fallback to IoU-only if embeddings unavailable
            cost = iou_d

        # Proximity gate: very low IoU → block the match
        cost[iou < self.proximity_gate] = 1.0

        return linear_assignment(cost, thresh)
