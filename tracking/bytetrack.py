"""
ByteTrack: Multi-Object Tracker with two-stage association.

Reference: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
           Zhang et al., ECCV 2022  (MIT License)

Key innovation over SORT/DeepSORT:
  Low-confidence detections (normally discarded) are saved for a second
  matching pass against tracks that had no high-confidence match.
  This dramatically reduces ID switches caused by partial occlusions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.track import STrack, TrackState


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single person detection from RT-DETRv2."""
    bbox: np.ndarray     # [x1, y1, x2, y2] pixel coords
    score: float
    embedding: np.ndarray | None = None   # FastReID feature (optional)


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bboxes_a  shape (M, 4)
        bboxes_b  shape (N, 4)

    Returns:
        iou_mat   shape (M, N)  values in [0, 1]
    """
    if bboxes_a.shape[0] == 0 or bboxes_b.shape[0] == 0:
        return np.zeros((bboxes_a.shape[0], bboxes_b.shape[0]))

    ax1, ay1, ax2, ay2 = (bboxes_a[:, i] for i in range(4))
    bx1, by1, bx2, by2 = (bboxes_b[:, i] for i in range(4))

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def cosine_distance_matrix(
    embeds_a: np.ndarray, embeds_b: np.ndarray
) -> np.ndarray:
    """
    Pairwise cosine distance (1 - cosine_similarity).

    Args:
        embeds_a  shape (M, D)  — L2-normalized
        embeds_b  shape (N, D)  — L2-normalized

    Returns:
        dist_mat  shape (M, N)  values in [0, 2]
    """
    if embeds_a.shape[0] == 0 or embeds_b.shape[0] == 0:
        return np.zeros((embeds_a.shape[0], embeds_b.shape[0]))
    # dot product of L2-normalized vectors = cosine similarity
    sim = embeds_a @ embeds_b.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    Run Hungarian algorithm and return matched / unmatched indices.

    Returns:
        matches         list of (row_idx, col_idx)
        unmatched_rows  list of row indices
        unmatched_cols  list of col indices
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [
        (r, c) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= thresh
    ]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    Multi-object tracker using ByteTrack's two-stage detection association.

    Stages:
      1. High-score detections  ↔  active (Confirmed + Tentative) tracks
         Cost: fused (IoU distance + Re-ID cosine distance)
      2. Low-score detections   ↔  unmatched tracks from stage 1
         Cost: IoU only (low-score detections are too noisy for Re-ID)
      3. New tentative tracks for unmatched high-score detections
      4. Mark unmatched tracks as Lost; remove tracks past track_buffer
    """

    def __init__(
        self,
        track_thresh: float = 0.50,
        track_buffer: int = 150,
        match_thresh: float = 0.80,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.50,
        reid_cost_weight: float = 0.40,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.reid_cost_weight = reid_cost_weight

        self.tracked_stracks: list[STrack] = []   # Confirmed + Tentative
        self.lost_stracks: list[STrack] = []       # Lost but within buffer
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
        low_dets: list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        """
        Process one frame's detections and return all active tracks.

        Args:
            high_dets  detections with score ≥ track_thresh
            low_dets   detections with low_thresh ≤ score < track_thresh

        Returns:
            List of STrack objects that are Confirmed or Tentative.
        """
        self.frame_id += 1

        # ---- Step 0: advance Kalman prediction for all tracks -----------
        all_active = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_active)

        # ---- Step 1: stage-1 — high-score dets vs. active tracks -------
        active_tracks = self.tracked_stracks  # Confirmed + Tentative

        (
            matches_s1,
            unmatched_tracks_s1,
            unmatched_high_det_idx,
        ) = self._associate(high_dets, active_tracks, self.match_thresh, use_reid=True)

        # Update matched tracks
        for t_idx, d_idx in matches_s1:
            active_tracks[t_idx].update(
                high_dets[d_idx].bbox,
                high_dets[d_idx].score,
                high_dets[d_idx].embedding,
            )

        # Tracks not matched in stage 1
        unmatched_tracks_stage1 = [active_tracks[i] for i in unmatched_tracks_s1]

        # ---- Step 2: stage-2 — low-score dets vs. unmatched tracks ----
        # Use IoU only; also include Lost tracks that are still within buffer
        candidate_tracks_s2 = unmatched_tracks_stage1 + self.lost_stracks

        (
            matches_s2,
            unmatched_tracks_s2,
            _unmatched_low_det_idx,
        ) = self._associate(
            low_dets, candidate_tracks_s2, self.iou_thresh_stage2, use_reid=False
        )

        for t_idx, d_idx in matches_s2:
            candidate_tracks_s2[t_idx].update(
                low_dets[d_idx].bbox,
                low_dets[d_idx].score,
                None,  # don't update Re-ID from low-score dets
            )

        # Mark truly unmatched tracks as Lost
        matched_stage2_track_idx = {t_idx for t_idx, _ in matches_s2}
        for i, track in enumerate(candidate_tracks_s2):
            if i not in matched_stage2_track_idx:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # ---- Step 3: initialize new tentative tracks -------------------
        for d_idx in unmatched_high_det_idx:
            det = high_dets[d_idx]
            if det.score < self.track_thresh:
                continue
            new_track = STrack(det.bbox, det.score)
            new_track.activate(self.frame_id)
            if det.embedding is not None:
                new_track.reid_embedding = det.embedding
                new_track.embedding_history.append(det.embedding)
            self.tracked_stracks.append(new_track)

        # ---- Step 4: rebuild track lists --------------------------------
        # Collect newly lost tracks from both active and candidate pools
        newly_lost = [
            t for t in unmatched_tracks_stage1
            if t.state == TrackState.Lost
        ] + [
            candidate_tracks_s2[i]
            for i in unmatched_tracks_s2
            if candidate_tracks_s2[i].state == TrackState.Lost
        ]

        # Deduplicate (a track may appear in both unmatched lists)
        seen_ids: set[int] = set()
        dedup_lost: list[STrack] = []
        for t in newly_lost:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                dedup_lost.append(t)

        # Remove tracks that have exceeded the buffer
        still_lost: list[STrack] = []
        for track in self.lost_stracks + dedup_lost:
            if track.track_id in seen_ids and track not in dedup_lost:
                continue  # already counted above
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            else:
                still_lost.append(track)

        self.lost_stracks = still_lost

        # Active = Confirmed + Tentative (anything updated this frame)
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _associate(
        self,
        detections: list[Detection],
        tracks: list[STrack],
        cost_thresh: float,
        use_reid: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Build cost matrix and run Hungarian matching.

        Returns:
            matches            list of (track_idx, det_idx)
            unmatched_tracks   list of track indices
            unmatched_dets     list of det indices
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])        # (M, 4)
        det_boxes = np.array([d.bbox for d in detections])       # (N, 4)

        iou_dist = 1.0 - iou_matrix(track_boxes, det_boxes)      # (M, N)

        if use_reid:
            # Build Re-ID embeddings for tracks and detections
            track_embeds = np.array(
                [t.reid_embedding for t in tracks if t.reid_embedding is not None]
            )
            det_embeds = np.array(
                [d.embedding for d in detections if d.embedding is not None]
            )

            if track_embeds.shape[0] == len(tracks) and det_embeds.shape[0] == len(detections):
                reid_dist = cosine_distance_matrix(track_embeds, det_embeds)  # (M, N)
                cost = (
                    (1.0 - self.reid_cost_weight) * iou_dist
                    + self.reid_cost_weight * reid_dist
                )
            else:
                cost = iou_dist
        else:
            cost = iou_dist

        return linear_assignment(cost, cost_thresh)
