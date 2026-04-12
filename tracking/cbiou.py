"""
C-BIoU: Consistent-Velocity Bounded IoU Tracker
Liu et al., 2023 — MIT License

Core idea: before computing IoU between a track and a detection, expand
the track's predicted bounding box by a factor that scales with the
track's observed velocity. This creates a "catch zone" that tolerates
position uncertainty, especially during fast motion or camera shake.

Two-stage matching identical to ByteTrack, but IoU is computed against
the velocity-expanded bbox instead of the raw Kalman prediction.
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, linear_assignment


# ---------------------------------------------------------------------------

def expand_bbox(tlbr: np.ndarray, expand: float) -> np.ndarray:
    """Expand bbox by `expand` fraction in each direction."""
    x1, y1, x2, y2 = tlbr
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h   = (x2 - x1), (y2 - y1)
    w2 = w * (1 + expand)
    h2 = h * (1 + expand)
    return np.array([cx - w2/2, cy - h2/2, cx + w2/2, cy + h2/2])


def velocity_from_track(track: STrack) -> float:
    """Estimate track velocity magnitude (pixels/frame) from Kalman state."""
    if track.mean is None:
        return 0.0
    vx, vy = track.mean[4], track.mean[5]
    return float(np.sqrt(vx**2 + vy**2))


def expanded_iou_matrix(
    tracks: list[STrack],
    det_boxes: np.ndarray,
    base_expand: float = 0.2,
    vel_scale: float = 0.01,
) -> np.ndarray:
    """
    Compute IoU between velocity-expanded track boxes and detections.

    expand = base_expand + vel_scale * velocity_magnitude
    capped at 0.6 to prevent runaway catch zones.
    """
    expanded_boxes = []
    for t in tracks:
        vel = velocity_from_track(t)
        expand = min(base_expand + vel_scale * vel, 0.6)
        expanded_boxes.append(expand_bbox(t.tlbr, expand))
    track_boxes = np.array(expanded_boxes)
    return iou_matrix(track_boxes, det_boxes)


# ---------------------------------------------------------------------------

class CBIoUTracker:
    """
    C-BIoU multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).
    Uses velocity-expanded bounding boxes for IoU matching.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.80,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.45,
        base_expand: float = 0.20,
        vel_scale: float = 0.01,
    ) -> None:
        self.track_thresh      = track_thresh
        self.track_buffer      = track_buffer
        self.match_thresh      = match_thresh
        self.min_hits          = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.base_expand       = base_expand
        self.vel_scale         = vel_scale

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

        # Stage 1: high dets vs active tracks (C-BIoU)
        (m1, unm_t1, unm_d1) = self._associate_cbiou(
            high_dets, self.tracked_stracks, self.match_thresh
        )
        for t_idx, d_idx in m1:
            self.tracked_stracks[t_idx].update(
                high_dets[d_idx].bbox, high_dets[d_idx].score, high_dets[d_idx].embedding
            )
        unmatched_active = [self.tracked_stracks[i] for i in unm_t1]

        # Stage 2: low dets vs unmatched + lost (plain IoU)
        candidate_s2 = unmatched_active + self.lost_stracks
        (m2, unm_s2, _) = self._associate_plain_iou(
            low_dets, candidate_s2, self.iou_thresh_stage2
        )
        matched_s2 = {t for t, _ in m2}
        for t_idx, d_idx in m2:
            candidate_s2[t_idx].update(
                low_dets[d_idx].bbox, low_dets[d_idx].score, None
            )
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
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
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

    def get_confirmed_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]

    def get_lost_tracks(self) -> list[STrack]:
        return list(self.lost_stracks)

    def _associate_cbiou(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        det_boxes = np.array([d.bbox for d in detections])
        iou = expanded_iou_matrix(tracks, det_boxes, self.base_expand, self.vel_scale)
        cost = 1.0 - iou
        return linear_assignment(cost, thresh)

    def _associate_plain_iou(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        return linear_assignment(cost, thresh)
