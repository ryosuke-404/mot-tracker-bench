"""
Demo script: ByteTrack + strong privacy blur on detected persons.

All person bounding boxes are pixelated (mosaic) before any annotation is
drawn, making individual identification impossible while preserving the
tracker visualization (ID badges, tripwire, stats).

Usage:
    .venv/bin/python scripts/run_demo_blur.py \
        --video   assets/demo.mp4 \
        --out     outputs/demo_blurred.mp4 \
        --detector yolox \
        --yolox-model yolox_s
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.rtdetr_detector import RTDETRv2Detector
from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import ByteTracker, Detection as TrackDetection
from tracking.track import STrack, TrackState
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery
from tripwire.tripwire_manager import TripwireManager, CrossingType
from od.od_tracker import ODTracker
from od.stop_manager import StopManager
from storage.db import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_demo_blur")


# ---------------------------------------------------------------------------
# Privacy blur (mosaic / pixelation)
# ---------------------------------------------------------------------------

def blur_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                mosaic_size: int = 16) -> None:
    """Pixelate the bounding box region in-place (strong privacy protection)."""
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    # Shrink to mosaic_size then enlarge back → strong pixelation
    small = cv2.resize(roi, (max(1, w // mosaic_size), max(1, h // mosaic_size)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    TrackState.Tentative: (200, 200, 0),
    TrackState.Confirmed: (0, 230, 0),
    TrackState.Lost:      (0, 120, 255),
}


def draw_frame(
    frame: np.ndarray,
    tracks: list[STrack],
    crossing_events,
    frame_id: int,
    fps: float,
    inference_ms: float,
    on_bus_count: int,
    total_boarded: int,
    total_alighted: int,
    frame_w: int,
    frame_h: int,
    detector_name: str = "yolox",
) -> np.ndarray:
    out = frame.copy()

    # Tripwire lines
    for y_norm, color, label in [
        (0.60, (0, 255, 255), "outer"),
        (0.74, (0, 165, 255), "inner"),
    ]:
        y_px = int(y_norm * frame_h)
        x_l  = int(0.25 * frame_w)
        x_r  = int(0.75 * frame_w)
        cv2.line(out, (x_l, y_px), (x_r, y_px), color, 2)
        cv2.putText(out, label, (x_l + 4, y_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for track in tracks:
        color = _STATE_COLORS.get(track.state, (180, 180, 180))
        x1, y1, x2, y2 = track.tlbr.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        box_h = max(y2 - y1, 1)
        font_scale = max(0.8, min(2.0, box_h / 220))
        thickness  = max(1, int(font_scale * 1.5))
        label = f" ID {track.track_id} "
        font  = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_x1 = x1
        badge_y1 = max(0, y1 - th - baseline - 8)
        badge_x2 = x1 + tw
        badge_y2 = max(th + baseline + 8, y1)
        cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
        cv2.putText(out, label, (badge_x1, badge_y2 - baseline - 3),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    ev_fs   = max(1.2, frame_h / 720)
    ev_th   = max(2, int(ev_fs * 1.8))
    ev_font = cv2.FONT_HERSHEY_DUPLEX
    for ev in crossing_events:
        cx_n, cy_n = ev.position
        cx_px = int(cx_n * frame_w)
        cy_px = int(cy_n * frame_h)
        if ev.crossing_type == CrossingType.BOARD:
            ev_label = f"  BOARD  ID {ev.track_id}  "
            ev_color = (0, 210, 70)
        else:
            ev_label = f"  ALIGHT  ID {ev.track_id}  "
            ev_color = (0, 50, 240)
        (ew, eh), ebl = cv2.getTextSize(ev_label, ev_font, ev_fs, ev_th)
        ex1 = cx_px - ew // 2 - 6
        ey2 = cy_px - 10
        ey1 = ey2 - eh - ebl - 10
        ex2 = cx_px + ew // 2 + 6
        cv2.rectangle(out, (ex1, ey1), (ex2, ey2), ev_color, cv2.FILLED)
        cv2.putText(out, ev_label, (ex1 + 6, ey2 - ebl - 3),
                    ev_font, ev_fs, (255, 255, 255), ev_th, cv2.LINE_AA)
        cv2.circle(out, (cx_px, cy_px + int(20 * ev_fs)), int(14 * ev_fs), ev_color, -1)

    sf      = frame_h / 720
    st_font = cv2.FONT_HERSHEY_DUPLEX
    st_fs   = 0.65 * sf
    st_th   = max(1, int(sf))
    line_h  = int(32 * sf)
    pad     = int(16 * sf)
    stats   = [
        f"Tracker: ByteTrack",
        f"Det    : {detector_name.upper()}",
        f"Frame  : {frame_id}",
        f"FPS    : {fps:.1f}",
        f"Infer  : {inference_ms:.0f} ms",
        f"On bus : {on_bus_count}",
        f"Board  : {total_boarded}",
        f"Alight : {total_alighted}",
    ]
    (pw, _), _ = cv2.getTextSize("Tracker: ByteTrack", st_font, st_fs, st_th)
    panel_w = pw + pad * 2
    panel_h = line_h * len(stats) + pad * 2
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, out, 0.40, 0, out)
    for i, text in enumerate(stats):
        cv2.putText(out, text, (pad, pad + line_h * (i + 1) - int(6 * sf)),
                    st_font, st_fs, (220, 220, 220), st_th, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubGps:
    @property
    def latest_coord(self):
        return (35.0, 135.0)


class _StubStopManager:
    def get_current_stop(self, gps):
        return "STOP_UNKNOWN"


class _StubDb:
    def __init__(self):
        self._records: list[dict] = []
        self._next_id = 1

    def insert_od_event(self, record, route_id, vehicle_id):
        rid = self._next_id
        self._next_id += 1
        self._records.append({
            "id": rid,
            "track_id": record.track_id,
            "board_stop": record.board_stop_id,
            "board_ts": record.board_timestamp.isoformat(),
            "alight_stop": None,
            "alight_ts": None,
        })
        return rid

    def mark_alight(self, record_id, alight_stop, alight_ts, alight_gps):
        for r in self._records:
            if r["id"] == record_id:
                r["alight_stop"] = alight_stop
                r["alight_ts"] = alight_ts.isoformat()

    def to_list(self):
        return self._records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="assets/demo.mp4")
    parser.add_argument("--out",    default="outputs/demo_blurred.mp4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector", choices=["rtdetr", "yolox"], default="yolox")
    parser.add_argument("--yolox-model", default="yolox_s",
                        choices=["yolox_nano", "yolox_tiny", "yolox_s",
                                 "yolox_m", "yolox_l", "yolox_x"])
    parser.add_argument("--scale",  type=float, default=0.5)
    parser.add_argument("--skip",   type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--mosaic", type=int, default=16,
                        help="Mosaic block size in pixels (larger = stronger blur)")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open: {args.video}"
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %dx%d @ %.1ffps  (%d frames)", orig_w, orig_h, src_fps, total_frames)

    det_w = int(orig_w * args.scale)
    det_h = int(orig_h * args.scale)

    for fourcc_str in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(args.out, fourcc, src_fps / args.skip, (orig_w, orig_h))
        if writer.isOpened():
            logger.info("VideoWriter opened with codec %s", fourcc_str)
            break
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {args.out}")

    if args.detector == "yolox":
        logger.info("Loading YOLOX (%s) …", args.yolox_model)
        detector = YOLOXDetector(
            model_name=args.yolox_model,
            device=args.device,
            high_score_thresh=0.45,
            low_score_thresh=0.10,
            nms_thresh=0.45,
        )
        detector.load_model()
    else:
        logger.info("Loading RT-DETRv2 …")
        detector = RTDETRv2Detector(
            model_name="PekingU/rtdetr_v2_r50vd",
            device=args.device,
            high_score_thresh=0.45,
            low_score_thresh=0.10,
            input_size=(det_w, det_h),
        )
        detector.load_model()

    tracker = ByteTracker(
        track_thresh=0.50,
        track_buffer=150,
        match_thresh=0.80,
        min_hits=3,
    )

    logger.info("Loading Re-ID extractor …")
    reid_extractor = FastReIDExtractor(
        model_path=None,
        device=args.device,
        input_size=(256, 128),
        use_onnx=False,
    )
    reid_extractor.load_model()

    reid_gallery = ReIDGallery(
        max_gallery_size=200,
        similarity_thresh=0.72,
        max_embeddings_per_id=8,
    )

    tripwire_mgr = TripwireManager(
        config_path="config/tripwire_video.yaml",
        sequence_timeout_frames=60,
        direction_dot_thresh=0.25,
    )
    tripwire_mgr.load_config()

    stub_db = _StubDb()
    od_tracker = ODTracker(
        stop_manager=_StubStopManager(),
        db=stub_db,
        route_id="VIDEO_DEMO",
        vehicle_id="BUS_001",
    )
    gps = _StubGps()

    frame_id = 0
    proc_frame_id = 0
    prev_centroids: dict[int, tuple[float, float]] = {}
    total_boarded = 0
    total_alighted = 0
    fps_t0 = time.perf_counter()
    fps_count = 0
    fps_display = 0.0
    last_crossing_events = []

    logger.info("Starting processing (mosaic block=%dpx) …", args.mosaic)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % args.skip != 0:
            continue
        if args.max_frames and proc_frame_id >= args.max_frames:
            break
        proc_frame_id += 1

        ts = datetime.now(timezone.utc)

        if args.detector == "yolox":
            det_frame = frame
            scale_inv = 1.0
        elif args.scale < 1.0:
            det_frame = cv2.resize(frame, (det_w, det_h))
            scale_inv = 1.0 / args.scale
        else:
            det_frame = frame
            scale_inv = 1.0

        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(det_frame)
        inference_ms = (time.perf_counter() - t0) * 1000

        if scale_inv != 1.0:
            for d in high_raw:
                d.bbox *= scale_inv
            for d in low_raw:
                d.bbox *= scale_inv

        # --- Privacy blur: apply BEFORE annotation so boxes overlay on blurred content
        for d in high_raw:
            x1, y1, x2, y2 = d.bbox.astype(int)
            blur_region(frame, x1, y1, x2, y2, mosaic_size=args.mosaic)
        for d in low_raw:
            x1, y1, x2, y2 = d.bbox.astype(int)
            blur_region(frame, x1, y1, x2, y2, mosaic_size=args.mosaic)

        high_embeddings = reid_extractor.extract_batch(
            frame, [d.bbox for d in high_raw]
        ) if high_raw else np.zeros((0, reid_extractor.feature_dim), np.float32)

        high_dets = [
            TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embeddings[i])
            for i, d in enumerate(high_raw)
        ]
        low_dets = [
            TrackDetection(bbox=d.bbox, score=d.score)
            for d in low_raw
        ]

        active_tracks = tracker.update(high_dets, low_dets)

        confirmed = [t for t in active_tracks if t.state == TrackState.Confirmed]
        active_ids = {t.track_id for t in active_tracks}

        for track in confirmed:
            if track.reid_embedding is not None:
                reid_gallery.add_embedding(track.track_id, track.reid_embedding, proc_frame_id)

        for track in active_tracks:
            if track.state == TrackState.Tentative and track.hits == 1 and track.reid_embedding is not None:
                matched_id, sim = reid_gallery.query(track.reid_embedding, exclude_ids=active_ids)
                if matched_id is not None:
                    new_id = track.track_id
                    track.reassign_id(matched_id)
                    reid_gallery.remove_track(new_id)

        if proc_frame_id % 150 == 0:
            lost_ids = {t.track_id for t in tracker.lost_stracks}
            reid_gallery.prune_old_tracks(active_ids | lost_ids, 3000, proc_frame_id)

        curr_centroids = {
            t.track_id: (t.centroid[0] / orig_w, t.centroid[1] / orig_h)
            for t in active_tracks
        }
        crossing_events = tripwire_mgr.update(
            camera_id="front",
            frame_id=proc_frame_id,
            track_positions=curr_centroids,
            prev_positions=prev_centroids,
        )
        prev_centroids = curr_centroids

        for ev in crossing_events:
            od_tracker.process_event(ev, ts, gps.latest_coord)
            if ev.crossing_type == CrossingType.BOARD:
                total_boarded += 1
            else:
                total_alighted += 1

        if crossing_events:
            last_crossing_events = crossing_events

        fps_count += 1
        if fps_count >= 30:
            fps_display = fps_count / (time.perf_counter() - fps_t0)
            fps_count = 0
            fps_t0 = time.perf_counter()

        det_label = args.yolox_model if args.detector == "yolox" else "rtdetr_v2"
        vis = draw_frame(
            frame, active_tracks, last_crossing_events,
            proc_frame_id, fps_display, inference_ms,
            od_tracker.active_count(), total_boarded, total_alighted,
            orig_w, orig_h,
            detector_name=det_label,
        )
        last_crossing_events = []
        writer.write(vis)

        if proc_frame_id % 50 == 0:
            pct = frame_id / total_frames * 100
            logger.info("[%5.1f%%] frame=%d  tracks=%d  boarded=%d  alighted=%d",
                        pct, proc_frame_id, len(confirmed), total_boarded, total_alighted)

    cap.release()
    writer.release()

    od_tracker.close_orphan_records(set(), datetime.now(timezone.utc))

    od_json_path = Path(args.out).with_suffix(".json")
    with open(od_json_path, "w", encoding="utf-8") as f:
        json.dump(stub_db.to_list(), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  DEMO (PRIVACY BLURRED) — ByteTrack")
    print("=" * 60)
    print(f"  Frames processed : {proc_frame_id}")
    print(f"  Total boarded    : {total_boarded}")
    print(f"  Total alighted   : {total_alighted}")
    print(f"  Completed OD     : {od_tracker.total_completed()}")
    print(f"  Output video     : {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
