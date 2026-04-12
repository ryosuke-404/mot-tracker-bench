"""
Main entry point for the bus OD tracking system.

Run:
    python scripts/run_bus.py --config config/system.yaml

For systemd autostart on Jetson:
    ExecStart=/usr/bin/python3 /opt/bus-od/scripts/run_bus.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.rtdetr_detector import RTDETRv2Detector
from tracking.sort import SORTTracker
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery
from tripwire.tripwire_manager import TripwireManager
from od.od_tracker import ODTracker
from od.stop_manager import StopManager
from storage.db import Database
from gps.gps_reader import GpsReader, MockGpsReader
from pipeline.frame_processor import FrameProcessor
from pipeline.visualization import (
    draw_tracks, draw_tripwire_lines, draw_crossing_events, draw_stats
)
from system.lifecycle import SystemLifecycle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
def build_processor(cfg: dict, camera_cfg: dict) -> FrameProcessor:
    """Instantiate all components for a single camera stream."""
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    reid_cfg = cfg["reid"]
    gps_cfg = cfg["gps"]
    sys_cfg = cfg["system"]
    lifecycle_cfg = cfg["lifecycle"]

    detector = RTDETRv2Detector(
        model_name=det_cfg["model_name"],
        device=sys_cfg["device"],
        high_score_thresh=det_cfg["high_score_thresh"],
        low_score_thresh=det_cfg["low_score_thresh"],
        input_size=tuple(det_cfg["input_size"]),
        onnx_path=det_cfg.get("onnx_path"),
    )
    detector.load_model()

    tracker = SORTTracker(
        track_thresh=trk_cfg["track_thresh"],
        max_age=trk_cfg["max_age"],
        min_hits=trk_cfg["min_hits"],
        iou_thresh=trk_cfg["iou_thresh"],
    )

    reid_extractor = FastReIDExtractor(
        model_path=reid_cfg.get("model_path"),
        device=sys_cfg["device"],
        input_size=tuple(reid_cfg["input_size"]),
        feature_dim=reid_cfg["feature_dim"],
    )
    if reid_cfg["enabled"]:
        reid_extractor.load_model()

    reid_gallery = ReIDGallery(
        max_gallery_size=reid_cfg["gallery_max_size"],
        similarity_thresh=reid_cfg["similarity_thresh"],
        max_embeddings_per_id=reid_cfg["embeddings_per_id"],
    )

    tripwire_mgr = TripwireManager(
        config_path=cfg["tripwire"]["config_path"],
        sequence_timeout_frames=cfg["tripwire"]["sequence_timeout_frames"],
        direction_dot_thresh=cfg["tripwire"]["direction_dot_thresh"],
    )
    tripwire_mgr.load_config()

    stop_mgr = StopManager(
        stop_list_path=gps_cfg["stop_list_path"],
        proximity_radius_m=gps_cfg["proximity_radius_m"],
    )
    stop_mgr.load_stops()

    os.makedirs(os.path.dirname(sys_cfg["db_path"]), exist_ok=True)
    db = Database(sys_cfg["db_path"])

    od_tracker = ODTracker(
        stop_manager=stop_mgr,
        db=db,
        route_id=sys_cfg["route_id"],
        vehicle_id=sys_cfg["vehicle_id"],
    )

    if gps_cfg["enabled"]:
        gps_reader = GpsReader(port=gps_cfg["port"], baud_rate=gps_cfg["baud_rate"])
        gps_reader.start()
    else:
        gps_reader = MockGpsReader()
        gps_reader.start()

    w, h = camera_cfg["resolution"]
    processor = FrameProcessor(
        camera_id=camera_cfg["id"],
        detector=detector,
        tracker=tracker,
        reid_extractor=reid_extractor,
        reid_gallery=reid_gallery,
        tripwire_manager=tripwire_mgr,
        od_tracker=od_tracker,
        gps_reader=gps_reader,
        frame_width=w,
        frame_height=h,
    )
    return processor, od_tracker, db


# ---------------------------------------------------------------------------
def run_camera(
    camera_cfg: dict,
    processor: FrameProcessor,
    od_tracker: ODTracker,
    lifecycle: SystemLifecycle,
    visualize: bool = False,
) -> None:
    source = camera_cfg["source"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open camera source: %s", source)
        return

    w, h = camera_cfg["resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, camera_cfg.get("fps", 25))

    logger.info("Camera '%s' opened: %s", camera_cfg["id"], source)

    fps_counter = 0
    fps_start = time.perf_counter()
    fps_display = 0.0

    while lifecycle.is_running:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed on '%s'", camera_cfg["id"])
            time.sleep(0.1)
            continue

        lifecycle.heartbeat()
        result = processor.process_frame(frame, datetime.utcnow())

        # FPS calculation (every 30 frames)
        fps_counter += 1
        if fps_counter >= 30:
            fps_display = fps_counter / (time.perf_counter() - fps_start)
            fps_counter = 0
            fps_start = time.perf_counter()

        if visualize:
            vis = draw_tracks(frame, result.active_tracks, show_reid=True)
            vis = draw_crossing_events(vis, result.crossing_events, w, h)
            vis = draw_stats(vis, result.frame_id, od_tracker.active_count(),
                             fps_display, result.inference_ms)
            cv2.imshow(f"Bus OD — {camera_cfg['id']}", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                lifecycle._initiate_shutdown()
                break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    # Close any orphan records at route end
    active_ids = {t.track_id for t in processor.tracker.get_confirmed_tracks()}
    od_tracker.close_orphan_records(active_ids, datetime.utcnow())
    logger.info("Camera '%s' stopped. OD matrix: %s", camera_cfg["id"], od_tracker.get_od_matrix())


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bus OD Tracking System")
    parser.add_argument("--config", default="config/system.yaml")
    parser.add_argument("--visualize", action="store_true",
                        help="Show debug visualization window")
    parser.add_argument("--camera", default=None,
                        help="Run only this camera ID (default: all)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["system"].get("log_level", "INFO"))
    logger.info("Bus OD Tracking System starting")

    lifecycle = SystemLifecycle(
        ignition_gpio_pin=cfg["lifecycle"].get("ignition_gpio_pin"),
        shutdown_delay_s=cfg["lifecycle"]["shutdown_delay_s"],
        watchdog_interval_s=cfg["lifecycle"]["watchdog_interval_s"],
    )
    lifecycle.setup_signal_handlers()
    lifecycle.wait_for_ignition()
    lifecycle.monitor_ignition_off()
    lifecycle.start_watchdog()

    cameras = cfg.get("cameras", [])
    if args.camera:
        cameras = [c for c in cameras if c["id"] == args.camera]

    if not cameras:
        logger.error("No cameras configured")
        sys.exit(1)

    # For single-camera or sequential processing; extend to threading for multi-camera
    for camera_cfg in cameras:
        processor, od_tracker, db = build_processor(cfg, camera_cfg)
        lifecycle.on_shutdown(lambda: od_tracker.close_orphan_records(set(), datetime.utcnow()))

        run_camera(camera_cfg, processor, od_tracker, lifecycle, visualize=args.visualize)


if __name__ == "__main__":
    main()
