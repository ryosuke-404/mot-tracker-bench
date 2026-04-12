"""
Multi-tracker comparison: process a video with multiple trackers and generate
one annotated output video per tracker, plus a precision evaluation.

Usage:
    .venv/bin/python scripts/run_video_compare.py \
        --video   assets/test.mp4 \
        --out_dir outputs/compare \
        --gt_persons 0 \
        --trackers sort_tuned ocsort_tuned bytetrack deepocsort_tuned hybridsort_tuned

Each tracker produces:
    outputs/compare/test_<tracker>.mp4   — annotated video
    outputs/compare/eval_compare.json    — precision metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.bytetrack import ByteTracker, Detection as TrackDetection
from tracking.botsort import BoTSORT
from tracking.ocsort import OCSORTTracker
from tracking.sort import SORTTracker
from tracking.hybridsort import HybridSORTTracker
from tracking.deepocsort import DeepOCSORTTracker
from tracking.track import STrack, TrackState
from detection.yolox_detector import YOLOXDetector
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_video_compare")

# ── colour palette per tracker ────────────────────────────────────────────────
TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
    "sort_tuned":        (0,   200, 255),   # cyan
    "ocsort_tuned":      (50,  205, 50),    # lime green
    "hybridsort_tuned":  (255, 165, 0),     # orange
    "deepocsort_tuned":  (200, 0,   200),   # magenta
    "bytetrack":         (0,   120, 255),   # blue
    "bytetrack_tuned":   (80,  160, 255),   # light blue
    "sort":              (150, 150, 0),     # olive
    "ocsort":            (0,   180, 100),   # teal
}
DEFAULT_COLOR = (180, 180, 180)

_STATE_COLORS = {
    TrackState.Tentative: (200, 200, 0),
    TrackState.Confirmed: (0,   230, 0),
    TrackState.Lost:      (0,   120, 255),
}


# ── tracker factory ───────────────────────────────────────────────────────────

def make_tracker(name: str):
    if name == "sort":
        return SORTTracker(track_thresh=0.45, max_age=90, min_hits=3, iou_thresh=0.30)
    elif name == "sort_tuned":
        return SORTTracker(track_thresh=0.60, max_age=90, min_hits=3, iou_thresh=0.30)
    elif name == "bytetrack":
        return ByteTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                           min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.35)
    elif name == "bytetrack_tuned":
        return ByteTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                           min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.45)
    elif name == "ocsort":
        return OCSORTTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                             min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.35,
                             ocm_weight=0.20, delta_t=3)
    elif name == "ocsort_tuned":
        return OCSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                             min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.45,
                             ocm_weight=0.20, delta_t=5)
    elif name == "hybridsort_tuned":
        return HybridSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                                 min_hits=3, iou_thresh_stage2=0.45,
                                 iou_weight=0.40, height_weight=0.20,
                                 shape_weight=0.10, reid_weight=0.30)
    elif name == "deepocsort_tuned":
        return DeepOCSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                                 min_hits=3, iou_thresh_stage2=0.45,
                                 reid_weight=0.55, ocm_weight=0.15)
    else:
        raise ValueError(f"Unknown tracker: {name}")


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_frame(
    frame: np.ndarray,
    tracks: list,
    tracker_name: str,
    frame_id: int,
    fps: float,
    inference_ms: float,
    frag_count: int,
    rescue_count: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    out = frame.copy()
    tag_color = TRACKER_COLORS.get(tracker_name, DEFAULT_COLOR)

    # ── per-track boxes ──────────────────────────────────────────────────────
    for track in tracks:
        x1, y1, x2, y2 = track.tlbr.astype(int)
        state_color = _STATE_COLORS.get(track.state, (180, 180, 180))

        # Thicker box
        cv2.rectangle(out, (x1, y1), (x2, y2), state_color, 3)

        # ID badge
        box_h = max(y2 - y1, 1)
        font_scale = max(0.8, min(2.0, box_h / 220))
        thickness  = max(1, int(font_scale * 1.5))
        label = f" ID {track.track_id} "
        font  = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        bx1 = x1
        by2 = max(th + baseline + 8, y1)
        by1 = max(0, y1 - th - baseline - 8)
        bx2 = x1 + tw
        cv2.rectangle(out, (bx1, by1), (bx2, by2), state_color, cv2.FILLED)
        cv2.putText(out, label, (bx1, by2 - baseline - 3),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # ── large tracker name banner (top centre) ───────────────────────────────
    sf        = frame_h / 720
    tag_font  = cv2.FONT_HERSHEY_DUPLEX
    tag_fs    = 1.6 * sf
    tag_th    = max(2, int(sf * 2.5))
    tag_label = f"  {tracker_name.upper()}  "
    (tw, th), tbl = cv2.getTextSize(tag_label, tag_font, tag_fs, tag_th)

    pad_x = int(20 * sf)
    pad_y = int(12 * sf)
    bx1 = (frame_w - tw) // 2 - pad_x
    by1 = int(10 * sf)
    bx2 = (frame_w + tw) // 2 + pad_x
    by2 = by1 + th + tbl + pad_y * 2

    # Filled banner background
    overlay = out.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), tag_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.putText(out, tag_label,
                ((frame_w - tw) // 2, by1 + pad_y + th),
                tag_font, tag_fs, (0, 0, 0), tag_th, cv2.LINE_AA)

    # ── stats panel (top-left) ───────────────────────────────────────────────
    st_font = cv2.FONT_HERSHEY_DUPLEX
    st_fs   = 0.65 * sf
    st_th   = max(1, int(sf))
    line_h  = int(32 * sf)
    pad     = int(16 * sf)
    stats   = [
        f"Tracker: {tracker_name}",
        f"Frame  : {frame_id}",
        f"FPS    : {fps:.1f}",
        f"Infer  : {inference_ms:.0f} ms",
        f"Active : {len([t for t in tracks if t.state == TrackState.Confirmed])}",
        f"Frags  : {frag_count}",
        f"Rescue : {rescue_count}",
    ]
    (pw, _), _ = cv2.getTextSize("Tracker: sort_tuned", st_font, st_fs, st_th)
    panel_w = pw + pad * 2
    panel_h = line_h * len(stats) + pad * 2
    ovl = out.copy()
    cv2.rectangle(ovl, (0, 0), (panel_w, panel_h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ovl, 0.60, out, 0.40, 0, out)
    for i, text in enumerate(stats):
        cv2.putText(out, text, (pad, pad + line_h * (i + 1) - int(6 * sf)),
                    st_font, st_fs, (220, 220, 220), st_th, cv2.LINE_AA)

    return out


# ── single tracker run ────────────────────────────────────────────────────────

def run_tracker(
    frames: list[np.ndarray],
    tracker_name: str,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    out_path: Path,
    src_fps: float,
    skip: int,
    gt_persons: int,
) -> dict:
    logger.info("  [%s] start  → %s", tracker_name, out_path.name)
    STrack.reset_id_counter()

    tracker = make_tracker(tracker_name)
    gallery = ReIDGallery(max_gallery_size=200, similarity_thresh=0.72,
                          max_embeddings_per_id=8)

    frame_h, frame_w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (frame_w, frame_h))

    # counters
    all_tids: set[int] = set()
    rescue_count = 0
    rescue_sims: list[float] = []
    active_counts: list[int] = []
    track_life: dict[int, int] = {}
    det_times: list[float] = []
    reid_times: list[float] = []

    fps_t0   = time.perf_counter()
    fps_cnt  = 0
    fps_disp = 0.0

    t_start = time.perf_counter()

    for fid, frame in enumerate(frames, 1):
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        high_embs = reid_ext.extract_batch(frame, [d.bbox for d in high_raw]) \
            if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32)
        reid_times.append((time.perf_counter() - t1) * 1000)

        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        active = tracker.update(high_dets, low_dets)
        active_ids = {t.track_id for t in active}
        all_tids.update(active_ids)

        confirmed = [t for t in active if t.state == TrackState.Confirmed]
        for t in confirmed:
            if t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, fid)
            track_life[t.track_id] = track_life.get(t.track_id, 0) + 1

        # Re-ID rescue
        for t in active:
            if t.state == TrackState.Tentative and t.hits == 1 and t.reid_embedding is not None:
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    rescue_count += 1
                    rescue_sims.append(sim)
                    old_id = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old_id)

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_counts.append(len(active))

        # FPS
        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt  = 0
            fps_t0   = time.perf_counter()

        vis = draw_frame(
            frame, active, tracker_name, fid, fps_disp,
            det_times[-1], len(all_tids), rescue_count, frame_w, frame_h,
        )
        writer.write(vis)

    writer.release()
    total_s = time.perf_counter() - t_start
    logger.info("  [%s] done  %.1fs  frags=%d  rescue=%d", tracker_name, total_s, len(all_tids), rescue_count)

    # ── compute metrics ─────────────────────────────────────────────────────
    n_frames = len(frames)
    long_life_thresh = n_frames * 0.5
    long_lived = sum(1 for v in track_life.values() if v >= long_life_thresh)

    m: dict = {
        "tracker":        tracker_name,
        "frames":         n_frames,
        "track_fragments": len(all_tids),
        "rescue_count":   rescue_count,
        "avg_rescue_sim": float(np.mean(rescue_sims)) if rescue_sims else 0.0,
        "avg_active_per_frame": float(np.mean(active_counts)) if active_counts else 0.0,
        "avg_det_ms":     float(np.mean(det_times)) if det_times else 0.0,
        "total_time_s":   total_s,
    }

    if gt_persons > 0:
        gt = gt_persons
        id_precision = min(gt / max(len(all_tids), 1), 1.0)
        coverage     = min(long_lived / gt, 1.0)
        od_rate      = m["avg_active_per_frame"] / gt
        under_pen    = min(1.0 / max(od_rate, 0.01), 1.0)
        components   = [id_precision, coverage, under_pen]
        denom        = sum(1.0 / max(c, 1e-6) for c in components)
        gt_score     = len(components) / denom if denom > 0 else 0.0
        m.update({
            "gt_persons":   gt_persons,
            "id_precision": id_precision,
            "coverage":     coverage,
            "over_det_rate": od_rate,
            "gt_score":     gt_score,
        })

    return m


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test.mp4")
    parser.add_argument("--out_dir",     default="outputs/compare")
    parser.add_argument("--trackers",    nargs="+",
                        default=["sort_tuned", "ocsort_tuned", "bytetrack",
                                 "deepocsort_tuned", "hybridsort_tuned"])
    parser.add_argument("--gt_persons",  type=int, default=0)
    parser.add_argument("--frames",      type=int, default=0,
                        help="Max frames to process (0=all)")
    parser.add_argument("--skip",        type=int, default=1)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load frames ──────────────────────────────────────────────────────────
    logger.info("Loading video: %s", args.video)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open {args.video}"
    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("  %dx%d @ %.1f fps  (%d frames total)", orig_w, orig_h, src_fps, total_frames)

    frames: list[np.ndarray] = []
    fid = 0
    max_f = args.frames if args.frames > 0 else 10**9
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid % args.skip != 0:
            continue
        frames.append(frame)
    cap.release()
    logger.info("  → %d frames loaded (skip=%d)", len(frames), args.skip)

    # ── load detector & reid ─────────────────────────────────────────────────
    logger.info("Loading YOLOX (%s) …", args.yolox_model)
    detector = YOLOXDetector(
        model_name=args.yolox_model,
        device=args.device,
        high_score_thresh=0.45,
        low_score_thresh=0.10,
        nms_thresh=0.45,
    )
    detector.load_model()
    logger.info("  detector ready")

    logger.info("Loading Re-ID (osnet) …")
    reid_ext = FastReIDExtractor(model_path=None, device=args.device,
                                 input_size=(256, 128), use_onnx=False)
    reid_ext.load_model()
    logger.info("  reid ready (feature_dim=%d)", reid_ext.feature_dim)

    # ── run each tracker ─────────────────────────────────────────────────────
    stem = Path(args.video).stem
    results: list[dict] = []

    print(f"\nRunning {len(args.trackers)} trackers × {len(frames)} frames …\n")

    for i, trk in enumerate(args.trackers, 1):
        out_path = out_dir / f"{stem}_{trk}.mp4"
        print(f"  [{i}/{len(args.trackers)}] {trk:<30}", end="  ", flush=True)
        t0 = time.perf_counter()
        m = run_tracker(frames, trk, detector, reid_ext, out_path,
                        src_fps, args.skip, args.gt_persons)
        elapsed = time.perf_counter() - t0
        frag_str = f"frags={m['track_fragments']}"
        score_str = f"GTscore={m['gt_score']:.3f}" if args.gt_persons > 0 else ""
        print(f"✓  {elapsed:.1f}s  {frag_str}  {score_str}")
        results.append(m)

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    sep = "─" * 90
    print(sep)
    if args.gt_persons > 0:
        results_sorted = sorted(results, key=lambda x: -x.get("gt_score", 0))
        print(f"  {'Rank':<5} {'Tracker':<25} │ {'GTscore':>7} │ {'IDprec':>6} │ {'Cover':>5} │ {'ODrate':>6} │ {'Frags':>5} │ {'Rescue':>6} │ {'Tot s':>6}")
        print(sep)
        for rank, m in enumerate(results_sorted, 1):
            print(f"  #{rank:<4} {m['tracker']:<25} │ {m['gt_score']:>7.3f} │ {m['id_precision']:>6.3f} │ {m['coverage']:>5.3f} │ {m['over_det_rate']:>5.2f}x │ {m['track_fragments']:>5} │ {m['rescue_count']:>6} │ {m['total_time_s']:>6.1f}")
    else:
        results_sorted = sorted(results, key=lambda x: x["track_fragments"])
        print(f"  {'Rank':<5} {'Tracker':<25} │ {'Frags':>5} │ {'Rescue':>6} │ {'AvgSim':>6} │ {'Active':>6} │ {'Tot s':>6}")
        print(sep)
        for rank, m in enumerate(results_sorted, 1):
            print(f"  #{rank:<4} {m['tracker']:<25} │ {m['track_fragments']:>5} │ {m['rescue_count']:>6} │ {m['avg_rescue_sim']:>6.3f} │ {m['avg_active_per_frame']:>6.2f} │ {m['total_time_s']:>6.1f}")
    print(sep)

    # ── save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "eval_compare.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nVideos   → {out_dir}/")
    print(f"Metrics  → {json_path}")


if __name__ == "__main__":
    main()
