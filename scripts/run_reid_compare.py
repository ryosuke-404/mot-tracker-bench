"""
Re-ID model comparison: run hybridsort_tuned with every person Re-ID model
from boxmot, generate one annotated video per Re-ID model.

Usage:
    .venv/bin/python scripts/run_reid_compare.py \
        --video   assets/test3.mp4 \
        --out_dir outputs/reid_compare \
        --gt_persons 6 \
        --skip 2

Output:
    outputs/reid_compare/{stem}_hybridsort_tuned_{reid_name}.mp4
    outputs/reid_compare/eval_reid_compare.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── boxmot imported lazily after torch is loaded via tracking modules ─────────
def _get_boxmot():
    """Import boxmot after torch is already in sys.modules (avoids NumPy ABI conflict)."""
    import torch  # noqa: F401 — ensure torch is loaded first
    import boxmot  # noqa: F401
    from boxmot.reid.core.auto_backend import ReidAutoBackend
    from boxmot.reid.core.config import TRAINED_URLS
    from boxmot.utils import WEIGHTS
    return ReidAutoBackend, TRAINED_URLS, WEIGHTS

from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import Detection as TrackDetection
from tracking.hybridsort import HybridSORTTracker
from tracking.track import STrack, TrackState
from reid.gallery import ReIDGallery

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_reid_compare")

# ── Person Re-ID models (loaded at runtime) ───────────────────────────────────
PERSON_REID_MODELS: list[str] = []  # populated in main() after torch is ready

# ── Colour palette for Re-ID model families ───────────────────────────────────
FAMILY_COLORS = {
    "osnet_x0_25": (150, 200, 255),
    "osnet_x0_5":  (100, 180, 255),
    "osnet_x0_75": ( 50, 160, 255),
    "osnet_x1_0":  (  0, 140, 255),
    "osnet_ibn":   (  0, 100, 200),
    "osnet_ain":   (  0,  60, 180),
    "resnet50_fc": (  0, 220, 120),
    "resnet50":    (  0, 180,  80),
    "mobilenetv2_x1_0": (255, 160,  0),
    "mobilenetv2_x1_4": (255, 120,  0),
    "mlfn":        (200,   0, 200),
    "hacnn":       (180,   0, 160),
    "lmbn":        (255, 220,   0),
    "clip":        (  0, 255, 200),
}

_STATE_COLORS = {
    TrackState.Tentative: (200, 200,   0),
    TrackState.Confirmed: (  0, 230,   0),
    TrackState.Lost:      (  0, 120, 255),
}

TRACKER_NAME = "hybridsort_tuned"


def get_family_color(model_name: str) -> tuple[int, int, int]:
    for key, color in FAMILY_COLORS.items():
        if model_name.startswith(key):
            return color
    return (180, 180, 180)


# ── boxmot Re-ID extractor wrapper ────────────────────────────────────────────

class BoxmotReIDExtractor:
    """Wraps boxmot ReidAutoBackend to match our extract_batch() interface."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device_str = device
        self._backend = None
        self.feature_dim = 512  # will be set after first inference
        self._tag = model_name

    def load_model(self) -> None:
        import torch
        ReidAutoBackend, _, WEIGHTS = _get_boxmot()
        weight_path = WEIGHTS / self.model_name
        if not weight_path.exists():
            logger.info("Downloading %s …", self.model_name)
            WEIGHTS.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.device_str)
        self._backend = ReidAutoBackend(weights=weight_path, device=device, half=False)
        logger.info("Loaded Re-ID model: %s", self.model_name)

    def extract_batch(self, frame: np.ndarray, bboxes: list[np.ndarray]) -> np.ndarray:
        if not bboxes or self._backend is None:
            return np.zeros((len(bboxes) if bboxes else 0, self.feature_dim), np.float32)

        xyxys = np.array([[b[0], b[1], b[2], b[3]] for b in bboxes], dtype=np.float32)
        feats = self._backend.model.get_features(xyxys, frame)

        if feats is None or (isinstance(feats, np.ndarray) and feats.size == 0):
            return np.zeros((len(bboxes), self.feature_dim), np.float32)

        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[np.newaxis, :]

        self.feature_dim = feats.shape[1]
        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)


# ── tracker factory ───────────────────────────────────────────────────────────

def make_hybridsort_tuned():
    return HybridSORTTracker(
        track_thresh=0.50, track_buffer=180, match_thresh=0.90, min_hits=3,
        iou_thresh_stage2=0.45, iou_weight=0.40, height_weight=0.20,
        shape_weight=0.10, reid_weight=0.30,
    )


# ── drawing ───────────────────────────────────────────────────────────────────

class FT:
    __slots__ = ("track_id", "tlbr", "state")
    def __init__(self, tid, tlbr, state):
        self.track_id = tid
        self.tlbr = tlbr
        self.state = state


def draw_frame(
    frame: np.ndarray,
    ft_list: list,
    reid_name: str,
    frame_id: int,
    fps: float,
    inference_ms: float,
    frag_count: int,
    rescue_count: int,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    out = frame.copy()
    tag_color = get_family_color(reid_name)
    sf = frame_h / 720

    # ── bounding boxes ────────────────────────────────────────────────────────
    for ft in ft_list:
        x1, y1, x2, y2 = ft.tlbr.astype(int)
        sc = _STATE_COLORS.get(ft.state, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), sc, 3)
        box_h = max(y2 - y1, 1)
        fs = max(0.7, min(2.0, box_h / 220))
        th = max(1, int(fs * 1.5))
        lbl = f" ID {ft.track_id} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, tth), bl = cv2.getTextSize(lbl, font, fs, th)
        by2 = max(tth + bl + 8, y1);  by1 = max(0, y1 - tth - bl - 8)
        cv2.rectangle(out, (x1, by1), (x1 + tw, by2), sc, cv2.FILLED)
        cv2.putText(out, lbl, (x1, by2 - bl - 3), font, fs, (0, 0, 0), th, cv2.LINE_AA)

    # ── Re-ID model name banner ───────────────────────────────────────────────
    tf = cv2.FONT_HERSHEY_DUPLEX
    t_fs = 1.3 * sf;  t_th = max(2, int(sf * 2))

    # two lines: tracker on top, reid below
    line1 = f"  {TRACKER_NAME.upper()}  "
    line2 = f"  Re-ID: {reid_name}  "
    (w1, h1), bl1 = cv2.getTextSize(line1, tf, t_fs * 0.85, t_th)
    (w2, h2), bl2 = cv2.getTextSize(line2, tf, t_fs, t_th)
    bw = max(w1, w2) + int(40 * sf)
    bh = h1 + h2 + bl1 + bl2 + int(24 * sf)
    bx1 = (frame_w - bw) // 2;  by1_b = int(8 * sf)

    ovl = out.copy()
    cv2.rectangle(ovl, (bx1, by1_b), (bx1 + bw, by1_b + bh), tag_color, cv2.FILLED)
    cv2.addWeighted(ovl, 0.85, out, 0.15, 0, out)
    pad = int(10 * sf)
    cv2.putText(out, line1, ((frame_w - w1) // 2, by1_b + pad + h1),
                tf, t_fs * 0.85, (0, 0, 0), t_th, cv2.LINE_AA)
    cv2.putText(out, line2, ((frame_w - w2) // 2, by1_b + pad + h1 + bl1 + h2 + int(6 * sf)),
                tf, t_fs, (0, 0, 0), t_th, cv2.LINE_AA)

    # ── stats panel ───────────────────────────────────────────────────────────
    sf_font = cv2.FONT_HERSHEY_DUPLEX
    sf_fs = 0.58 * sf;  sf_th = max(1, int(sf));  lh = int(28 * sf);  pd = int(12 * sf)
    stats = [
        f"Tracker : {TRACKER_NAME}",
        f"ReID    : {reid_name}",
        f"Frame   : {frame_id}",
        f"FPS     : {fps:.1f}",
        f"Infer   : {inference_ms:.0f} ms",
        f"Active  : {sum(1 for ft in ft_list if ft.state == TrackState.Confirmed)}",
        f"Frags   : {frag_count}",
        f"Rescue  : {rescue_count}",
    ]
    (pw, _), _ = cv2.getTextSize("ReID    : osnet_x1_0_msmt17", sf_font, sf_fs, sf_th)
    pw2 = pw + pd * 2;  ph = lh * len(stats) + pd * 2
    ovl2 = out.copy()
    cv2.rectangle(ovl2, (0, 0), (pw2, ph), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ovl2, 0.60, out, 0.40, 0, out)
    for i, text in enumerate(stats):
        cv2.putText(out, text, (pd, pd + lh * (i + 1) - int(4 * sf)),
                    sf_font, sf_fs, (220, 220, 220), sf_th, cv2.LINE_AA)
    return out


# ── one Re-ID model run ───────────────────────────────────────────────────────

def run_one(
    frames: list[np.ndarray],
    detector: YOLOXDetector,
    reid_ext: BoxmotReIDExtractor,
    out_path: Path,
    src_fps: float,
    skip: int,
    gt_persons: int,
) -> dict:
    STrack.reset_id_counter()
    tracker = make_hybridsort_tuned()
    gallery = ReIDGallery(max_gallery_size=200, similarity_thresh=0.72,
                          max_embeddings_per_id=8)

    frame_h, frame_w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (frame_w, frame_h))

    all_tids: set[int] = set()
    track_birth: dict[int, int] = {}
    track_death: dict[int, int] = {}
    rescue_count = 0
    rescue_sims: list[float] = []
    det_times: list[float] = []
    active_cnts: list[int] = []
    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0
    t_start = time.perf_counter()

    for fid, frame in enumerate(frames, 1):
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        high_embs = reid_ext.extract_batch(frame, [d.bbox for d in high_raw]) \
            if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32)

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
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

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

        active_cnts.append(len(active))
        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt = 0;  fps_t0 = time.perf_counter()

        ft_list = [FT(t.track_id, t.tlbr.copy(), t.state) for t in active]
        vis = draw_frame(frame, ft_list, reid_ext.model_name,
                         fid, fps_disp, det_times[-1],
                         len(all_tids), rescue_count)
        writer.write(vis)

    writer.release()
    total_s = time.perf_counter() - t_start

    n_frames = len(frames)
    long_lived = sum(1 for tid in all_tids
                     if (track_death.get(tid, 0) - track_birth.get(tid, 0)) >= n_frames * 0.5)
    avg_active = float(np.mean(active_cnts)) if active_cnts else 0.0

    m: dict = {
        "reid_model":    reid_ext.model_name,
        "tracker":       TRACKER_NAME,
        "frames":        n_frames,
        "frags":         len(all_tids),
        "rescue_count":  rescue_count,
        "avg_rescue_sim": float(np.mean(rescue_sims)) if rescue_sims else 0.0,
        "avg_active":    avg_active,
        "total_s":       total_s,
    }
    if gt_persons > 0:
        gt = gt_persons
        id_prec  = min(gt / max(len(all_tids), 1), 1.0)
        coverage = min(long_lived / gt, 1.0)
        od_rate  = avg_active / gt
        under_p  = min(1.0 / max(od_rate, 0.01), 1.0)
        comps    = [id_prec, coverage, under_p]
        denom    = sum(1.0 / max(c, 1e-6) for c in comps)
        m.update({
            "gt_persons":   gt,
            "id_precision": id_prec,
            "coverage":     coverage,
            "od_rate":      od_rate,
            "gt_score":     len(comps) / denom if denom > 0 else 0.0,
        })
    return m


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test3.mp4")
    parser.add_argument("--out_dir",     default="outputs/reid_compare")
    parser.add_argument("--gt_persons",  type=int, default=6)
    parser.add_argument("--frames",      type=int, default=0)
    parser.add_argument("--skip",        type=int, default=2)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load frames ───────────────────────────────────────────────────────────
    print(f"\nLoading video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {orig_w}×{orig_h}  {src_fps:.1f}fps  ({total_src} frames)")

    frames: list[np.ndarray] = []
    fid = 0;  max_f = args.frames if args.frames > 0 else 10**9
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid % args.skip != 0:
            continue
        frames.append(frame)
    cap.release()
    stem = Path(args.video).stem
    print(f"  → {len(frames)} frames loaded (skip={args.skip})")

    # ── load detector ─────────────────────────────────────────────────────────
    print(f"\nLoading YOLOX ({args.yolox_model}) …")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10,
                             nms_thresh=0.45)
    detector.load_model()
    print("  → detector ready")

    # ── populate Re-ID model list (torch already loaded via detector) ─────────
    global PERSON_REID_MODELS
    if not PERSON_REID_MODELS:
        _, TRAINED_URLS, _ = _get_boxmot()
        _VEHICLE = {"clip_veri.pt", "clip_vehicleid.pt"}
        PERSON_REID_MODELS = [k for k in TRAINED_URLS.keys() if k not in _VEHICLE]
        print(f"\nPerson Re-ID models available: {len(PERSON_REID_MODELS)}")

    # ── run each Re-ID model ──────────────────────────────────────────────────
    n = len(PERSON_REID_MODELS)
    print(f"\nRunning {TRACKER_NAME} × {n} Re-ID models × {len(frames)} frames …\n")

    all_results: list[dict] = []

    for idx, reid_name in enumerate(PERSON_REID_MODELS, 1):
        out_path = out_dir / f"{stem}_{TRACKER_NAME}_{reid_name.replace('.pt', '')}.mp4"
        print(f"  [{idx:>2}/{n}] {reid_name:<35}", end="  ", flush=True)

        try:
            reid_ext = BoxmotReIDExtractor(reid_name, device=args.device)
            reid_ext.load_model()

            t0 = time.perf_counter()
            m = run_one(frames, detector, reid_ext, out_path,
                        src_fps, args.skip, args.gt_persons)
            elapsed = time.perf_counter() - t0

            score_str = f"GTscore={m['gt_score']:.3f}" if args.gt_persons > 0 else ""
            print(f"✓  {elapsed:.0f}s  frags={m['frags']:>3}  rescue={m['rescue_count']:>3}  {score_str}")
            all_results.append(m)
        except Exception as e:
            print(f"✗  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    sep = "─" * 95
    print(sep)
    if args.gt_persons > 0:
        sorted_r = sorted(all_results, key=lambda x: -x.get("gt_score", 0))
        print(f"  {'Rank':<5} {'Re-ID Model':<35} │ {'GTscore':>7} │ {'IDprec':>6} │ "
              f"{'Cover':>5} │ {'ODrate':>6} │ {'Frags':>5} │ {'Rescue':>6} │ {'Tot s':>6}")
        print(sep)
        for rank, m in enumerate(sorted_r, 1):
            print(f"  #{rank:<4} {m['reid_model'].replace('.pt',''):<35} │ "
                  f"{m.get('gt_score',0):>7.3f} │ {m.get('id_precision',0):>6.3f} │ "
                  f"{m.get('coverage',0):>5.3f} │ {m.get('od_rate',0):>5.2f}x │ "
                  f"{m['frags']:>5} │ {m['rescue_count']:>6} │ {m['total_s']:>6.0f}")
    else:
        sorted_r = sorted(all_results, key=lambda x: x["frags"])
        print(f"  {'Rank':<5} {'Re-ID Model':<35} │ {'Frags':>5} │ {'Rescue':>6} │ "
              f"{'AvgSim':>6} │ {'Active':>6} │ {'Tot s':>6}")
        print(sep)
        for rank, m in enumerate(sorted_r, 1):
            print(f"  #{rank:<4} {m['reid_model'].replace('.pt',''):<35} │ "
                  f"{m['frags']:>5} │ {m['rescue_count']:>6} │ "
                  f"{m['avg_rescue_sim']:>6.3f} │ {m['avg_active']:>6.2f} │ {m['total_s']:>6.0f}")
    print(sep)

    json_path = out_dir / "eval_reid_compare.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nVideos  → {out_dir}/  ({len(all_results)} files)")
    print(f"Metrics → {json_path}")


if __name__ == "__main__":
    main()
