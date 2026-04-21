"""
厳選20コンボをtest10.mp4で実行し、評価スコアを動画内に大きく表示する。

コンボ内訳:
  [A] tuned 8トラッカー × none          (8本)
  [B] ベース代表 8トラッカー × none      (8本)
  [C] hybridsort_tuned × DE/ABC/ABCDE   (3本)
  [D] hybridsort_tuned + osnet_ibn rescue可視化 (1本)

Usage:
    .venv/bin/python scripts/run_top20.py \
        --video   assets/test10.mp4 \
        --out_dir outputs/test10_top20 \
        --gt_persons 10 \
        --skip 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import ByteTracker, Detection as TrackDetection
from tracking.botsort import BoTSORT
from tracking.ocsort import OCSORTTracker
from tracking.sort import SORTTracker
from tracking.strongsort import StrongSORTTracker
from tracking.deepsort import DeepSORTTracker
from tracking.hybridsort import HybridSORTTracker
from tracking.deepocsort import DeepOCSORTTracker
from tracking.smiletrack import SMILETracker
from tracking.sparsetrack import SparseTracker
from tracking.ghost import GHOSTTracker
from tracking.transtrack import TransTracker
from tracking.track import STrack, TrackState
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery
from tracking.postprocess import (
    compute_mean_embeddings, merge_by_appearance,
    cluster_reassign, apply_id_map, compose_maps,
)

# ── lazy boxmot import ────────────────────────────────────────────────────────
def _get_boxmot():
    import torch  # noqa
    import boxmot  # noqa
    from boxmot.reid.core.auto_backend import ReidAutoBackend
    from boxmot.utils import WEIGHTS
    return ReidAutoBackend, WEIGHTS

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── 20コンボ定義 ──────────────────────────────────────────────────────────────
COMBOS: list[tuple[str, str]] = [
    # [A] tuned × none (8本)
    ("hybridsort_tuned",  "none"),
    ("bytetrack_tuned",   "none"),
    ("botsort_tuned",     "none"),
    ("ocsort_tuned",      "none"),
    ("deepocsort_tuned",  "none"),
    ("sparsetrack_tuned", "none"),
    ("sort_tuned",        "none"),
    ("transtrack_tuned",  "none"),
    # [B] ベース代表 × none (8本)
    ("bytetrack",   "none"),
    ("botsort",     "none"),
    ("deepsort",    "none"),
    ("strongsort",  "none"),
    ("ghost",       "none"),
    ("ocsort",      "none"),
    ("deepocsort",  "none"),
    ("smiletrack",  "none"),
    # [C] hybridsort_tuned × 後処理 (3本)
    ("hybridsort_tuned", "de"),
    ("hybridsort_tuned", "abc"),
    ("hybridsort_tuned", "abcde"),
    # [D] rescue可視化 (1本) — 別フラグで管理
    ("hybridsort_tuned", "rescue"),
]

TRACKER_COLORS: dict[str, tuple] = {
    "sort_tuned":        (  0, 200, 255),
    "bytetrack":         (  0, 120, 255),
    "bytetrack_tuned":   ( 80, 160, 255),
    "ocsort":            (  0, 200, 100),
    "ocsort_tuned":      ( 50, 230,  50),
    "hybridsort":        (200, 120,   0),
    "hybridsort_tuned":  (255, 165,   0),
    "deepocsort":        (160,   0, 160),
    "deepocsort_tuned":  (220,   0, 220),
    "botsort":           (200,  80,  20),
    "botsort_tuned":     (255, 120,  40),
    "deepsort":          ( 20, 180, 180),
    "strongsort":        (  0, 140, 200),
    "smiletrack":        (  0, 220, 180),
    "sparsetrack_tuned": (210, 130, 230),
    "ghost":             (140, 140, 140),
    "transtrack_tuned":  (140, 230, 230),
}
POSTPROC_ACCENT = {
    "none":   (30,  30,  30),
    "abc":    (20, 130,  20),
    "de":     (180, 30,  30),
    "abcde":  (120, 20, 180),
    "rescue": (  0, 160,  80),
}
_STATE_COLORS = {
    TrackState.Tentative: (200, 200,   0),
    TrackState.Confirmed: (  0, 230,   0),
    TrackState.Lost:      (  0, 120, 255),
}
RESCUE_DISPLAY_FRAMES = 60


class FT(NamedTuple):
    track_id: int
    tlbr:     np.ndarray
    state:    TrackState


@dataclass
class RescueEvent:
    frame_id: int
    old_id:   int
    new_id:   int
    sim:      float
    tlbr:     np.ndarray


# ── tracker factory ───────────────────────────────────────────────────────────
def make_tracker(name: str):
    kw = dict(track_thresh=0.45, track_buffer=120, min_hits=3)
    if name in ("bytetrack", "bytetrack_tuned"):
        thresh = 0.90 if "tuned" in name else 0.85
        return ByteTracker(**kw, match_thresh=thresh,
                           iou_thresh_stage2=0.45, reid_cost_weight=0.35)
    if name in ("botsort", "botsort_tuned"):
        thresh = 0.88 if "tuned" in name else 0.85
        return BoTSORT(**kw, match_thresh=thresh)
    if name in ("ocsort", "ocsort_tuned"):
        thresh = 0.88 if "tuned" in name else 0.85
        return OCSORTTracker(**kw, match_thresh=thresh)
    if name == "sort_tuned":
        return SORTTracker(track_thresh=0.45, track_buffer=120, match_thresh=0.90, min_hits=3)
    if name == "deepsort":
        return DeepSORTTracker(track_thresh=0.45, track_buffer=120, min_hits=3,
                               max_cosine_dist=0.25, iou_thresh_stage2=0.45)
    if name == "strongsort":
        return StrongSORTTracker(track_thresh=0.45, track_buffer=120, min_hits=3,
                                 max_cosine_dist=0.25, iou_thresh_stage2=0.45)
    if name in ("hybridsort", "hybridsort_tuned"):
        rw = 0.30 if "tuned" in name else 0.20
        return HybridSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                                 min_hits=3, iou_thresh_stage2=0.45, iou_weight=0.40,
                                 height_weight=0.20, shape_weight=0.10, reid_weight=rw)
    if name in ("deepocsort", "deepocsort_tuned"):
        thresh = 0.88 if "tuned" in name else 0.85
        return DeepOCSORTTracker(**kw, match_thresh=thresh)
    if name in ("sparsetrack", "sparsetrack_tuned"):
        thresh = 0.88 if "tuned" in name else 0.85
        return SparseTracker(**kw, match_thresh=thresh)
    if name in ("transtrack", "transtrack_tuned"):
        thresh = 0.88 if "tuned" in name else 0.85
        return TransTracker(**kw, match_thresh=thresh)
    if name == "smiletrack":
        return SMILETracker(**kw, match_thresh=0.85)
    if name == "ghost":
        return GHOSTTracker(track_thresh=0.45, track_buffer=120,
                            match_thresh_short=0.35, match_thresh_long=0.50,
                            match_thresh_iou=0.45, min_hits=3, proximity_gate=0.10)
    raise ValueError(f"Unknown tracker: {name}")


# ── 評価スコア計算 ────────────────────────────────────────────────────────────
def compute_gt_score(all_tids, track_birth, track_death, active_cnts, n_frames, gt):
    long_lived = sum(1 for tid in all_tids
                     if (track_death.get(tid, 0) - track_birth.get(tid, 0)) >= n_frames * 0.5)
    avg_active  = float(np.mean(active_cnts)) if active_cnts else 0.0
    id_prec     = min(gt / max(len(all_tids), 1), 1.0)
    coverage    = min(long_lived / gt, 1.0)
    od_rate     = avg_active / gt
    under_p     = min(1.0 / max(od_rate, 0.01), 1.0)
    comps       = [id_prec, coverage, under_p]
    denom       = sum(1.0 / max(c, 1e-6) for c in comps)
    gt_score    = len(comps) / denom if denom > 0 else 0.0
    return dict(gt_score=gt_score, id_precision=id_prec, coverage=coverage,
                od_rate=od_rate, frags=len(all_tids), avg_active=avg_active)


# ── スコア表示（右下大バナー） ────────────────────────────────────────────────
def draw_score_panel(out: np.ndarray, metrics: dict | None, gt: int) -> np.ndarray:
    """動画右下に評価スコアを大きく表示"""
    h, w = out.shape[:2]
    sf   = h / 720
    font = cv2.FONT_HERSHEY_DUPLEX

    if metrics is None:
        lines = [("GTscore", "---"), ("IDprec", "---"),
                 ("Coverage", "---"), ("Frags", "---")]
    else:
        score   = metrics.get("gt_score", 0)
        idprec  = metrics.get("id_precision", 0)
        cov     = metrics.get("coverage", 0)
        frags   = metrics.get("frags", 0)
        # スコア色: 0.7以上=緑, 0.4〜0.7=黄, 未満=赤
        lines = [
            ("GTscore",  f"{score:.3f}",  (0,230,0) if score>0.7 else (0,220,220) if score>0.4 else (80,80,255)),
            ("IDprec",   f"{idprec:.3f}", (220,220,220)),
            ("Coverage", f"{cov:.3f}",    (220,220,220)),
            ("Frags",    f"{frags}",       (220,220,220)),
            ("GT",       f"{gt}人",        (180,180,180)),
        ]

    lh  = int(38 * sf)
    pd  = int(14 * sf)
    fs  = 0.75 * sf
    th  = max(1, int(sf * 1.5))
    fs2 = 1.1 * sf
    th2 = max(2, int(sf * 2))

    # パネル幅を計算
    max_w = 0
    for item in lines:
        label = item[0];  val = item[1]
        (lw, _), _ = cv2.getTextSize(f"{label} : {val}", font, fs2, th2)
        max_w = max(max_w, lw)
    pw = max_w + pd * 3
    ph = lh * len(lines) + pd * 2

    px1 = w - pw - int(10 * sf)
    py1 = h - ph - int(10 * sf)

    # 背景
    ovl = out.copy()
    cv2.rectangle(ovl, (px1 - pd, py1 - pd), (w - int(6*sf), h - int(6*sf)),
                  (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(ovl, 0.75, out, 0.25, 0, out)

    # テキスト描画
    for i, item in enumerate(lines):
        label = item[0];  val = item[1]
        color = item[2] if len(item) > 2 else (220, 220, 220)
        # GTscoreは特に大きく
        if label == "GTscore":
            (lw, lh2), _ = cv2.getTextSize(f"{label} : {val}", font, fs2, th2)
            cv2.putText(out, f"{label} : {val}",
                        (px1, py1 + pd + lh * (i + 1)),
                        font, fs2, color, th2, cv2.LINE_AA)
        else:
            cv2.putText(out, f"{label} : {val}",
                        (px1, py1 + pd + lh * (i + 1)),
                        font, fs, color, th, cv2.LINE_AA)
    return out


# ── 描画（上部バナー + 左上パネル + 右下スコア） ──────────────────────────────
def draw_frame(
    frame: np.ndarray,
    ft_list: list,
    id_map: dict,
    tracker_name: str,
    postproc: str,
    frame_id: int,
    fps: float,
    infer_ms: float,
    metrics: dict | None,
    gt: int,
    active_rescues: list | None = None,
) -> np.ndarray:
    h, w  = frame.shape[:2]
    out   = frame.copy()
    sf    = h / 720
    color = TRACKER_COLORS.get(tracker_name, (180, 180, 180))
    accent = POSTPROC_ACCENT.get(postproc, (30, 30, 30))

    # ── バウンディングボックス ─────────────────────────────────────────────────
    for ft in ft_list:
        disp_id = id_map.get(ft.track_id, ft.track_id) if id_map else ft.track_id
        x1, y1, x2, y2 = ft.tlbr.astype(int)
        sc = _STATE_COLORS.get(ft.state, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), sc, 3)
        bh = max(y2 - y1, 1)
        fs = max(0.6, min(2.0, bh / 180))
        th = max(1, int(fs * 1.5))
        lbl = f" ID {disp_id} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, tth), bl = cv2.getTextSize(lbl, font, fs, th)
        by2 = max(tth + bl + 8, y1);  by1 = max(0, y1 - tth - bl - 8)
        cv2.rectangle(out, (x1, by1), (x1 + tw, by2), sc, cv2.FILLED)
        cv2.putText(out, lbl, (x1, by2 - bl - 3), font, fs, (0, 0, 0), th, cv2.LINE_AA)

    # ── rescue イベント表示 ────────────────────────────────────────────────────
    if active_rescues:
        for ev in active_rescues:
            age   = frame_id - ev.frame_id
            alpha = max(0.0, 1.0 - age / RESCUE_DISPLAY_FRAMES)
            x1, y1, x2, y2 = ev.tlbr.astype(int)
            thick = max(4, int(12 * sf * alpha))
            ovl2  = out.copy()
            cv2.rectangle(ovl2, (x1, y1), (x2, y2), (0, 255, 100), thick)
            cv2.addWeighted(ovl2, alpha, out, 1 - alpha, 0, out)
            if age < RESCUE_DISPLAY_FRAMES // 2:
                msg  = f"RESCUED {ev.old_id}->{ev.new_id} sim={ev.sim:.2f}"
                mfont= cv2.FONT_HERSHEY_DUPLEX
                mfs  = 0.6 * sf;  mth = max(1, int(mfs * 1.8))
                (mw, mh), mbl = cv2.getTextSize(msg, mfont, mfs, mth)
                mx = max(0, x1);  my = max(mh + mbl + 4, y1 - 4)
                bg  = out.copy()
                cv2.rectangle(bg, (mx, my - mh - mbl - 4), (mx + mw + 8, my + 4),
                              (0, 60, 0), cv2.FILLED)
                cv2.addWeighted(bg, 0.75 * alpha, out, 1 - 0.75 * alpha, 0, out)
                cv2.putText(out, msg, (mx + 4, my - mbl - 2),
                            mfont, mfs, (0, 255, 100), mth, cv2.LINE_AA)

        # 発生直後フラッシュ
        just = [ev for ev in active_rescues if frame_id - ev.frame_id < 6]
        if just:
            fa   = 0.2 * (1 - (frame_id - just[0].frame_id) / 6)
            ovl3 = out.copy()
            cv2.rectangle(ovl3, (0, 0), (w, h), (0, 255, 120), cv2.FILLED)
            cv2.addWeighted(ovl3, fa, out, 1 - fa, 0, out)
            msg2  = f"RE-ID RESCUE  {just[0].old_id} -> {just[0].new_id}"
            cfont = cv2.FONT_HERSHEY_DUPLEX
            cfs   = 1.6 * sf;  cth = max(2, int(sf * 3))
            (cw, ch), _ = cv2.getTextSize(msg2, cfont, cfs, cth)
            cx = (w - cw) // 2;  cy = h // 2
            cv2.rectangle(out, (cx - 12, cy - ch - 12), (cx + cw + 12, cy + 12),
                          (0, 40, 0), cv2.FILLED)
            cv2.putText(out, msg2, (cx, cy), cfont, cfs, (0, 255, 100), cth, cv2.LINE_AA)

    # ── 上部バナー ─────────────────────────────────────────────────────────────
    tf    = cv2.FONT_HERSHEY_DUPLEX
    t_fs  = 1.4 * sf;  t_th = max(2, int(sf * 2.5))
    pp_label = {"none":"","abc":"+ABC","de":"+DE","abcde":"+ABCDE","rescue":"+RESCUE"}
    tag   = f"  {tracker_name.upper()}{pp_label.get(postproc,'')}  "
    (tw, th), tbl = cv2.getTextSize(tag, tf, t_fs, t_th)
    px = int(20 * sf);  py = int(10 * sf)
    bx1 = (w - tw) // 2 - px;  by1 = int(10 * sf)
    bx2 = (w + tw) // 2 + px;  by2 = by1 + th + tbl + py * 2
    ovl4 = out.copy()
    cv2.rectangle(ovl4, (bx1, by1), (bx2, by2), color, cv2.FILLED)
    cv2.addWeighted(ovl4, 0.85, out, 0.15, 0, out)
    cv2.putText(out, tag, ((w - tw) // 2, by1 + py + th),
                tf, t_fs, (0, 0, 0), t_th, cv2.LINE_AA)
    if postproc != "none":
        cv2.rectangle(out, (bx1, by1), (bx2, by2), accent, max(3, int(4 * sf)))

    # ── 左上パネル ─────────────────────────────────────────────────────────────
    sf_font = cv2.FONT_HERSHEY_DUPLEX
    sf_fs = 0.55 * sf;  sf_th = max(1, int(sf));  lh = int(26 * sf);  pd = int(12 * sf)
    stats = [
        f"Frame : {frame_id}",
        f"FPS   : {fps:.1f}",
        f"Infer : {infer_ms:.0f}ms",
        f"Active: {sum(1 for ft in ft_list if ft.state == TrackState.Confirmed)}",
        f"IDs   : {metrics['frags'] if metrics else '---'}",
    ]
    (pw2, _), _ = cv2.getTextSize("Infer : 9999ms", sf_font, sf_fs, sf_th)
    pw3 = pw2 + pd * 2;  ph = lh * len(stats) + pd * 2
    ovl5 = out.copy()
    cv2.rectangle(ovl5, (0, 0), (pw3, ph), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ovl5, 0.55, out, 0.45, 0, out)
    for i, text in enumerate(stats):
        cv2.putText(out, text, (pd, pd + lh * (i + 1) - int(4 * sf)),
                    sf_font, sf_fs, (220, 220, 220), sf_th, cv2.LINE_AA)

    # ── 右下スコアパネル ───────────────────────────────────────────────────────
    out = draw_score_panel(out, metrics, gt)
    return out


# ── トラッキング実行（none/de/abc/abcde 共通） ─────────────────────────────────
def run_combo(
    frames: list[np.ndarray],
    tracker_name: str,
    postproc: str,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    out_path: Path,
    src_fps: float,
    skip: int,
    gt: int,
) -> dict:
    liberal = postproc in ("abc", "abcde")
    STrack.reset_id_counter()
    tracker = make_tracker(tracker_name)
    gallery = ReIDGallery(
        max_gallery_size=500 if liberal else 200,
        similarity_thresh=0.60 if liberal else 0.72,
        max_embeddings_per_id=16 if liberal else 8,
    )

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps / skip, (w, h))

    all_tids:    set[int]       = set()
    track_birth: dict[int, int] = {}
    track_death: dict[int, int] = {}
    track_embs:  dict           = {}
    frame_data   = []
    rescue_count = 0
    rescue_sims: list[float] = []
    det_times:   list[float] = []
    active_cnts: list[int]   = []
    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0
    running_metrics = None

    for fid, frame in enumerate(frames, 1):
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        high_embs = reid_ext.extract_batch(frame, [d.bbox for d in high_raw]) \
            if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32)
        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        active    = tracker.update(high_dets, low_dets)
        active_ids = {t.track_id for t in active}
        all_tids.update(active_ids)

        confirmed = [t for t in active if t.state == TrackState.Confirmed]
        for t in confirmed:
            if t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, fid)
                track_embs.setdefault(t.track_id, []).append(t.reid_embedding)
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

        # ABC rescue
        if liberal:
            for t in active:
                if t.state == TrackState.Tentative and t.hits == 1 and t.reid_embedding is not None:
                    mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                    if mid is not None:
                        rescue_count += 1
                        rescue_sims.append(sim)
                        old = t.track_id
                        t.reassign_id(mid)
                        gallery.remove_track(old)

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            if not liberal:
                gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_cnts.append(len(active))
        fts = [FT(t.track_id, t.tlbr.copy(), t.state) for t in active]
        frame_data.append(fts)

        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt = 0;  fps_t0 = time.perf_counter()

        # 中間メトリクス（確定値ではない）
        if fid % 50 == 0 or fid == len(frames):
            running_metrics = compute_gt_score(
                all_tids, track_birth, track_death, active_cnts, fid, gt)

    # ── DE後処理 ─────────────────────────────────────────────────────────────
    id_map: dict[int, int] = {}
    if postproc in ("de", "abcde"):
        mean_embs = compute_mean_embeddings(track_embs)
        merged    = merge_by_appearance(mean_embs, similarity_thresh=0.72)
        reassigned = cluster_reassign(all_tids, merged)
        id_map    = compose_maps(merged, reassigned)

    # 最終メトリクス（DE適用後のID数で再計算）
    effective_ids = {id_map.get(tid, tid) for tid in all_tids}
    final_metrics = compute_gt_score(
        effective_ids, track_birth, track_death, active_cnts, len(frames), gt)
    final_metrics["rescue_count"] = rescue_count

    # ── 動画書き出し ─────────────────────────────────────────────────────────
    for fid, (frame, fts) in enumerate(zip(frames, frame_data), 1):
        vis = draw_frame(frame, fts, id_map, tracker_name, postproc,
                         fid, fps_disp, det_times[fid - 1] if fid - 1 < len(det_times) else 0,
                         final_metrics, gt)
        writer.write(vis)
    writer.release()
    return final_metrics


# ── rescue可視化コンボ ────────────────────────────────────────────────────────
def run_rescue_vis(
    frames: list[np.ndarray],
    detector: YOLOXDetector,
    out_path: Path,
    src_fps: float,
    skip: int,
    gt: int,
    device: str,
) -> dict:
    import torch
    ReidAutoBackend, WEIGHTS = _get_boxmot()
    REID_MODEL = "osnet_ibn_x1_0_msmt17.pt"

    class BoxmotExt:
        def __init__(self):
            self.feature_dim = 512
            self._b = ReidAutoBackend(
                weights=WEIGHTS / REID_MODEL,
                device=torch.device(device), half=False)
        def extract_batch(self, frame, bboxes):
            if not bboxes: return np.zeros((0, self.feature_dim), np.float32)
            xy = np.array([[b[0],b[1],b[2],b[3]] for b in bboxes], np.float32)
            f  = self._b.model.get_features(xy, frame)
            if f is None or (isinstance(f, np.ndarray) and f.size == 0):
                return np.zeros((len(bboxes), self.feature_dim), np.float32)
            f = np.asarray(f, np.float32)
            if f.ndim == 1: f = f[np.newaxis]
            self.feature_dim = f.shape[1]
            norms = np.linalg.norm(f, axis=1, keepdims=True)
            return f / np.maximum(norms, 1e-12)

    reid_ext = BoxmotExt()
    STrack.reset_id_counter()
    tracker = make_tracker("hybridsort_tuned")
    gallery = ReIDGallery(max_gallery_size=200, similarity_thresh=0.72, max_embeddings_per_id=8)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps / skip, (w, h))

    all_tids: set[int] = set()
    track_birth: dict = {};  track_death: dict = {}
    active_cnts: list[int] = []
    rescue_events: list[RescueEvent] = []
    active_rescues: list[RescueEvent] = []
    det_times: list[float] = []
    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0
    metrics = None

    for fid, frame in enumerate(frames, 1):
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        bboxes    = [d.bbox for d in high_raw]
        high_embs = reid_ext.extract_batch(frame, bboxes) if bboxes \
                    else np.zeros((0, reid_ext.feature_dim), np.float32)
        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        active    = tracker.update(high_dets, low_dets)
        active_ids = {t.track_id for t in active}
        all_tids.update(active_ids)

        for t in [x for x in active if x.state == TrackState.Confirmed]:
            if t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, fid)
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

        for t in active:
            if t.state == TrackState.Tentative and t.hits == 1 and t.reid_embedding is not None:
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    old = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old)
                    ev = RescueEvent(fid, old, mid, float(sim), t.tlbr.copy())
                    active_rescues.append(ev)
                    rescue_events.append(ev)

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_rescues = [ev for ev in active_rescues
                          if fid - ev.frame_id < RESCUE_DISPLAY_FRAMES]
        active_cnts.append(len(active))

        if fid % 50 == 0 or fid == len(frames):
            metrics = compute_gt_score(all_tids, track_birth, track_death,
                                       active_cnts, fid, gt)

        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt  = 0;  fps_t0 = time.perf_counter()

        fts = [FT(t.track_id, t.tlbr.copy(), t.state) for t in active]
        vis = draw_frame(frame, fts, {}, "hybridsort_tuned", "rescue",
                         fid, fps_disp, det_times[-1], metrics, gt, active_rescues)
        writer.write(vis)

    writer.release()
    m = compute_gt_score(all_tids, track_birth, track_death, active_cnts, len(frames), gt)
    m["rescue_count"] = len(rescue_events)
    return m


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test10.mp4")
    parser.add_argument("--out_dir",     default="outputs/test10_top20")
    parser.add_argument("--gt_persons",  type=int, default=10)
    parser.add_argument("--frames",      type=int, default=0)
    parser.add_argument("--skip",        type=int, default=2)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    # ── フレーム読み込み ───────────────────────────────────────────────────────
    print(f"\n動画読み込み: {args.video}")
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {ow}×{oh}  {src_fps:.1f}fps  ({total_src}フレーム)")

    frames: list[np.ndarray] = []
    fid = 0;  max_f = args.frames if args.frames > 0 else 10**9
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        if fid % args.skip != 0: continue
        frames.append(frame)
    cap.release()
    print(f"  → {len(frames)} フレーム (skip={args.skip})")

    # ── 検出器・Re-ID 読み込み ────────────────────────────────────────────────
    print(f"\nYOLOX ({args.yolox_model}) 読み込み中…")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10, nms_thresh=0.45)
    detector.load_model()

    print("Re-ID (osnet_x0_25 torchreid) 読み込み中…")
    reid_ext = FastReIDExtractor(model_path="osnet_x0_25", device=args.device, use_onnx=False)
    reid_ext.load_model()
    print("  → 準備完了\n")

    # ── 20コンボ実行 ───────────────────────────────────────────────────────────
    n = len(COMBOS)
    print(f"20コンボ実行開始 (GT={args.gt_persons}人)\n")
    print(f"{'No':<4} {'Combo':<40} {'GTscore':>8} {'frags':>6} {'cov':>6} {'rescue':>7} {'時間':>6}")
    print("─" * 80)

    all_results: list[dict] = []

    for idx, (tracker_name, postproc) in enumerate(COMBOS, 1):
        label = f"{tracker_name}+{postproc}"
        out_path = out_dir / f"{stem}_{tracker_name}_{postproc}.mp4"
        print(f"[{idx:>2}/{n}] {label:<38}", end="  ", flush=True)
        t0 = time.perf_counter()

        try:
            if postproc == "rescue":
                m = run_rescue_vis(frames, detector, out_path,
                                   src_fps, args.skip, args.gt_persons, args.device)
            else:
                m = run_combo(frames, tracker_name, postproc, detector, reid_ext,
                              out_path, src_fps, args.skip, args.gt_persons)

            elapsed = time.perf_counter() - t0
            m.update({"tracker": tracker_name, "postproc": postproc,
                       "combo": label, "total_s": elapsed})
            all_results.append(m)
            print(f"✓  {m['gt_score']:.3f}  {m['frags']:>5}  "
                  f"{m['coverage']:.3f}  {m.get('rescue_count',0):>6}  {elapsed:>5.0f}s")
        except Exception as e:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"✗  ERROR: {e}  ({elapsed:.0f}s)")
            traceback.print_exc()

    # ── ランキング表示 ────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  最終ランキング (GTscore 降順)\n")
    sorted_r = sorted(all_results, key=lambda x: -x.get("gt_score", 0))
    print(f"  {'Rank':<5} {'Combo':<40} │ {'GTscore':>7} │ {'IDprec':>6} │ "
          f"{'Cover':>5} │ {'Frags':>5} │ {'Rescue':>6}")
    print(f"  {'─'*80}")
    for rank, m in enumerate(sorted_r, 1):
        print(f"  #{rank:<4} {m['combo']:<40} │ {m['gt_score']:>7.3f} │ "
              f"{m['id_precision']:>6.3f} │ {m['coverage']:>5.3f} │ "
              f"{m['frags']:>5} │ {m.get('rescue_count',0):>6}")
    print(f"  {'─'*80}")

    json_path = out_dir / "eval_top20.json"
    with open(json_path, "w") as f:
        json.dump(sorted_r, f, indent=2, ensure_ascii=False)
    print(f"\n動画 → {out_dir}/  ({len(all_results)}本)")
    print(f"JSON → {json_path}")


if __name__ == "__main__":
    main()
