"""
hybridsort_tuned + osnet_ibn_x1_0_msmt17 で tracking し、
Re-ID リカバリー発生時に動画内へ視覚的に通知する。

Usage:
    .venv/bin/python scripts/run_reid_rescue_vis.py \
        --video   assets/test3.mp4 \
        --out_dir outputs/reid_rescue_vis \
        --gt_persons 6 \
        --skip 2
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── lazy boxmot import (NumPy ABI 回避) ───────────────────────────────────────
def _get_boxmot():
    import torch  # noqa
    import boxmot  # noqa
    from boxmot.reid.core.auto_backend import ReidAutoBackend
    from boxmot.utils import WEIGHTS
    return ReidAutoBackend, WEIGHTS


from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import Detection as TrackDetection
from tracking.hybridsort import HybridSORTTracker
from tracking.track import STrack, TrackState
from reid.gallery import ReIDGallery

REID_MODEL   = "osnet_ibn_x1_0_msmt17.pt"
TRACKER_NAME = "hybridsort_tuned"
BANNER_COLOR = (0, 100, 200)   # osnet_ibn 系の青

_STATE_COLORS = {
    TrackState.Tentative: (200, 200,   0),
    TrackState.Confirmed: (  0, 230,   0),
    TrackState.Lost:      (  0, 120, 255),
}

# リカバリー通知を何フレーム表示し続けるか
RESCUE_DISPLAY_FRAMES = 60   # ≒2秒（30fps換算）


# ── Re-ID 抽出器 ──────────────────────────────────────────────────────────────
class BoxmotReIDExtractor:
    def __init__(self, device: str = "cpu"):
        self.device_str = device
        self._backend   = None
        self.feature_dim = 512

    def load_model(self) -> None:
        import torch
        ReidAutoBackend, WEIGHTS = _get_boxmot()
        weight_path = WEIGHTS / REID_MODEL
        self._backend = ReidAutoBackend(
            weights=weight_path, device=torch.device(self.device_str), half=False
        )
        print(f"  Re-ID モデル読み込み完了: {REID_MODEL}")

    def extract_batch(self, frame: np.ndarray, bboxes: list) -> np.ndarray:
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
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)


# ── tracker factory ───────────────────────────────────────────────────────────
def make_tracker():
    return HybridSORTTracker(
        track_thresh=0.50, track_buffer=180, match_thresh=0.90, min_hits=3,
        iou_thresh_stage2=0.45, iou_weight=0.40, height_weight=0.20,
        shape_weight=0.10, reid_weight=0.30,
    )


# ── リカバリーイベント ─────────────────────────────────────────────────────────
@dataclass
class RescueEvent:
    frame_id:  int
    old_id:    int
    new_id:    int
    sim:       float
    tlbr:      np.ndarray   # バウンディングボックス位置


# ── 描画 ───────────────────────────────────────────────────────────────────────
def draw_frame(
    frame:         np.ndarray,
    ft_list:       list,
    fid:           int,
    fps_disp:      float,
    infer_ms:      float,
    total_ids:     int,
    rescue_total:  int,
    active_rescues: list[RescueEvent],   # まだ表示中のイベント
) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()
    sf   = h / 720

    # ── バウンディングボックス ────────────────────────────────────────────────
    for ft in ft_list:
        x1, y1, x2, y2 = ft.tlbr.astype(int)
        sc = _STATE_COLORS.get(ft.state, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), sc, 3)
        box_h = max(y2 - y1, 1)
        fs = max(0.7, min(2.0, box_h / 220))
        th = max(1, int(fs * 1.5))
        lbl  = f" ID {ft.track_id} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, tth), bl = cv2.getTextSize(lbl, font, fs, th)
        by2 = max(tth + bl + 8, y1);  by1 = max(0, y1 - tth - bl - 8)
        cv2.rectangle(out, (x1, by1), (x1 + tw, by2), sc, cv2.FILLED)
        cv2.putText(out, lbl, (x1, by2 - bl - 3), font, fs, (0, 0, 0), th, cv2.LINE_AA)

    # ── リカバリー発生：バウンディングボックス強調 ────────────────────────────
    for ev in active_rescues:
        age = fid - ev.frame_id   # 0 = 発生フレーム
        alpha = max(0.0, 1.0 - age / RESCUE_DISPLAY_FRAMES)  # フェードアウト

        x1, y1, x2, y2 = ev.tlbr.astype(int)

        # 発光枠（太い黄緑）
        thickness = max(4, int(12 * sf * alpha))
        ovl = out.copy()
        cv2.rectangle(ovl, (x1, y1), (x2, y2), (0, 255, 100), thickness)
        cv2.addWeighted(ovl, alpha, out, 1 - alpha, 0, out)

        # "RECOVERED" ラベル（ボックス内上部）
        if age < RESCUE_DISPLAY_FRAMES // 2:
            msg   = f"RECOVERED  {ev.old_id} -> {ev.new_id}  sim={ev.sim:.2f}"
            mfont = cv2.FONT_HERSHEY_DUPLEX
            mfs   = max(0.55, min(1.2, (y2 - y1) / 200)) * sf
            mth   = max(1, int(mfs * 1.8))
            (mw, mh), mbl = cv2.getTextSize(msg, mfont, mfs, mth)
            mx = max(0, x1)
            my = max(mh + mbl + 4, y1 - 4)
            bg_alpha = 0.75 * alpha
            ovl2 = out.copy()
            cv2.rectangle(ovl2, (mx, my - mh - mbl - 4), (mx + mw + 8, my + 4),
                          (0, 60, 0), cv2.FILLED)
            cv2.addWeighted(ovl2, bg_alpha, out, 1 - bg_alpha, 0, out)
            cv2.putText(out, msg, (mx + 4, my - mbl - 2),
                        mfont, mfs, (0, 255, 100), mth, cv2.LINE_AA)

    # ── 画面中央フラッシュ（発生直後のみ） ───────────────────────────────────
    just_now = [ev for ev in active_rescues if fid - ev.frame_id < 8]
    if just_now:
        flash_alpha = 0.25 * (1 - (fid - just_now[0].frame_id) / 8)
        ovl3 = out.copy()
        cv2.rectangle(ovl3, (0, 0), (w, h), (0, 255, 120), cv2.FILLED)
        cv2.addWeighted(ovl3, flash_alpha, out, 1 - flash_alpha, 0, out)

        # 画面中央に大きなテキスト
        n = len(just_now)
        msg2  = f"RE-ID RESCUE  x{n}" if n > 1 else \
                f"RE-ID RESCUE  {just_now[0].old_id} -> {just_now[0].new_id}"
        cfs   = 1.8 * sf;  cth = max(2, int(sf * 3))
        cfont = cv2.FONT_HERSHEY_DUPLEX
        (cw, ch), cbl = cv2.getTextSize(msg2, cfont, cfs, cth)
        cx = (w - cw) // 2;  cy = h // 2
        cv2.rectangle(out, (cx - 12, cy - ch - 12), (cx + cw + 12, cy + cbl + 12),
                      (0, 40, 0), cv2.FILLED)
        cv2.putText(out, msg2, (cx, cy), cfont, cfs, (0, 255, 100), cth, cv2.LINE_AA)

    # ── 上部バナー（モデル名） ─────────────────────────────────────────────────
    tf    = cv2.FONT_HERSHEY_DUPLEX
    t_fs  = 1.1 * sf;  t_th = max(2, int(sf * 2))
    line1 = f"  {TRACKER_NAME.upper()}  "
    line2 = f"  Re-ID: {REID_MODEL.replace('.pt','')}  "
    (w1, h1), bl1 = cv2.getTextSize(line1, tf, t_fs * 0.85, t_th)
    (w2, h2), bl2 = cv2.getTextSize(line2, tf, t_fs, t_th)
    bw  = max(w1, w2) + int(40 * sf)
    bh  = h1 + h2 + bl1 + bl2 + int(24 * sf)
    bx1 = (w - bw) // 2;  by1_b = int(8 * sf)
    ovl4 = out.copy()
    cv2.rectangle(ovl4, (bx1, by1_b), (bx1 + bw, by1_b + bh), BANNER_COLOR, cv2.FILLED)
    cv2.addWeighted(ovl4, 0.85, out, 0.15, 0, out)
    pad = int(10 * sf)
    cv2.putText(out, line1, ((w - w1) // 2, by1_b + pad + h1),
                tf, t_fs * 0.85, (255, 255, 255), t_th, cv2.LINE_AA)
    cv2.putText(out, line2, ((w - w2) // 2, by1_b + pad + h1 + bl1 + h2 + int(6 * sf)),
                tf, t_fs, (255, 255, 255), t_th, cv2.LINE_AA)

    # ── 左下ステータスパネル ───────────────────────────────────────────────────
    sf_font = cv2.FONT_HERSHEY_DUPLEX
    sf_fs   = 0.58 * sf;  sf_th = max(1, int(sf));  lh = int(28 * sf);  pd = int(12 * sf)
    stats = [
        f"Frame   : {fid}",
        f"FPS     : {fps_disp:.1f}",
        f"Infer   : {infer_ms:.0f} ms",
        f"Active  : {sum(1 for ft in ft_list if ft.state == TrackState.Confirmed)}",
        f"Total IDs: {total_ids}",
        f"Rescue  : {rescue_total}",
    ]
    (pw, _), _ = cv2.getTextSize("Total IDs: 999", sf_font, sf_fs, sf_th)
    pw2 = pw + pd * 2;  ph = lh * len(stats) + pd * 2
    ovl5 = out.copy()
    cv2.rectangle(ovl5, (0, 0), (pw2, ph), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ovl5, 0.60, out, 0.40, 0, out)
    for i, text in enumerate(stats):
        cv2.putText(out, text, (pd, pd + lh * (i + 1) - int(4 * sf)),
                    sf_font, sf_fs, (220, 220, 220), sf_th, cv2.LINE_AA)
    return out


# ── メイン処理 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test3.mp4")
    parser.add_argument("--out_dir",     default="outputs/reid_rescue_vis")
    parser.add_argument("--gt_persons",  type=int, default=6)
    parser.add_argument("--frames",      type=int, default=0)
    parser.add_argument("--skip",        type=int, default=2)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── フレーム読み込み ───────────────────────────────────────────────────────
    print(f"\n動画読み込み: {args.video}")
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"動画を開けません: {args.video}"
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {orig_w}×{orig_h}  {src_fps:.1f}fps  ({total_src}フレーム)")

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
    print(f"  → {len(frames)} フレーム読み込み完了 (skip={args.skip})")

    # ── 検出器・Re-ID 読み込み ─────────────────────────────────────────────────
    print(f"\nYOLOX ({args.yolox_model}) 読み込み中…")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10,
                             nms_thresh=0.45)
    detector.load_model()

    print(f"Re-ID モデル読み込み中…")
    reid_ext = BoxmotReIDExtractor(device=args.device)
    reid_ext.load_model()

    # ── トラッキング実行 ───────────────────────────────────────────────────────
    STrack.reset_id_counter()
    tracker = make_tracker()
    gallery = ReIDGallery(max_gallery_size=200, similarity_thresh=0.72,
                          max_embeddings_per_id=8)

    out_path = out_dir / f"{stem}_{TRACKER_NAME}_{REID_MODEL.replace('.pt','')}_rescue.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(str(out_path), fourcc, src_fps / args.skip, (orig_w, orig_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / args.skip, (orig_w, orig_h))
    print(f"  VideoWriter opened: {writer.isOpened()}  codec={'avc1' if writer.isOpened() else 'mp4v'}")

    all_tids:    set[int]        = set()
    track_birth: dict[int, int]  = {}
    track_death: dict[int, int]  = {}
    rescue_total = 0
    rescue_sims: list[float]     = []
    rescue_log:  list[dict]      = []
    active_rescues: list[RescueEvent] = []  # 表示中のイベント

    det_times:   list[float]     = []
    active_cnts: list[int]       = []
    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0
    t_start = time.perf_counter()

    print(f"\n処理開始: {len(frames)} フレーム × {REID_MODEL}")
    print(f"出力先: {out_path}\n")

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
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

        # ── Re-ID リカバリー ──────────────────────────────────────────────────
        for t in active:
            if t.state == TrackState.Tentative and t.hits == 1 and t.reid_embedding is not None:
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    rescue_total += 1
                    rescue_sims.append(sim)
                    old_id = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old_id)

                    ev = RescueEvent(
                        frame_id=fid,
                        old_id=old_id,
                        new_id=mid,
                        sim=float(sim),
                        tlbr=t.tlbr.copy(),
                    )
                    active_rescues.append(ev)
                    rescue_log.append({
                        "frame": fid, "old_id": old_id, "new_id": mid, "sim": float(sim)
                    })
                    print(f"  [frame {fid:>5}] RESCUE  ID {old_id:>4} → {mid:>4}  sim={sim:.3f}")

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        # 期限切れのイベントを削除
        active_rescues = [ev for ev in active_rescues
                          if fid - ev.frame_id < RESCUE_DISPLAY_FRAMES]

        active_cnts.append(len(active))
        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt  = 0;  fps_t0 = time.perf_counter()

        ft_list = [type('FT', (), {
            'track_id': t.track_id,
            'tlbr':     t.tlbr.copy(),
            'state':    t.state
        })() for t in active]

        vis = draw_frame(frame, ft_list, fid, fps_disp, det_times[-1],
                         len(all_tids), rescue_total, active_rescues)
        writer.write(vis)

        if fid % 100 == 0:
            print(f"  [{fid:>5}/{len(frames)}] rescue={rescue_total}  IDs={len(all_tids)}")

    writer.release()
    total_s = time.perf_counter() - t_start

    # ── 評価 ──────────────────────────────────────────────────────────────────
    n_frames   = len(frames)
    long_lived = sum(1 for tid in all_tids
                     if (track_death.get(tid, 0) - track_birth.get(tid, 0)) >= n_frames * 0.5)
    avg_active = float(np.mean(active_cnts)) if active_cnts else 0.0

    gt = args.gt_persons
    id_prec  = min(gt / max(len(all_tids), 1), 1.0)
    coverage = min(long_lived / gt, 1.0)
    od_rate  = avg_active / gt
    under_p  = min(1.0 / max(od_rate, 0.01), 1.0)
    comps    = [id_prec, coverage, under_p]
    denom    = sum(1.0 / max(c, 1e-6) for c in comps)
    gt_score = len(comps) / denom if denom > 0 else 0.0

    result = {
        "reid_model":      REID_MODEL,
        "tracker":         TRACKER_NAME,
        "frames":          n_frames,
        "frags":           len(all_tids),
        "rescue_count":    rescue_total,
        "avg_rescue_sim":  float(np.mean(rescue_sims)) if rescue_sims else 0.0,
        "gt_persons":      gt,
        "id_precision":    id_prec,
        "coverage":        coverage,
        "od_rate":         od_rate,
        "gt_score":        gt_score,
        "total_s":         total_s,
        "rescue_log":      rescue_log,
    }

    json_path = out_dir / f"{stem}_{TRACKER_NAME}_{REID_MODEL.replace('.pt','')}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Re-ID: {REID_MODEL}
  GTscore   : {gt_score:.4f}
  IDprec    : {id_prec:.4f}
  Coverage  : {coverage:.4f}
  Total IDs : {len(all_tids)}  (GT={gt})
  Rescue    : {rescue_total} 回
  処理時間  : {total_s:.0f}秒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  動画  → {out_path}
  JSON  → {json_path}
""")


if __name__ == "__main__":
    main()
