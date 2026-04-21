"""
画質を変えてトラッキング精度への影響を検証する。

画質バリエーション（6種）:
  original  : オリジナル（ベースライン）
  blur_mild : 軽微ぼかし (Gaussian σ=2)
  blur_heavy: 強ぼかし   (Gaussian σ=5)
  jpeg_75   : JPEG圧縮 quality=75
  jpeg_30   : JPEG圧縮 quality=30（かなり荒い）
  half_res  : 解像度1/2にダウンスケール後アップスケール
  quarter_res: 解像度1/4にダウンスケール後アップスケール

設定: ReID-AGGR（前回最良）で固定、画質だけ変える

Usage:
    .venv/bin/python scripts/run_quality_compare.py \\
        --video   assets/test10.mp4 \\
        --out_dir outputs/test10_quality \\
        --gt_persons 10 \\
        --skip 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import Detection as TrackDetection
from tracking.hybridsort import HybridSORTTracker
from tracking.track import STrack, TrackState
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery
from tracking.postprocess import compute_mean_embeddings, merge_by_appearance

# ── 画質バリエーション定義 ────────────────────────────────────────────────────
QUALITY_PROFILES: list[dict] = [
    dict(key="original",    label="ORIGINAL",     desc="オリジナル画質",
         color=(200, 200, 200)),
    dict(key="blur_mild",   label="BLUR-MILD",    desc="軽ぼかし (σ=2)",
         color=(0, 200, 255)),
    dict(key="blur_heavy",  label="BLUR-HEAVY",   desc="強ぼかし (σ=5)",
         color=(0, 100, 255)),
    dict(key="jpeg_75",     label="JPEG-75",      desc="JPEG圧縮 quality=75",
         color=(0, 200, 100)),
    dict(key="jpeg_30",     label="JPEG-30",      desc="JPEG圧縮 quality=30",
         color=(0, 100, 50)),
    dict(key="half_res",    label="HALF-RES",     desc="解像度1/2 (→復元)",
         color=(200, 100, 0)),
    dict(key="quarter_res", label="QUARTER-RES",  desc="解像度1/4 (→復元)",
         color=(255, 50, 0)),
]

# ReID-AGGRパラメータ固定
TRACKER_CFG = dict(
    track_thresh     = 0.50,
    track_buffer     = 180,
    match_thresh     = 0.90,
    min_hits         = 3,
    iou_thresh_stage2= 0.40,
    iou_weight       = 0.40,
    height_weight    = 0.20,
    shape_weight     = 0.10,
    reid_weight      = 0.50,
)
GALLERY_CFG = dict(
    max_gallery_size     = 300,
    similarity_thresh    = 0.55,
    max_embeddings_per_id= 16,
)
DE_SIM_THRESH = 0.55

_STATE_COLORS = {
    TrackState.Tentative: (200, 200,   0),
    TrackState.Confirmed: (  0, 230,   0),
    TrackState.Lost:      (  0, 120, 255),
}


class FT(NamedTuple):
    track_id: int
    tlbr:     np.ndarray
    state:    TrackState


# ── 画質変換 ──────────────────────────────────────────────────────────────────
def degrade_frame(frame: np.ndarray, key: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if key == "original":
        return frame
    if key == "blur_mild":
        return cv2.GaussianBlur(frame, (0, 0), 2)
    if key == "blur_heavy":
        return cv2.GaussianBlur(frame, (0, 0), 5)
    if key == "jpeg_75":
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if key == "jpeg_30":
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if key == "half_res":
        small = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if key == "quarter_res":
        small = cv2.resize(frame, (w//4, h//4), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


# ── 評価指標 ──────────────────────────────────────────────────────────────────
def compute_metrics(all_tids, track_birth, track_death, active_cnts,
                    rescue_count, n_frames, gt) -> dict:
    long_lived = sum(1 for t in all_tids
                     if (track_death.get(t,0) - track_birth.get(t,0)) >= n_frames * 0.5)
    avg_active = float(np.mean(active_cnts)) if active_cnts else 0.0
    id_prec    = min(gt / max(len(all_tids), 1), 1.0)
    coverage   = min(long_lived / gt, 1.0)
    od_rate    = avg_active / gt
    under_p    = min(1.0 / max(od_rate, 0.01), 1.0)
    comps      = [id_prec, coverage, under_p]
    denom      = sum(1.0 / max(c, 1e-6) for c in comps)
    gt_score   = len(comps) / denom if denom > 0 else 0.0
    lifetimes  = [track_death.get(t,0) - track_birth.get(t,0) for t in all_tids]
    avg_life   = float(np.mean(lifetimes)) if lifetimes else 0.0
    id_surv    = long_lived / gt if gt > 0 else 0.0
    return dict(gt_score=gt_score, id_precision=id_prec, coverage=coverage,
                od_rate=od_rate, frags=len(all_tids), long_lived=long_lived,
                avg_id_life=avg_life, id_survival=id_surv,
                rescue_count=rescue_count, avg_active=avg_active)


# ── 描画 ──────────────────────────────────────────────────────────────────────
def draw_frame(out: np.ndarray, ft_list: list, qcfg: dict,
               fid: int, fps: float, det_ms: float, metrics: dict | None,
               gt: int, n_frames: int, degraded: np.ndarray) -> np.ndarray:
    h, w  = out.shape[:2]
    sf    = h / 720
    color = qcfg["color"]

    # ── 左半分に劣化画像プレビュー（小） ─────────────────────────────────────
    prev_h = int(h * 0.22)
    prev_w = int(prev_h * w / h)
    prev   = cv2.resize(degraded, (prev_w, prev_h))
    px, py = int(8*sf), h - prev_h - int(8*sf)
    # 枠
    cv2.rectangle(out, (px-2, py-2), (px+prev_w+2, py+prev_h+2), color, 2)
    out[py:py+prev_h, px:px+prev_w] = prev
    cv2.putText(out, "INPUT", (px+4, py+int(16*sf)),
                cv2.FONT_HERSHEY_DUPLEX, 0.45*sf, color, max(1,int(sf)), cv2.LINE_AA)

    # ── バウンディングボックス ─────────────────────────────────────────────────
    for ft in ft_list:
        x1, y1, x2, y2 = ft.tlbr.astype(int)
        sc = _STATE_COLORS.get(ft.state, (180, 180, 180))
        cv2.rectangle(out, (x1,y1), (x2,y2), sc, 3)
        bh = max(y2-y1, 1)
        fs = max(0.5, min(2.0, bh/160))
        th = max(1, int(fs*1.5))
        lbl = f" ID {ft.track_id} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw,tth),bl = cv2.getTextSize(lbl, font, fs, th)
        by2 = max(tth+bl+8, y1);  by1b = max(0, y1-tth-bl-8)
        cv2.rectangle(out, (x1,by1b), (x1+tw,by2), sc, cv2.FILLED)
        cv2.putText(out, lbl, (x1, by2-bl-3), font, fs, (0,0,0), th, cv2.LINE_AA)

    # ── 上部バナー ─────────────────────────────────────────────────────────────
    tf   = cv2.FONT_HERSHEY_DUPLEX
    t_fs = 1.3*sf;  t_th = max(2, int(sf*2.5))
    tag  = f"  {qcfg['label']}  "
    sub  = f"  {qcfg['desc']}  "
    (tw1,th1),_ = cv2.getTextSize(tag, tf, t_fs, t_th)
    (tw2,th2),_ = cv2.getTextSize(sub, tf, t_fs*0.7, t_th-1)
    bw  = max(tw1,tw2) + int(40*sf)
    bh2 = th1 + th2 + int(28*sf)
    bx1 = (w-bw)//2;  by1 = int(8*sf)
    ovl = out.copy()
    cv2.rectangle(ovl, (bx1,by1), (bx1+bw, by1+bh2), color, cv2.FILLED)
    cv2.addWeighted(ovl, 0.88, out, 0.12, 0, out)
    pad = int(10*sf)
    cv2.putText(out, tag, ((w-tw1)//2, by1+pad+th1), tf, t_fs, (0,0,0), t_th, cv2.LINE_AA)
    cv2.putText(out, sub, ((w-tw2)//2, by1+pad+th1+int(4*sf)+th2),
                tf, t_fs*0.7, (0,0,0), max(1,t_th-1), cv2.LINE_AA)

    # ── 右下スコアパネル ───────────────────────────────────────────────────────
    if metrics:
        score = metrics["gt_score"]
        frags = metrics["frags"]
        alife = metrics["avg_id_life"]
        surv  = metrics["id_survival"]
        cov   = metrics["coverage"]
        det_c = metrics.get("avg_det_count", 0)

        def sc2(v, g, m):
            return (0,230,0) if v>=g else (0,220,220) if v>=m else (80,80,255)

        lines = [
            ("GTscore",  f"{score:.3f}", sc2(score,0.7,0.4), 1.15),
            ("Coverage", f"{cov:.3f}",   sc2(cov,0.8,0.5),   0.80),
            ("frags",    f"{frags}  (GT={gt})", sc2(gt/max(frags,1),0.7,0.4), 0.80),
            ("avgLife",  f"{alife:.0f}f", (200,200,200),      0.75),
            ("IDsurv",   f"{surv:.3f}",  sc2(surv,0.8,0.5),  0.75),
            ("Detects",  f"{det_c:.1f}/f",(180,180,180),      0.70),
            ("rescue",   f"{metrics['rescue_count']}回",(180,180,180), 0.70),
        ]

        ef     = cv2.FONT_HERSHEY_DUPLEX
        bfs    = 0.85*sf;  bth = max(1,int(sf*1.5))
        lhe    = int(34*sf);  pde = int(12*sf)
        max_lw = 0
        for lbl,val,col,fmul in lines:
            (lw,_),_ = cv2.getTextSize(f"{lbl} : {val}", ef, bfs*fmul, bth)
            max_lw = max(max_lw, lw)
        pw4 = max_lw + pde*3
        ph4 = lhe*len(lines) + pde*2
        ex1 = w - pw4 - int(8*sf);  ey1 = h - ph4 - int(8*sf)
        ovl2 = out.copy()
        cv2.rectangle(ovl2, (ex1-pde, ey1-pde), (w-int(4*sf), h-int(4*sf)),
                      (10,10,10), cv2.FILLED)
        cv2.addWeighted(ovl2, 0.78, out, 0.22, 0, out)
        cv2.rectangle(out, (ex1-pde, ey1-pde), (ex1-pde+int(5*sf), h-int(4*sf)),
                      color, cv2.FILLED)
        for i, (lbl,val,col,fmul) in enumerate(lines):
            fs_i = bfs*fmul;  th_i = max(1,int(fs_i*1.8))
            cv2.putText(out, f"{lbl} : {val}", (ex1, ey1+pde+lhe*(i+1)),
                        ef, fs_i, col, th_i, cv2.LINE_AA)

    # ── フレーム情報（右上） ──────────────────────────────────────────────────
    info = f"Frame {fid}/{n_frames}  {fps:.1f}fps  det:{det_ms:.0f}ms"
    (iw,ih),_ = cv2.getTextSize(info, cv2.FONT_HERSHEY_DUPLEX, 0.5*sf, max(1,int(sf)))
    cv2.putText(out, info, (w-iw-int(10*sf), int(20*sf)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5*sf, (200,200,200), max(1,int(sf)), cv2.LINE_AA)
    return out


# ── 1画質プロファイル実行 ─────────────────────────────────────────────────────
def run_quality(
    frames:   list[np.ndarray],
    qcfg:     dict,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    out_path: Path,
    src_fps:  float,
    skip:     int,
    gt:       int,
) -> dict:
    key = qcfg["key"]

    STrack.reset_id_counter()
    tracker = HybridSORTTracker(**TRACKER_CFG)
    gallery = ReIDGallery(**GALLERY_CFG)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps/skip, (w,h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps/skip, (w,h))

    all_tids:    set[int]      = set()
    track_birth: dict[int,int] = {}
    track_death: dict[int,int] = {}
    track_embs:  dict          = {}
    active_cnts: list[int]     = []
    det_counts:  list[int]     = []
    det_times:   list[float]   = []
    rescue_count = 0
    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0
    metrics = None

    for fid, orig_frame in enumerate(frames, 1):
        # 画質劣化を適用（検出・Re-ID は劣化画像で行う）
        deg_frame = degrade_frame(orig_frame, key)

        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(deg_frame)
        det_times.append((time.perf_counter() - t0) * 1000)
        det_counts.append(len(high_raw))

        high_embs = reid_ext.extract_batch(deg_frame, [d.bbox for d in high_raw]) \
            if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32)
        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        active     = tracker.update(high_dets, low_dets)
        active_ids = {t.track_id for t in active}
        all_tids.update(active_ids)

        for t in [x for x in active if x.state == TrackState.Confirmed]:
            if t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, fid)
                track_embs.setdefault(t.track_id, []).append(t.reid_embedding)
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

        for t in active:
            if t.state == TrackState.Tentative and t.hits == 1 and t.reid_embedding is not None:
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    rescue_count += 1
                    old = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old)

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_cnts.append(len(active))
        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt  = 0;  fps_t0 = time.perf_counter()

        if fid % 30 == 0 or fid == len(frames):
            metrics = compute_metrics(all_tids, track_birth, track_death,
                                      active_cnts, rescue_count, fid, gt)
            metrics["avg_det_count"] = float(np.mean(det_counts))

        fts = [FT(t.track_id, t.tlbr.copy(), t.state) for t in active]
        # 描画はオリジナル画像に行い、小プレビューで劣化画像を表示
        vis = draw_frame(orig_frame.copy(), fts, qcfg, fid, fps_disp,
                         det_times[-1], metrics, gt, len(frames), deg_frame)
        writer.write(vis)

    writer.release()

    # DE後処理
    mean_embs = compute_mean_embeddings(track_embs)
    merged    = merge_by_appearance(track_birth, track_death, mean_embs,
                                    sim_thresh=DE_SIM_THRESH)
    effective = {merged.get(t,t) for t in all_tids}

    final = compute_metrics(effective, track_birth, track_death,
                            active_cnts, rescue_count, len(frames), gt)
    final["avg_det_count"] = float(np.mean(det_counts)) if det_counts else 0.0
    final["avg_det_ms"]    = float(np.mean(det_times)) if det_times else 0.0
    final["key"]   = key
    final["label"] = qcfg["label"]
    final["desc"]  = qcfg["desc"]
    return final


# ── サマリー動画（3×3グリッド風、7本→余白あり） ──────────────────────────────
def make_summary(frames_count: int, out_dir: Path, stem: str,
                 src_fps: float, skip: int, results: list[dict]):
    n = len(QUALITY_PROFILES)
    cols = 3;  rows = (n + cols - 1) // cols

    # 最初の動画からサイズ取得
    first = out_dir / f"{stem}_{QUALITY_PROFILES[0]['key']}.mp4"
    cap0  = cv2.VideoCapture(str(first))
    fw    = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    gh = fh // rows;  gw = fw // cols
    out_w = gw * cols;  out_h = gh * rows

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_path = out_dir / f"{stem}_summary.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps/skip, (out_w, out_h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps/skip, (out_w, out_h))

    caps = []
    for qcfg in QUALITY_PROFILES:
        p = out_dir / f"{stem}_{qcfg['key']}.mp4"
        caps.append(cv2.VideoCapture(str(p)) if p.exists() else None)

    # スコア dict
    score_map = {r["key"]: r["gt_score"] for r in results}

    while True:
        grid = np.zeros((out_h, out_w, 3), np.uint8)
        any_frame = False
        for i, (qcfg, cap) in enumerate(zip(QUALITY_PROFILES, caps)):
            row, col = divmod(i, cols)
            if cap is None:
                cell = np.zeros((gh, gw, 3), np.uint8)
            else:
                ret, cell = cap.read()
                if ret:
                    any_frame = True
                    cell = cv2.resize(cell, (gw, gh))
                else:
                    cell = np.zeros((gh, gw, 3), np.uint8)

            # スコアオーバーレイ
            s = score_map.get(qcfg["key"], 0)
            scol = (0,230,0) if s>0.7 else (0,220,220) if s>0.4 else (80,80,255)
            sf2  = gh / 720
            ef   = cv2.FONT_HERSHEY_DUPLEX
            txt  = f"{qcfg['label']} GTscore:{s:.3f}"
            fs2  = 0.7*sf2;  th2 = max(1,int(sf2*1.2))
            (lw2,lh2),_ = cv2.getTextSize(txt, ef, fs2, th2)
            cv2.rectangle(cell, (2, 2), (lw2+10, lh2+10), (10,10,10), cv2.FILLED)
            cv2.putText(cell, txt, (6, lh2+4), ef, fs2, scol, th2, cv2.LINE_AA)

            grid[row*gh:(row+1)*gh, col*gw:(col+1)*gw] = cell

        if not any_frame:
            break
        writer.write(grid)

    for cap in caps:
        if cap: cap.release()
    writer.release()
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test10.mp4")
    parser.add_argument("--out_dir",     default="outputs/test10_quality")
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
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {ow}×{oh}  {src_fps:.1f}fps  ({total}フレーム)")

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

    # ── 検出器・Re-ID ─────────────────────────────────────────────────────────
    print(f"\nYOLOX ({args.yolox_model}) 読み込み中…")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10, nms_thresh=0.45)
    detector.load_model()
    print("Re-ID (osnet_x0_25) 読み込み中…")
    reid_ext = FastReIDExtractor(model_path="osnet_x0_25", device=args.device, use_onnx=False)
    reid_ext.load_model()
    print("  → 準備完了\n")

    # ── 7画質プロファイル実行 ─────────────────────────────────────────────────
    n = len(QUALITY_PROFILES)
    sep = "─" * 100
    print(f"画質比較実験 (GT={args.gt_persons}人,  設定=ReID-AGGR固定)\n")
    print(f"{'No':>3} {'画質':^14} {'GTscore':>8} {'frags':>6} {'avgLife':>8} "
          f"{'IDsurv':>7} {'Cover':>6} {'Detects/f':>10} {'Det ms':>7} {'秒':>5}")
    print(sep)

    all_results: list[dict] = []

    for idx, qcfg in enumerate(QUALITY_PROFILES, 1):
        out_path = out_dir / f"{stem}_{qcfg['key']}.mp4"
        print(f"[{idx}/{n}] {qcfg['label']:<14} {qcfg['desc']:<26}", end="  ", flush=True)
        t0 = time.perf_counter()
        try:
            m = run_quality(frames, qcfg, detector, reid_ext,
                            out_path, src_fps, args.skip, args.gt_persons)
            elapsed = time.perf_counter() - t0
            m["total_s"] = elapsed
            all_results.append(m)
            print(f"✓  {m['gt_score']:.3f}  {m['frags']:>5}  "
                  f"{m['avg_id_life']:>7.1f}f  {m['id_survival']:>6.3f}  "
                  f"{m['coverage']:>5.3f}  {m['avg_det_count']:>9.1f}  "
                  f"{m['avg_det_ms']:>6.1f}ms  {elapsed:>4.0f}s")
        except Exception as e:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"✗  ERROR: {e}  ({elapsed:.0f}s)")
            traceback.print_exc()

    # ── サマリー動画 ──────────────────────────────────────────────────────────
    print(f"\nサマリー動画（グリッド）生成中…")
    try:
        sp = make_summary(len(frames), out_dir, stem, src_fps, args.skip, all_results)
        print(f"  → {sp}")
    except Exception as e:
        print(f"  エラー: {e}")

    # ── 最終比較表 ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    best_score = max(r["gt_score"] for r in all_results) if all_results else 0
    baseline   = next((r for r in all_results if r["key"] == "original"), None)

    print(f"\n  画質劣化による精度変化まとめ (ベースライン=ORIGINAL)\n")
    print(f"  {'画質':<14} │ {'GTscore':>8} │ {'差分':>7} │ {'frags':>5} │ "
          f"{'avgLife':>7} │ {'Cover':>5} │ {'Detect/f':>8}")
    print(f"  {'─'*80}")
    for m in all_results:
        diff  = m["gt_score"] - (baseline["gt_score"] if baseline else 0)
        best  = "★" if m["gt_score"] >= best_score else " "
        dcol  = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
        print(f"  {best}{m['label']:<13} │ {m['gt_score']:>8.3f} │ {dcol:>7} │ "
              f"{m['frags']:>5} │ {m['avg_id_life']:>6.1f}f │ "
              f"{m['coverage']:>5.3f} │ {m['avg_det_count']:>8.1f}")
    print(f"  {'─'*80}")

    print("""
  指標説明:
    GTscore  : 総合スコア（高いほど良い）
    frags    : 総ID生成数（少ないほど同一IDを維持できた）
    avgLife  : 1IDの平均寿命（長いほど安定）
    Cover    : GT全員を長期追跡できたか
    Detect/f : 1フレームあたりの平均検出数（画質が下がると減る）
""")

    json_path = out_dir / "eval_quality.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  動画  → {out_dir}/  ({len(all_results)}本 + サマリー1本)")
    print(f"  JSON  → {json_path}")


if __name__ == "__main__":
    main()
