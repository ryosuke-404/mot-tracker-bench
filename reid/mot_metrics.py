"""
MOT評価指標: IDF1 と HOTA の実装

使い方:
    evaluator = MOTEvaluator(iou_thresh=0.5)
    evaluator.update(gt_frame, pred_frame)   # フレームごとに呼ぶ
    results = evaluator.compute()

データ形式:
    frame: list of (track_id: int, bbox: np.ndarray[4])  # xyxy
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """(N,4) × (M,4) → (N,M) IoU行列"""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), np.float32)

    ax1, ay1, ax2, ay2 = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]
    bx1, by1, bx2, by2 = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]

    ix1 = np.maximum(ax1[:,None], bx1[None,:])
    iy1 = np.maximum(ay1[:,None], by1[None,:])
    ix2 = np.minimum(ax2[:,None], bx2[None,:])
    iy2 = np.minimum(ay2[:,None], by2[None,:])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return inter / np.maximum(union, 1e-6)


class MOTEvaluator:
    """
    フレームごとにGT/予測を受け取り、最後にIDF1とHOTAを計算する。

    GT が存在しない場合は疑似GT（別トラッカーの出力）を使う。
    """

    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh
        # フレームデータ: list of (gt_list, pred_list)
        # each list: [(tid, bbox_xyxy), ...]
        self._frames: list[tuple[list, list]] = []

    def update(
        self,
        gt_frame:   list[tuple[int, np.ndarray]],
        pred_frame: list[tuple[int, np.ndarray]],
    ) -> None:
        self._frames.append((gt_frame, pred_frame))

    def compute(self) -> dict:
        idf1  = self._compute_idf1()
        hota  = self._compute_hota()
        return {**idf1, **hota}

    # ── IDF1 ─────────────────────────────────────────────────────────────────
    def _compute_idf1(self) -> dict:
        """
        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)

        GT-trackとPred-trackを全フレーム共通でハンガリアンマッチング
        → 最も多くのフレームをカバーするペアを採用
        """
        # GT/Pred ごとのトラックID一覧
        gt_tids:   set[int] = set()
        pred_tids: set[int] = set()
        for gt_f, pred_f in self._frames:
            for tid, _ in gt_f:   gt_tids.add(tid)
            for tid, _ in pred_f: pred_tids.add(tid)

        gt_list   = sorted(gt_tids)
        pred_list = sorted(pred_tids)
        n_gt   = len(gt_list)
        n_pred = len(pred_list)

        if n_gt == 0 or n_pred == 0:
            return dict(idf1=0.0, idtp=0, idfp=0, idfn=0,
                        id_precision=0.0, id_recall=0.0)

        # マッチング行列: cost[i,j] = GT_i と Pred_j が同じフレームで
        #                              IoU >= thresh で検出された回数（共通フレーム数）
        gt_idx   = {t: i for i, t in enumerate(gt_list)}
        pred_idx = {t: i for i, t in enumerate(pred_list)}
        match_count = np.zeros((n_gt, n_pred), np.int32)

        for gt_f, pred_f in self._frames:
            if not gt_f or not pred_f:
                continue
            gt_ids  = [t for t, _ in gt_f]
            pred_ids= [t for t, _ in pred_f]
            gt_boxes  = np.array([b for _, b in gt_f],   np.float32)
            pred_boxes= np.array([b for _, b in pred_f], np.float32)

            iou = iou_matrix(gt_boxes, pred_boxes)
            ri, ci = linear_sum_assignment(-iou)
            for r, c in zip(ri, ci):
                if iou[r, c] >= self.iou_thresh:
                    gi = gt_idx[gt_ids[r]]
                    pi = pred_idx[pred_ids[c]]
                    match_count[gi, pi] += 1

        # ハンガリアンでGT↔Pred IDを1対1割り当て
        ri, ci = linear_sum_assignment(-match_count)
        idtp = int(sum(match_count[r, c] for r, c in zip(ri, ci)))

        # IDFN: GT フレーム総数 - IDTP
        gt_total   = sum(len(gf) for gf, _ in self._frames)
        pred_total = sum(len(pf) for _, pf in self._frames)
        idfn = gt_total   - idtp
        idfp = pred_total - idtp

        denom = 2 * idtp + idfp + idfn
        idf1 = 2 * idtp / max(denom, 1)
        id_precision = idtp / max(idtp + idfp, 1)
        id_recall    = idtp / max(idtp + idfn, 1)

        return dict(
            idf1         = float(idf1),
            idtp         = int(idtp),
            idfp         = int(idfp),
            idfn         = int(idfn),
            id_precision = float(id_precision),
            id_recall    = float(id_recall),
        )

    # ── HOTA ─────────────────────────────────────────────────────────────────
    def _compute_hota(self, alphas: list[float] | None = None) -> dict:
        """
        HOTA = mean_α sqrt( DetA(α) * AssA(α) )

        DetA: 検出精度（IoU閾値α でのTP/(TP+FP+FN)）
        AssA: 関連付け精度（同一GTに割り当てられた予測IDの一貫性）
        """
        if alphas is None:
            alphas = [0.05 * i for i in range(1, 20)]  # 0.05〜0.95

        hota_per_alpha: list[float] = []
        deta_per_alpha: list[float] = []
        assa_per_alpha: list[float] = []

        for alpha in alphas:
            det_a, ass_a = self._hota_at_alpha(alpha)
            hota_per_alpha.append(np.sqrt(det_a * ass_a))
            deta_per_alpha.append(det_a)
            assa_per_alpha.append(ass_a)

        return dict(
            hota = float(np.mean(hota_per_alpha)),
            deta = float(np.mean(deta_per_alpha)),
            assa = float(np.mean(assa_per_alpha)),
        )

    def _hota_at_alpha(self, alpha: float) -> tuple[float, float]:
        # フレームごとにTP/FP/FNを集計、さらに関連付けの一貫性も追跡
        tp_total = fp_total = fn_total = 0
        # 関連付け精度: matched (gt_id, pred_id) ペアごとに
        #   TPA[pair] = そのペアが一致したフレーム数
        #   FPA[pair] = gt_idに対して別pred_idが割り当てられたフレーム数
        #   FNA[pair] = gt_idが検出されたがそのpred_idではなかったフレーム数
        tpa: dict[tuple, int] = defaultdict(int)
        fpa: dict[tuple, int] = defaultdict(int)
        fna: dict[tuple, int] = defaultdict(int)

        for gt_f, pred_f in self._frames:
            if not gt_f and not pred_f:
                continue

            gt_ids   = [t for t, _ in gt_f]
            pred_ids = [t for t, _ in pred_f]
            gt_boxes   = np.array([b for _, b in gt_f],   np.float32) if gt_f   else np.zeros((0,4))
            pred_boxes = np.array([b for _, b in pred_f], np.float32) if pred_f else np.zeros((0,4))

            if len(gt_boxes) == 0:
                fp_total += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                fn_total += len(gt_boxes)
                continue

            iou = iou_matrix(gt_boxes, pred_boxes)
            ri, ci = linear_sum_assignment(-iou)

            matched_gt  = set()
            matched_pred= set()
            for r, c in zip(ri, ci):
                if iou[r, c] >= alpha:
                    tp_total += 1
                    matched_gt.add(r)
                    matched_pred.add(c)
                    pair = (gt_ids[r], pred_ids[c])
                    tpa[pair] += 1
                    # FPA: このgtに対して他のpredが割り当てられた場合
                    for c2 in range(len(pred_ids)):
                        if c2 not in matched_pred and iou[r, c2] >= alpha:
                            fpa[(gt_ids[r], pred_ids[c2])] += 1
                    # FNA: このpredに対して他のgtが存在する場合
                    for r2 in range(len(gt_ids)):
                        if r2 not in matched_gt and iou[r2, c] >= alpha:
                            fna[(gt_ids[r2], pred_ids[c])] += 1

            fp_total += len(pred_ids) - len(matched_pred)
            fn_total += len(gt_ids)   - len(matched_gt)

        # DetA
        denom_det = tp_total + fp_total + fn_total
        det_a = tp_total / max(denom_det, 1)

        # AssA: 全matchedペアの関連付け精度の平均
        all_pairs = set(tpa.keys()) | set(fpa.keys()) | set(fna.keys())
        if not all_pairs:
            ass_a = 1.0 if tp_total == 0 else 0.0
        else:
            ass_scores = []
            for pair in all_pairs:
                tp_a = tpa.get(pair, 0)
                fp_a = fpa.get(pair, 0)
                fn_a = fna.get(pair, 0)
                denom_a = tp_a + fp_a + fn_a
                ass_scores.append(tp_a / max(denom_a, 1))
            ass_a = float(np.mean(ass_scores))

        return det_a, ass_a
