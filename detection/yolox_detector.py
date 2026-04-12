"""
YOLOX person detector wrapper.

Supports:
  - torch.hub (Megvii-BaseDetection/YOLOX) — yolox_nano/tiny/s/m/l/x
  - ONNX Runtime (for Jetson/TensorRT deployment)

License: YOLOX Apache 2.0 — commercial-safe without source disclosure.

YOLOX preprocessing differs from standard ImageNet normalization:
  - Input: BGR float32, NOT divided by 255, NOT normalized
  - Padded to square with gray (114, 114, 114)
  - Standard sizes: 416, 640 (yolox_s default), 800 (yolox_m/l/x)

Output is identical to RTDETRv2Detector:
  detect() -> (high_dets: list[RawDetection], low_dets: list[RawDetection])
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from detection.rtdetr_detector import RawDetection   # reuse same dataclass

logger = logging.getLogger(__name__)

# YOLOX COCO class index for "person"
_PERSON_CLASS_ID = 0

# Available model variants via torch.hub
YOLOX_MODELS = {
    "yolox_nano": 416,
    "yolox_tiny": 416,
    "yolox_s":    640,
    "yolox_m":    640,
    "yolox_l":    640,
    "yolox_x":    640,
}


class YOLOXDetector:
    """
    YOLOX person detector with the same interface as RTDETRv2Detector.

    Usage:
        det = YOLOXDetector(model_name="yolox_s", device="cpu")
        det.load_model()
        high_dets, low_dets = det.detect(frame)   # frame: BGR np.ndarray
    """

    def __init__(
        self,
        model_name: str = "yolox_s",
        device: str = "cpu",
        high_score_thresh: float = 0.45,
        low_score_thresh: float = 0.10,
        nms_thresh: float = 0.45,
        input_size: Optional[int] = None,    # None → use model default
        onnx_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.nms_thresh = nms_thresh
        self.onnx_path = onnx_path
        self._use_onnx = onnx_path is not None

        # Determine input size
        if input_size is not None:
            self.input_size = input_size
        else:
            self.input_size = YOLOX_MODELS.get(model_name, 640)

        self._model = None
        self._onnx_session = None

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._use_onnx:
            self._load_onnx()
        else:
            self._load_torch_hub()

    def _load_torch_hub(self) -> None:
        import torch
        logger.info("Loading YOLOX via torch.hub: %s (input_size=%d)",
                    self.model_name, self.input_size)
        self._model = torch.hub.load(
            "Megvii-BaseDetection/YOLOX",
            self.model_name,
            pretrained=True,
            trust_repo=True,
            verbose=False,
        )
        self._model.eval()
        self._model.to(self.device)
        logger.info("YOLOX %s loaded on %s", self.model_name, self.device)

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        logger.info("Loading YOLOX ONNX: %s", self.onnx_path)
        self._onnx_session = ort.InferenceSession(self.onnx_path, providers=providers)
        logger.info("YOLOX ONNX session ready")

    # ------------------------------------------------------------------
    def detect(
        self, frame: np.ndarray
    ) -> tuple[list[RawDetection], list[RawDetection]]:
        """
        Detect persons in a BGR frame.

        Returns:
            high_dets  score ≥ high_score_thresh
            low_dets   low_score_thresh ≤ score < high_score_thresh
        """
        if self._use_onnx:
            results = self._detect_onnx(frame)
        else:
            results = self._detect_torch(frame)

        high_dets, low_dets = [], []
        for det in results:
            if det.score >= self.high_score_thresh:
                high_dets.append(det)
            elif det.score >= self.low_score_thresh:
                low_dets.append(det)
        return high_dets, low_dets

    # ------------------------------------------------------------------
    def _detect_torch(self, frame: np.ndarray) -> list[RawDetection]:
        import torch

        orig_h, orig_w = frame.shape[:2]
        img, ratio = self._preprocess(frame, self.input_size)

        tensor = torch.from_numpy(img[np.newaxis]).to(self.device)   # (1, 3, H, W)

        with torch.no_grad():
            outputs = self._model(tensor)   # (1, N, 85) for COCO

        # outputs[0]: (N, 85)  — [cx, cy, w, h, obj_conf, cls_conf x 80]
        preds = outputs[0].cpu().numpy()
        return self._postprocess(preds, ratio, orig_w, orig_h)

    def _detect_onnx(self, frame: np.ndarray) -> list[RawDetection]:
        orig_h, orig_w = frame.shape[:2]
        img, ratio = self._preprocess(frame, self.input_size)

        input_name = self._onnx_session.get_inputs()[0].name
        output_name = self._onnx_session.get_outputs()[0].name
        outputs = self._onnx_session.run(
            [output_name], {input_name: img[np.newaxis]}
        )[0]  # (1, N, 85)

        preds = outputs[0]
        return self._postprocess(preds, ratio, orig_w, orig_h)

    # ------------------------------------------------------------------
    def _preprocess(
        self, frame: np.ndarray, input_size: int
    ) -> tuple[np.ndarray, float]:
        """
        Resize + letterbox to square, keeping aspect ratio.
        Returns:
            img    float32 CHW array (no normalization — YOLOX raw pixel input)
            ratio  scale factor applied (to invert bbox coords later)
        """
        orig_h, orig_w = frame.shape[:2]
        ratio = min(input_size / orig_h, input_size / orig_w)
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)

        resized = cv2.resize(frame, (new_w, new_h))

        # Letterbox padding with gray (114, 114, 114)
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # BGR → float32, no normalization
        img = padded.astype(np.float32).transpose(2, 0, 1)   # CHW
        return img, ratio

    def _postprocess(
        self,
        preds: np.ndarray,
        ratio: float,
        orig_w: int,
        orig_h: int,
    ) -> list[RawDetection]:
        """
        Decode YOLOX raw output to RawDetection list.

        preds shape: (N, num_classes + 5)
          cols: [cx, cy, w, h, obj_conf, cls_conf_0, ..., cls_conf_79]
        """
        if preds.shape[0] == 0:
            return []

        # Objectness × class confidence
        obj_conf = preds[:, 4:5]          # (N, 1)
        cls_conf = preds[:, 5:]           # (N, num_classes)
        scores   = obj_conf * cls_conf    # (N, num_classes)

        # Person class only
        person_scores = scores[:, _PERSON_CLASS_ID]  # (N,)

        # Filter below minimum threshold
        mask = person_scores >= self.low_score_thresh
        if mask.sum() == 0:
            return []

        preds_f        = preds[mask]
        person_scores  = person_scores[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2 (in padded input space)
        cx = preds_f[:, 0]
        cy = preds_f[:, 1]
        w  = preds_f[:, 2]
        h  = preds_f[:, 3]

        x1 = (cx - w / 2) / ratio
        y1 = (cy - h / 2) / ratio
        x2 = (cx + w / 2) / ratio
        y2 = (cy + h / 2) / ratio

        # Clip to original image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS per-class (person only)
        boxes_nms = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        keep = self._nms(boxes_nms, person_scores, self.nms_thresh)

        detections = []
        for i in keep:
            bbox = np.array([x1[i], y1[i], x2[i], y2[i]], dtype=float)
            detections.append(RawDetection(bbox=bbox, score=float(person_scores[i])))
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        """Simple greedy NMS. Returns indices of kept boxes."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
            order = order[1:][iou <= iou_thresh]
        return keep
