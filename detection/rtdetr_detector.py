"""
RT-DETRv2 object detector wrapper.

Supports two backends:
  1. HuggingFace `transformers` (default, PyTorch)
  2. ONNX Runtime (for optimized Jetson deployment)

Only "person" class detections are returned.
Detections are split into high-score and low-score lists for ByteTrack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    """Person detection output from RT-DETRv2."""
    bbox: np.ndarray    # [x1, y1, x2, y2] in pixel coordinates
    score: float


class RTDETRv2Detector:
    """
    Wraps RT-DETRv2 (via HuggingFace transformers or ONNX) and returns
    person-only detections split into high- and low-confidence lists.
    """

    COCO_PERSON_LABEL = "person"

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_v2_r50vd",
        device: str = "cuda",
        high_score_thresh: float = 0.50,
        low_score_thresh: float = 0.10,
        input_size: tuple[int, int] = (640, 640),
        onnx_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.input_size = input_size  # (width, height)
        self.onnx_path = onnx_path

        self._processor = None
        self._model = None
        self._onnx_session = None
        self._use_onnx = onnx_path is not None

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load model weights. Call once before first `detect()`."""
        if self._use_onnx:
            self._load_onnx()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        try:
            from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers>=4.40.0 and torch are required for RT-DETRv2. "
                f"Original error: {e}"
            )

        logger.info("Loading RT-DETRv2 from HuggingFace: %s", self.model_name)
        self._processor = RTDetrImageProcessor.from_pretrained(self.model_name)
        self._model = RTDetrV2ForObjectDetection.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("RT-DETRv2 loaded on device=%s", self.device)

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(f"onnxruntime is required for ONNX backend: {e}")

        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Loading RT-DETRv2 ONNX from: %s", self.onnx_path)
        self._onnx_session = ort.InferenceSession(self.onnx_path, providers=providers)
        logger.info("RT-DETRv2 ONNX session ready")

    # ------------------------------------------------------------------
    def detect(
        self, frame: np.ndarray
    ) -> tuple[list[RawDetection], list[RawDetection]]:
        """
        Detect persons in a BGR frame (as returned by OpenCV).

        Returns:
            high_dets  score ≥ high_score_thresh
            low_dets   low_score_thresh ≤ score < high_score_thresh
        """
        if self._use_onnx:
            results = self._detect_onnx(frame)
        else:
            results = self._detect_transformers(frame)

        high_dets: list[RawDetection] = []
        low_dets: list[RawDetection] = []

        for det in results:
            if det.score >= self.high_score_thresh:
                high_dets.append(det)
            elif det.score >= self.low_score_thresh:
                low_dets.append(det)

        return high_dets, low_dets

    # ------------------------------------------------------------------
    def _detect_transformers(self, frame: np.ndarray) -> list[RawDetection]:
        import torch
        from PIL import Image

        # Convert BGR (OpenCV) → RGB PIL Image
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        orig_h, orig_w = frame.shape[:2]

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Decode results to the original image size
        target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        detections: list[RawDetection] = []
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()  # [x1, y1, x2, y2]

        id2label = self._model.config.id2label
        for label_id, score, box in zip(labels, scores, boxes):
            label_name = id2label.get(int(label_id), "").lower()
            if label_name != self.COCO_PERSON_LABEL:
                continue
            detections.append(RawDetection(bbox=box.astype(float), score=float(score)))

        return detections

    def _detect_onnx(self, frame: np.ndarray) -> list[RawDetection]:
        """ONNX inference path for Jetson/TensorRT deployment."""
        import cv2

        orig_h, orig_w = frame.shape[:2]
        w, h = self.input_size

        # Preprocess: resize, BGR→RGB, normalize to [0,1], add batch dim
        resized = cv2.resize(frame, (w, h))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        input_name = self._onnx_session.get_inputs()[0].name
        output_names = [o.name for o in self._onnx_session.get_outputs()]
        outs = self._onnx_session.run(output_names, {input_name: blob})

        # Expected outputs: boxes (N,4) in [0,1] normalized + scores (N,) + labels (N,)
        # (Exact format depends on how RT-DETRv2 was exported to ONNX)
        boxes_norm, scores, labels = outs[0][0], outs[1][0], outs[2][0]

        detections: list[RawDetection] = []
        for box_norm, score, label in zip(boxes_norm, scores, labels):
            if int(label) != 0:  # COCO person class = 0
                continue
            x1 = float(box_norm[0]) * orig_w
            y1 = float(box_norm[1]) * orig_h
            x2 = float(box_norm[2]) * orig_w
            y2 = float(box_norm[3]) * orig_h
            detections.append(
                RawDetection(bbox=np.array([x1, y1, x2, y2]), score=float(score))
            )

        return detections
