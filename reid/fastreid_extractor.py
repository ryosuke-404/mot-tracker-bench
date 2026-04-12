"""
FastReID feature extractor.

Supports:
  - FastReID ONNX export (recommended for Jetson/production)
  - FastReID PyTorch (for development/training)
  - Fallback: lightweight OSNet via torchreid (Apache 2.0)

Output: L2-normalized feature vectors (shape: [feat_dim])
These are used for:
  1. Re-ID gallery queries (long-term identity maintenance)
  2. ByteTrack stage-1 fused cost matrix (appearance + IoU)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FastReIDExtractor:
    """
    Wraps FastReID (or OSNet fallback) to extract per-person appearance
    embeddings from image crops.

    Usage:
        extractor = FastReIDExtractor(model_path="reid_r50ibn.onnx", device="cuda")
        extractor.load_model()
        embeddings = extractor.extract_batch(frame, bboxes)  # shape (N, feat_dim)
    """

    # Normalization constants (ImageNet mean/std, used by FastReID)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        input_size: tuple[int, int] = (256, 128),  # (height, width)
        feature_dim: int = 2048,
        use_onnx: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.input_size = input_size   # (H, W)
        self.feature_dim = feature_dim
        self.use_onnx = use_onnx and model_path is not None

        self._session = None          # ONNX session
        self._torch_model = None      # PyTorch model

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load model. Must be called before extract_batch()."""
        if self.use_onnx and self.model_path:
            self._load_onnx()
        elif self.model_path:
            self._load_torch()
        else:
            self._load_osnet_fallback()

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        logger.info("Loading FastReID ONNX: %s", self.model_path)
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        # Verify feature dim from model output shape
        out_shape = self._session.get_outputs()[0].shape
        if out_shape and out_shape[-1]:
            self.feature_dim = int(out_shape[-1])
        logger.info("FastReID ONNX ready, feature_dim=%d", self.feature_dim)

    def _load_torch(self) -> None:
        """Load FastReID as PyTorch model (requires fast-reid package)."""
        try:
            from fastreid.config import get_cfg
            from fastreid.engine import DefaultPredictor
        except ImportError:
            logger.warning(
                "fast-reid package not found, falling back to OSNet. "
                "Install from https://github.com/JDAI-CV/fast-reid"
            )
            self._load_osnet_fallback()
            return

        from fastreid.config import get_cfg
        cfg = get_cfg()
        cfg.merge_from_file(self.model_path.replace(".pth", ".yaml"))
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.DEVICE = self.device
        self._torch_model = DefaultPredictor(cfg)
        logger.info("FastReID PyTorch model loaded from %s", self.model_path)

    def _load_osnet_fallback(self) -> None:
        """
        Fallback Re-ID: try torchreid OSNet first, then torchvision ResNet50.
        model_path can be an OSNet variant name: osnet_x0_25, osnet_x0_5,
        osnet_x0_75, osnet_x1_0, osnet_ibn_x1_0, osnet_ain_x1_0
        """
        _OSNET_VARIANTS = {
            "osnet_x0_25", "osnet_x0_5", "osnet_x0_75", "osnet_x1_0",
            "osnet_ibn_x1_0", "osnet_ain_x1_0",
        }
        # If model_path is an osnet variant name, use it directly
        osnet_name = "osnet_x0_25"
        if self.model_path in _OSNET_VARIANTS:
            osnet_name = self.model_path

        # ---- Option A: torchreid OSNet (512-dim) ----
        try:
            import torchreid
            import torch
            logger.info("Using %s as Re-ID backbone (torchreid)", osnet_name)
            model = torchreid.models.build_model(
                name=osnet_name, num_classes=1, pretrained=True,
            )
            model.eval()
            model.to(self.device)
            # Wrap in a module that returns only the feature vector (no classifier)
            self._torch_model = _OSNetFeatureWrapper(model, self.device)
            self.feature_dim = 512
            return
        except Exception as e:
            logger.debug("torchreid OSNet load failed (%s), falling back to ResNet50", e)

        # ---- Option B: torchvision ResNet50 (2048-dim, heavier) -----------
        try:
            import torch
            import torchvision.models as tvm
            logger.info("Using ResNet50 (ImageNet pretrained) as Re-ID backbone (fallback)")
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            self._torch_model = torch.nn.Sequential(*list(backbone.children())[:-1])
            self._torch_model.eval()
            self._torch_model.to(self.device)
            self.feature_dim = 2048
        except ImportError as e:
            raise ImportError(
                "Neither fast-reid, torchreid, nor torchvision is installed. "
                f"Original error: {e}"
            )

    # ------------------------------------------------------------------
    def extract_batch(
        self, frame: np.ndarray, bboxes: list[np.ndarray]
    ) -> np.ndarray:
        """
        Extract Re-ID embeddings for a batch of person crops.

        Args:
            frame   BGR frame from OpenCV  shape (H, W, 3)
            bboxes  list of [x1, y1, x2, y2] arrays

        Returns:
            embeddings  shape (N, feature_dim), L2-normalized
        """
        if not bboxes:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        crops = [self._crop_and_preprocess(frame, bbox) for bbox in bboxes]
        batch = np.stack(crops, axis=0)  # (N, 3, H, W)

        if self._session is not None:
            return self._infer_onnx(batch)
        elif self._torch_model is not None:
            return self._infer_torch(batch)
        else:
            # Model not loaded — return zero embeddings (degraded mode)
            logger.warning("Re-ID model not loaded, returning zero embeddings")
            return np.zeros((len(bboxes), self.feature_dim), dtype=np.float32)

    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single pre-cropped BGR image.

        Args:
            crop  BGR image  shape (H, W, 3)

        Returns:
            embedding  shape (feature_dim,)
        """
        preprocessed = self._preprocess_crop(crop)  # (3, H, W)
        batch = preprocessed[np.newaxis, ...]        # (1, 3, H, W)

        if self._session is not None:
            return self._infer_onnx(batch)[0]
        elif self._torch_model is not None:
            return self._infer_torch(batch)[0]
        return np.zeros(self.feature_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def _infer_onnx(self, batch: np.ndarray) -> np.ndarray:
        """batch shape: (N, 3, H, W), float32 in [0,1]"""
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        features = self._session.run([output_name], {input_name: batch})[0]
        return self._l2_normalize(features)

    def _infer_torch(self, batch: np.ndarray) -> np.ndarray:
        import torch
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            features = self._torch_model(tensor)
            if hasattr(features, "cpu"):
                features = features.cpu().numpy()
        features = np.asarray(features, dtype=np.float32)
        # ResNet Sequential returns (N, D, 1, 1) — flatten spatial dims
        if features.ndim == 4:
            features = features.reshape(features.shape[0], -1)
        return self._l2_normalize(features)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _crop_and_preprocess(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """Crop bbox from frame and preprocess to model input format."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((3, *self.input_size), dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        return self._preprocess_crop(crop)

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize, convert BGR→RGB, normalize.
        Returns float32 array of shape (3, H, W) in [0, 1].
        """
        h, w = self.input_size  # (height, width)
        resized = cv2.resize(crop, (w, h))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        normalized = (rgb - self._MEAN) / self._STD
        return normalized.transpose(2, 0, 1)  # HWC → CHW

    @staticmethod
    def _l2_normalize(features: np.ndarray) -> np.ndarray:
        """L2-normalize rows of a (N, D) matrix."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)


class _OSNetFeatureWrapper:
    """
    Wraps a torchreid OSNet model to extract pooled feature vectors
    (bypasses the classifier head) using a forward hook on global_avgpool.
    Output shape: (N, 512)
    """

    def __init__(self, model, device: str) -> None:
        import torch
        self._model = model
        self._device = device
        self._feat_cache: list = []

        # Register hook on 'fc' layer to capture (N, 512) feature vector
        # (global_avgpool gives 128-dim for osnet_x0_25; fc projects to 512)
        def _hook(module, input, output):
            self._feat_cache.append(output)

        for name, module in model.named_children():
            if name == "fc":
                module.register_forward_hook(_hook)
                break

    def __call__(self, tensor):
        import torch
        self._feat_cache.clear()
        with torch.no_grad():
            self._model(tensor)    # trigger hook

        if self._feat_cache:
            feat = self._feat_cache[0]           # (N, 512) from fc
            if feat.dim() == 4:
                feat = feat.view(feat.size(0), -1)
            return feat

        # Fallback: run forward and hope the output is usable
        with torch.no_grad():
            return self._model(tensor)

    def to(self, device):
        self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()
