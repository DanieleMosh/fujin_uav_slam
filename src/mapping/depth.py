import cv2
import numpy as np

from frontend.feature_extraction import FeatureExtractor, ORBFeatureExtractor
from frontend.feature_matching import BFMatcher, FeatureMatcher
from input_process.sequence import Frame


class DepthAnythingV2:
    """Depth Anything V2 (relative) monocular depth network.

    Predicts inverse depth up to an unknown scale and shift; see align_inverse_depth.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """HxW float32 relative inverse depth (larger = closer)."""
        torch = self._torch
        h, w = image_bgr.shape[:2]
        inputs = self.processor(images=image_bgr[:, :, ::-1].copy(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(pred[None], size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].float().cpu().numpy()


def align_inverse_depth(pred, uv, depth, iterations=200, inlier_tol=0.1,
                        min_inliers=100, min_inlier_ratio=0.75, seed=0):
    """Fit scale and shift so that 1/metric_depth = a * pred + b at sparse anchors.

    pred: HxW relative inverse depth. uv: Nx2 pixel coordinates. depth: N metric depths.
    RANSAC over anchor pairs, then least squares on the inliers (an anchor is an
    inlier when its metric inverse depth is matched within inlier_tol, relative).
    Returns HxW metric depth (0 = invalid), or None if too few anchors agree:
    a poorly aligned keyframe would put a misplaced copy of the scene in the map.
    """
    n = len(uv)
    if n < min_inliers:
        return None
    x = pred[np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)].astype(np.float64)
    y = 1.0 / depth.astype(np.float64)

    rng = np.random.default_rng(seed)
    best = np.zeros(n, dtype=bool)
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        b = y[i] - a * x[i]
        inliers = np.abs(a * x + b - y) < inlier_tol * y
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(min_inliers, min_inlier_ratio * n):
        return None

    a, b = np.polyfit(x[best], y[best], 1)
    if a <= 0:  # the network's ordering must agree with the anchors
        return None
    inv = a * pred + b
    metric = np.zeros(pred.shape, dtype=np.float32)
    np.divide(1.0, inv, out=metric, where=inv > 1e-6)
    return metric


def triangulate_sparse_depth(frame: Frame, reference: Frame, extractor: FeatureExtractor,
                             matcher: FeatureMatcher, max_reproj: float = 2.0, min_parallax_deg: float = 1.0):
    """Metric depths at matched features of `frame`, triangulated against `reference`.

    Both frames need known poses. Matches are kept when they reproject within
    max_reproj pixels in both views and the viewing rays differ by min_parallax_deg.
    Returns (uv Nx2 pixels in frame, depth N metres).
    """
    gray_a = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(reference.image_bgr, cv2.COLOR_BGR2GRAY)
    kp_a, des_a, kp_b, des_b = extractor.detect_and_compute(gray_a, gray_b)
    if des_a is None or des_b is None:
        return np.zeros((0, 2)), np.zeros(0)
    matches = matcher.match(des_a, des_b)
    if len(matches) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    uv_a = np.float64([kp_a[m.queryIdx].pt for m in matches])
    uv_b = np.float64([kp_b[m.trainIdx].pt for m in matches])

    T_a, T_b = np.linalg.inv(frame.T_wc), np.linalg.inv(reference.T_wc)  # world -> camera
    P_a, P_b = frame.K @ T_a[:3], reference.K @ T_b[:3]
    X = cv2.triangulatePoints(P_a, P_b, uv_a.T, uv_b.T)
    X = (X[:3] / X[3]).T  # world points

    def project(P, T):
        cam = X @ T[:3, :3].T + T[:3, 3]
        px = X @ P[:, :3].T + P[:, 3]
        return cam[:, 2], px[:, :2] / px[:, 2:]

    z_a, px_a = project(P_a, T_a)
    z_b, px_b = project(P_b, T_b)
    ray_a = X - frame.T_wc[:3, 3]
    ray_b = X - reference.T_wc[:3, 3]
    cos = np.sum(ray_a * ray_b, axis=1) / (np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1))
    good = (z_a > 0) & (z_b > 0) \
        & (np.linalg.norm(px_a - uv_a, axis=1) < max_reproj) \
        & (np.linalg.norm(px_b - uv_b, axis=1) < max_reproj) \
        & (cos < np.cos(np.radians(min_parallax_deg)))
    return uv_a[good], z_a[good]


class AlignedMonoDepth:
    """Monocular network depth made metric with sparse depth anchors."""

    def __init__(self, network: DepthAnythingV2 = None, extractor: FeatureExtractor = None,
                 matcher: FeatureMatcher = None):
        self.network = network or DepthAnythingV2()
        self.extractor = extractor or ORBFeatureExtractor()
        self.matcher = matcher or BFMatcher()

    def estimate(self, frame: Frame, reference: Frame = None, anchors=None):
        """HxW metric depth for `frame` (0 = invalid), or None if it cannot be aligned.

        anchors: (uv, depth) of points with known depth in this frame, e.g. from the
        tracker. Without them, anchors are triangulated against the posed `reference`.
        """
        if anchors is None:
            if reference is None:
                return None
            anchors = triangulate_sparse_depth(frame, reference, self.extractor, self.matcher)
        return align_inverse_depth(self.network.predict(frame.image_bgr), *anchors)
