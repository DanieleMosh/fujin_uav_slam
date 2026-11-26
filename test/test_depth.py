import cv2
import numpy as np

from input_process.sequence import Frame
from mapping.depth import align_inverse_depth, triangulate_sparse_depth

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def test_alignment_recovers_scale_and_shift_despite_outliers():
    rng = np.random.default_rng(1)
    true_depth = rng.uniform(1.0, 4.0, (48, 64)).astype(np.float32)
    pred = (1.0 / true_depth - 0.05) / 0.7  # network output: 1/depth = 0.7 * pred + 0.05
    uv = np.stack([rng.integers(0, 64, 300), rng.integers(0, 48, 300)], axis=1).astype(float)
    anchors = true_depth[uv[:, 1].astype(int), uv[:, 0].astype(int)].copy()
    anchors[:40] *= 3.0  # bad triangulations

    metric = align_inverse_depth(pred, uv, anchors)
    np.testing.assert_allclose(metric, true_depth, rtol=1e-3)


def test_alignment_rejects_anchors_that_disagree():
    rng = np.random.default_rng(2)
    pred = rng.uniform(0.2, 1.0, (48, 64))
    uv = np.stack([rng.integers(0, 64, 300), rng.integers(0, 48, 300)], axis=1).astype(float)
    assert align_inverse_depth(pred, uv, rng.uniform(1.0, 4.0, 300)) is None
    assert align_inverse_depth(pred, uv[:10], np.ones(10)) is None  # too few anchors


class FakeExtractor:
    """Returns preset keypoints, so triangulation can be tested without real images."""
    def __init__(self, uv_a, uv_b):
        self.uv_a, self.uv_b = uv_a, uv_b

    def detect_and_compute(self, frame1, frame2):
        to_kp = lambda uv: [cv2.KeyPoint(float(u), float(v), 1) for u, v in uv]
        return to_kp(self.uv_a), self.uv_a, to_kp(self.uv_b), self.uv_b


class FakeMatcher:
    def match(self, des1, des2):
        return [cv2.DMatch(i, i, 0) for i in range(len(des1))]


def test_triangulated_depth_matches_known_points():
    rng = np.random.default_rng(3)
    points = rng.uniform([-1, -1, 2], [1, 1, 5], (50, 3))
    T_a, T_b = np.eye(4), np.eye(4)
    T_b[0, 3] = 0.3  # second camera 30 cm to the right

    def project(T_wc):
        cam = (points - T_wc[:3, 3]) @ T_wc[:3, :3]
        return cam[:, :2] / cam[:, 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]

    uv_a, uv_b = project(T_a), project(T_b)
    uv_b[:5] += 30.0  # wrong matches must be filtered out
    image = np.zeros((480, 640, 3), np.uint8)
    uv, depth = triangulate_sparse_depth(Frame(0, image, K, T_a), Frame(1, image, K, T_b),
                                         FakeExtractor(uv_a, uv_b), FakeMatcher())
    assert len(uv) == 45
    np.testing.assert_allclose(depth, points[5:, 2], rtol=1e-6)
