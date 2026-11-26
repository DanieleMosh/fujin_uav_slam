import os
from typing import Iterator

import cv2
import numpy as np

from input_process.sequence import Frame, Sequence
from utils.geometry import quat_to_rot

# TUM RGB-D benchmark: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
# Calibration per Freiburg camera: (fx, fy, cx, cy), (k1, k2, p1, p2, k3)
TUM_CALIBRATION = {
    "freiburg1": ((517.3, 516.5, 318.6, 255.3), (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "freiburg2": ((520.9, 521.0, 325.1, 249.7), (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
    "freiburg3": ((535.4, 539.2, 320.1, 247.6), (0.0, 0.0, 0.0, 0.0, 0.0)),
}
DEPTH_SCALE = 5000.0  # 16-bit depth PNG value per metre


def _read_list(path):
    """Read a TUM index file: '#' comments, then 'timestamp value...' rows."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                rows.append((float(parts[0]), parts[1:]))
    return rows


def _nearest(times, t):
    """Index of the entry in sorted `times` closest to t."""
    i = np.searchsorted(times, t)
    i = min(max(i, 1), len(times) - 1)
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


class TumRgbdSequence(Sequence):
    """TUM RGB-D sequence with sensor depth and ground-truth camera poses.

    Colour, depth and pose are recorded at different times, so each colour image
    is paired with the nearest depth image and pose within `max_dt` seconds.
    """

    def __init__(self, path: str, max_dt: float = 0.02):
        self.path = path
        name = os.path.basename(os.path.normpath(path))
        camera = next((c for c in TUM_CALIBRATION if c in name), None)
        if camera is None:
            raise ValueError(f"Cannot tell which Freiburg camera recorded '{name}' "
                             f"(expected one of {list(TUM_CALIBRATION)} in the folder name)")
        (fx, fy, cx, cy), dist = TUM_CALIBRATION[camera]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(dist)

        rgb = _read_list(os.path.join(path, "rgb.txt"))
        depth = _read_list(os.path.join(path, "depth.txt"))
        gt = _read_list(os.path.join(path, "groundtruth.txt"))
        depth_t = np.array([t for t, _ in depth])
        gt_t = np.array([t for t, _ in gt])

        self.entries = []  # (timestamp, rgb file, depth file, T_wc)
        for t, (rgb_file,) in rgb:
            di, gi = _nearest(depth_t, t), _nearest(gt_t, t)
            if abs(depth_t[di] - t) > max_dt or abs(gt_t[gi] - t) > max_dt:
                continue
            tx, ty, tz, qx, qy, qz, qw = map(float, gt[gi][1])
            T_wc = np.eye(4)
            T_wc[:3, :3] = quat_to_rot(qx, qy, qz, qw)
            T_wc[:3, 3] = (tx, ty, tz)
            self.entries.append((t, rgb_file, depth[di][1][0], T_wc))

        self._maps = None  # undistortion maps, built on first frame

    def __len__(self):
        return len(self.entries)

    def __iter__(self) -> Iterator[Frame]:
        for t, rgb_file, depth_file, T_wc in self.entries:
            image = cv2.imread(os.path.join(self.path, rgb_file), cv2.IMREAD_COLOR)
            depth = cv2.imread(os.path.join(self.path, depth_file), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / DEPTH_SCALE

            if self.dist.any():
                if self._maps is None:
                    h, w = image.shape[:2]
                    self._maps = cv2.initUndistortRectifyMap(
                        self.K, self.dist, None, self.K, (w, h), cv2.CV_32FC1)
                image = cv2.remap(image, *self._maps, cv2.INTER_LINEAR)
                # depth is registered to the colour image; nearest avoids blending across edges
                depth = cv2.remap(depth, *self._maps, cv2.INTER_NEAREST)

            yield Frame(timestamp=t, image_bgr=image, K=self.K, T_wc=T_wc, depth=depth)
