import os
from typing import Iterator

import cv2
import numpy as np
import yaml

from input_process.sequence import Frame, Sequence
from utils.geometry import quat_to_rot

# Data dir format for ETH's 'The EuRoC MAV Dataset' in ASL data format
# https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets


class EurocAslSequence(Sequence):
    """EuRoC MAV sequence in ASL format (one camera) with ground-truth camera poses.

    Images are undistorted with the calibration in the camera's sensor.yaml.
    Ground truth gives the body (IMU) pose at 200 Hz; it is interpolated to each
    image timestamp and composed with the camera extrinsics: T_wc = T_wb @ T_BS.
    Images outside the ground-truth time span are skipped.
    """

    def __init__(self, path: str, camera: str = "cam0"):
        self.image_dir = os.path.join(path, camera, "data")

        with open(os.path.join(path, camera, "sensor.yaml")) as f:
            sensor = yaml.safe_load(f)
        fx, fy, cx, cy = sensor["intrinsics"]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(sensor["distortion_coefficients"], dtype=np.float64)
        self.T_BS = np.array(sensor["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
        w, h = sensor["resolution"]
        self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_32FC1)

        images = np.loadtxt(os.path.join(path, camera, "data.csv"), delimiter=",", dtype=str, comments="#")
        image_ns = images[:, 0].astype(np.int64)
        # columns: timestamp [ns], position xyz, quaternion wxyz, ...
        gt = np.loadtxt(os.path.join(path, "state_groundtruth_estimate0", "data.csv"),
                        delimiter=",", comments="#")
        gt_ns = gt[:, 0].astype(np.int64)

        self.entries = []  # (timestamp seconds, filename, T_wc)
        for t_ns, filename in zip(image_ns, images[:, 1]):
            if t_ns < gt_ns[0] or t_ns > gt_ns[-1]:
                continue
            i = min(max(np.searchsorted(gt_ns, t_ns), 1), len(gt_ns) - 1)
            a = (t_ns - gt_ns[i - 1]) / (gt_ns[i] - gt_ns[i - 1])
            position = (1 - a) * gt[i - 1, 1:4] + a * gt[i, 1:4]
            q0, q1 = gt[i - 1, 4:8], gt[i, 4:8]
            if np.dot(q0, q1) < 0:  # q and -q are the same rotation; blend along the short way
                q1 = -q1
            qw, qx, qy, qz = (1 - a) * q0 + a * q1  # samples are 5 ms apart, so a linear blend is enough
            T_wb = np.eye(4)
            T_wb[:3, :3] = quat_to_rot(qx, qy, qz, qw)
            T_wb[:3, 3] = position
            self.entries.append((t_ns * 1e-9, filename.strip(), T_wb @ self.T_BS))

    def __len__(self):
        return len(self.entries)

    def __iter__(self) -> Iterator[Frame]:
        for t, filename, T_wc in self.entries:
            image = cv2.imread(os.path.join(self.image_dir, filename), cv2.IMREAD_COLOR)
            image = cv2.remap(image, *self._maps, cv2.INTER_LINEAR)
            yield Frame(timestamp=t, image_bgr=image, K=self.K, T_wc=T_wc)
