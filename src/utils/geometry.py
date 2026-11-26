import numpy as np


def quat_to_rot(qx, qy, qz, qw):
    """Unit quaternion to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_moved(T_a, T_b, min_translation=0.1, min_rotation_deg=10.0):
    """True if two 4x4 poses differ by more than the given translation (m) or rotation."""
    if np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) > min_translation:
        return True
    cos_angle = (np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1))) > min_rotation_deg
