import numpy as np

from frontend.visual_odometry import VisualOdometry


def test_triangulate_points_recovers_known_geometry():
    """Regression: frame2's pose was inverted twice, giving wrong 3D points."""
    vo = VisualOdometry()
    rng = np.random.default_rng(0)
    points = rng.uniform([-1, -1, 3], [1, 1, 6], (40, 3))  # in frame1 coordinates

    angle = np.radians(5)
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    t = np.array([-0.5, 0.05, 0.1])  # x2 = R x1 + t, as cv2.recoverPose returns

    def project(cam):
        px = cam @ vo.K.T
        return px[:, :2] / px[:, 2:]

    world = vo._triangulate_points(R, t, project(points), project(points @ R.T + t))
    np.testing.assert_allclose(world, points, atol=1e-6)  # current_pose is identity
