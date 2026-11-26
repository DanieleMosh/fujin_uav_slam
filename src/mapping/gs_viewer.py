import cv2
import numpy as np


class GsViewer:
    """Web viewer (viser) for the Gaussian map: splats, trajectory and the current camera.

    Open http://localhost:8080 in a browser. viser serves from background threads,
    so calls here return immediately.
    """

    def __init__(self, opacity: float = 0.9):
        import viser
        import viser.transforms as vtf
        self._vtf = vtf
        self.server = viser.ViserServer(port=8080)
        self.opacity = opacity
        self.positions = []

    def set_map(self, means, rgbs, scales):
        """Replace the displayed Gaussians with the current map."""
        if len(means) == 0:
            return
        covariances = (scales ** 2)[:, None, None] * np.eye(3, dtype=np.float32)
        self.server.scene.add_gaussian_splats(
            "/map",
            centers=means,
            covariances=covariances,
            rgbs=rgbs.astype(np.float32) / 255.0,
            opacities=np.full((len(means), 1), self.opacity, np.float32),
        )

    def set_camera(self, T_wc, image_bgr, K):
        """Move the camera frustum to this keyframe and extend the trajectory."""
        scene = self.server.scene
        h, w = image_bgr.shape[:2]
        thumb = cv2.resize(image_bgr, (w // 2, h // 2))[:, :, ::-1]
        scene.add_camera_frustum(
            "/camera",
            fov=2 * np.arctan2(h / 2, K[1, 1]),
            aspect=w / h,
            scale=0.15,
            color=(0, 200, 80),
            image=thumb,
            wxyz=self._vtf.SO3.from_matrix(T_wc[:3, :3]).wxyz,
            position=T_wc[:3, 3],
        )

        self.positions.append(T_wc[:3, 3].copy())
        if len(self.positions) > 1:
            p = np.array(self.positions, dtype=np.float32)
            scene.add_line_segments(
                "/trajectory",
                points=np.stack([p[:-1], p[1:]], axis=1),
                colors=(50, 130, 255),
                thickness=0.005,
            )

    def stop(self):
        self.server.stop()
