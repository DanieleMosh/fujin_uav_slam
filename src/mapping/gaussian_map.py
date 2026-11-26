import cv2
import numpy as np

from input_process.sequence import Frame

SH_C0 = 0.28209479177387814  # zeroth spherical-harmonic basis constant (3DGS colour encoding)


def unproject_keyframe(frame: Frame, stride: int = 4, max_depth: float = 5.0, edge_tol: float = 0.04):
    """Turn one posed RGB-D keyframe into Gaussians, one per `stride`-th pixel.

    Returns (means Nx3 world, rgbs Nx3 uint8, scales N, cells N). `cells` is each
    Gaussian's flat index in the (H//stride, W//stride) sampling grid.
    Pixels with no depth, depth beyond `max_depth`, or on a depth discontinuity
    (relative jump > edge_tol, which would unproject to points floating in mid-air)
    are skipped.
    """
    depth = frame.depth
    h, w = depth.shape
    gh, gw = h // stride, w // stride
    v, u = np.mgrid[0:gh, 0:gw] * stride + stride // 2

    kernel = np.ones((3, 3), np.uint8)
    jump = cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)

    d = depth[v, u]
    valid = (d > 0) & (d < max_depth) & (jump[v, u] < edge_tol * d)
    d, u, v = d[valid], u[valid], v[valid]

    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    pts_cam = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], axis=1)
    means = pts_cam @ frame.T_wc[:3, :3].T + frame.T_wc[:3, 3]

    # average the colour over each sampling cell rather than taking one noisy pixel
    small = cv2.resize(frame.image_bgr[:gh * stride, :gw * stride], (gw, gh), interpolation=cv2.INTER_AREA)
    rgbs = small[v // stride, u // stride, ::-1]
    scales = 0.5 * d * stride / fx  # sigma = half the spacing between neighbouring samples
    cells = (v // stride) * gw + (u // stride)
    return means.astype(np.float32), np.ascontiguousarray(rgbs), scales.astype(np.float32), cells


class GaussianMap:
    """Growing set of isotropic Gaussians built incrementally from keyframes."""

    def __init__(self, stride: int = 4, max_depth: float = 8.0, cover_tol: float = 0.1,
                 refine_ratio: float = 0.7):
        self.stride = stride
        self.max_depth = max_depth
        self.cover_tol = cover_tol        # relative depth difference that counts as the same surface
        self.refine_ratio = refine_ratio  # a closer view replaces Gaussians it is this much finer than
        self.means = np.zeros((0, 3), np.float32)
        self.rgbs = np.zeros((0, 3), np.uint8)
        self.scales = np.zeros((0,), np.float32)

    def __len__(self):
        return len(self.means)

    def _coverage(self, frame: Frame):
        """Find the map Gaussians that lie on the surface `frame` measures.

        Returns (indices into the map, the sampling cell each one falls in). A Gaussian
        matches when it projects into a cell and its depth is within cover_tol of the
        depth measured there.
        """
        h, w = frame.depth.shape
        gh, gw = h // self.stride, w // self.stride
        if len(self) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)

        T_cw = np.linalg.inv(frame.T_wc).astype(np.float32)
        pts = self.means @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = pts[:, 2]
        idx = np.flatnonzero(z > 1e-3)
        pts, z = pts[idx], z[idx]
        gu = np.floor((pts[:, 0] / z * frame.K[0, 0] + frame.K[0, 2]) / self.stride).astype(np.int64)
        gv = np.floor((pts[:, 1] / z * frame.K[1, 1] + frame.K[1, 2]) / self.stride).astype(np.int64)
        inside = (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh)
        idx, gu, gv, z = idx[inside], gu[inside], gv[inside], z[inside]

        s = self.stride
        measured = frame.depth[gv * s + s // 2, gu * s + s // 2]
        same_surface = np.abs(z - measured) < self.cover_tol * measured
        return idx[same_surface], (gv * gw + gu)[same_surface]

    def insert(self, frame: Frame):
        """Merge a posed RGB-D keyframe into the map.

        Gaussians are added where the map has nothing on the observed surface, and
        where this view is closer (finer sampling) than what the map holds; in the
        second case the coarser Gaussians are removed. Returns (added, removed) counts.
        """
        h, w = frame.depth.shape
        num_cells = (h // self.stride) * (w // self.stride)
        map_idx, map_cell = self._coverage(frame)
        means, rgbs, scales, cells = unproject_keyframe(frame, self.stride, self.max_depth)

        # finest existing Gaussian per cell (inf = uncovered)
        cover_scale = np.full(num_cells, np.inf, np.float32)
        np.minimum.at(cover_scale, map_cell, self.scales[map_idx])
        add = scales < self.refine_ratio * cover_scale[cells]

        new_scale = np.full(num_cells, np.inf, np.float32)
        new_scale[cells[add]] = scales[add]
        replaced = map_idx[new_scale[map_cell] < self.refine_ratio * self.scales[map_idx]]
        keep = np.ones(len(self), dtype=bool)
        keep[replaced] = False

        self.means = np.concatenate([self.means[keep], means[add]])
        self.rgbs = np.concatenate([self.rgbs[keep], rgbs[add]])
        self.scales = np.concatenate([self.scales[keep], scales[add]])
        return int(add.sum()), len(replaced)

    def export_ply(self, path: str, opacity: float = 0.9):
        """Write the map in the standard 3D Gaussian Splatting PLY layout
        (degree-0 colour, log scales, logit opacity, identity rotation)."""
        n = len(self)
        fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.zeros((n, len(fields)), dtype="<f4")
        data[:, 0:3] = self.means
        data[:, 6:9] = (self.rgbs / 255.0 - 0.5) / SH_C0
        data[:, 9] = np.log(opacity / (1 - opacity))
        data[:, 10:13] = np.log(self.scales)[:, None]
        data[:, 13] = 1.0
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" \
            + "".join(f"property float {f}\n" for f in fields) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(data.tobytes())
