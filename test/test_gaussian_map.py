import numpy as np

from input_process.sequence import Frame
from mapping.gaussian_map import GaussianMap, unproject_keyframe
from utils.geometry import pose_moved, quat_to_rot

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def make_frame(T_wc, depth_value=2.0):
    """A camera looking at a fronto-parallel wall `depth_value` metres ahead."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), depth_value, np.float32)
    return Frame(timestamp=0.0, image_bgr=image, K=K, T_wc=T_wc, depth=depth)


def test_unproject_reprojects_to_source_pixels():
    T_wc = np.eye(4)
    T_wc[:3, :3] = quat_to_rot(0.1, -0.2, 0.3, 0.9)
    T_wc[:3, 3] = (1.0, -2.0, 0.5)
    frame = make_frame(T_wc)
    stride = 8
    means, rgbs, scales, cells = unproject_keyframe(frame, stride=stride)

    assert len(means) == (480 // stride) * (640 // stride)
    T_cw = np.linalg.inv(T_wc)
    pts = means @ T_cw[:3, :3].T + T_cw[:3, 3]
    u = pts[:, 0] / pts[:, 2] * K[0, 0] + K[0, 2]
    v = pts[:, 1] / pts[:, 2] * K[1, 1] + K[1, 2]
    gw = 640 // stride
    np.testing.assert_allclose(u, (cells % gw) * stride + stride // 2, atol=1e-2)
    np.testing.assert_allclose(v, (cells // gw) * stride + stride // 2, atol=1e-2)
    # colour is the mean of the sampling cell, converted BGR -> RGB
    np.testing.assert_allclose(rgbs[0], frame.image_bgr[:stride, :stride, ::-1].mean(axis=(0, 1)), atol=1)
    np.testing.assert_allclose(scales, 0.5 * 2.0 * stride / 500.0)


def test_unproject_skips_invalid_far_and_edge_pixels():
    frame = make_frame(np.eye(4))
    frame.depth[:100] = 0.0       # no measurement
    frame.depth[100:200] = 9.0    # beyond max_depth
    means, _, _, _ = unproject_keyframe(frame, stride=4, max_depth=5.0)
    assert np.all(means[:, 2] == 2.0)
    assert np.all(means[:, 1] > (200 - 240) / 500.0 * 2.0)  # nothing from the top 200 rows


def test_reobserved_surface_is_not_duplicated():
    gaussian_map = GaussianMap(stride=4)
    added, _ = gaussian_map.insert(make_frame(np.eye(4)))
    assert added > 0
    assert gaussian_map.insert(make_frame(np.eye(4))) == (0, 0)


def test_shifted_view_adds_only_the_new_part():
    gaussian_map = GaussianMap(stride=4)
    first, _ = gaussian_map.insert(make_frame(np.eye(4)))
    shifted = np.eye(4)
    shifted[0, 3] = 1.28  # half an image width sideways at 2 m depth
    second, removed = gaussian_map.insert(make_frame(shifted))
    assert 0.4 * first < second < 0.6 * first
    assert removed == 0


def test_surface_at_different_depth_is_added():
    gaussian_map = GaussianMap(stride=4)
    first, _ = gaussian_map.insert(make_frame(np.eye(4), depth_value=2.0))
    second, removed = gaussian_map.insert(make_frame(np.eye(4), depth_value=1.0))
    assert second == first
    assert removed == 0


def test_closer_view_replaces_coarse_gaussians():
    gaussian_map = GaussianMap(stride=4)
    first, _ = gaussian_map.insert(make_frame(np.eye(4), depth_value=2.0))
    closer = np.eye(4)
    closer[2, 3] = 1.0  # 1 m from the same wall: sees its central quarter at twice the detail
    added, removed = gaussian_map.insert(make_frame(closer, depth_value=1.0))
    assert added == first
    assert 0.2 * first < removed < 0.3 * first
    # going back to the far view must not re-add coarse Gaussians over the fine ones
    assert gaussian_map.insert(make_frame(np.eye(4), depth_value=2.0)) == (0, 0)


def test_ply_export(tmp_path):
    gaussian_map = GaussianMap(stride=8)
    gaussian_map.insert(make_frame(np.eye(4)))
    ply = tmp_path / "map.ply"
    gaussian_map.export_ply(str(ply))
    raw = ply.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert f"element vertex {len(gaussian_map)}".encode() in header
    data = np.frombuffer(body, dtype="<f4").reshape(len(gaussian_map), 17)
    np.testing.assert_allclose(data[:, :3], gaussian_map.means)
    np.testing.assert_allclose(np.exp(data[:, 10]), gaussian_map.scales, rtol=1e-5)


def test_pose_moved():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = 0.05
    assert not pose_moved(a, b)
    b[0, 3] = 0.2
    assert pose_moved(a, b)
    c = np.eye(4)
    c[:3, :3] = quat_to_rot(0, 0, np.sin(np.radians(10)), np.cos(np.radians(10)))  # 20 deg about z
    assert pose_moved(a, c)
