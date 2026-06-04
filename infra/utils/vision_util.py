import numpy as np
import open3d as o3d


def depth_to_point_cloud(depth_m: np.ndarray, fx, fy, cx, cy):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def estimate_point_cloud_normals(
    points: np.ndarray,
    camera_pos: np.ndarray,
    radius: float = 0.02,
    max_nn: int = 30,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C>=3), got {points.shape}")

    xyz = np.asarray(points[:, :3], dtype=np.float64)
    camera_pos = np.asarray(camera_pos, dtype=np.float64).reshape(3)

    normals = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.any():
        return normals

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[finite_mask])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    pcd.orient_normals_towards_camera_location(camera_pos)
    normals[finite_mask] = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def normalize_point_cloud_workspace(
    points: np.ndarray,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C>=3), got {points.shape}")

    workspace_low = np.asarray(workspace_low, dtype=np.float32).reshape(3)
    workspace_high = np.asarray(workspace_high, dtype=np.float32).reshape(3)
    workspace_size = workspace_high - workspace_low
    if np.any(workspace_size <= 0):
        raise ValueError(
            f"workspace_high must be greater than workspace_low, got {workspace_low} and {workspace_high}"
        )

    normalized = points.astype(np.float32, copy=True)
    normalized[:, :3] = 2.0 * (normalized[:, :3] - workspace_low) / workspace_size - 1.0
    if clip:
        normalized[:, :3] = np.clip(normalized[:, :3], -1.0, 1.0)
    return normalized


class PointCloudDisplayer:
    def __init__(self, points: np.ndarray, left=100, top=100, width=640, height=480):
        self.window = o3d.visualization.Visualizer()
        self.window.create_window(
            window_name="Point Cloud",
            height=height,
            width=width,
            visible=True,
            left=left,
            top=top,
        )
        opt = self.window.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 3.0
        opt.show_coordinate_frame = True
        self.pc = o3d.geometry.PointCloud()
        self.pc.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        self.window.add_geometry(self.pc)
        self.window.add_geometry(self.coord_frame)
        self._first_frame = True

    def display(self, points: np.ndarray):
        self.pc.points = o3d.utility.Vector3dVector(
            points[:, :3].astype(np.float64)
        )
        if self._first_frame:
            self.window.reset_view_point(True)
            self._first_frame = False
        self.window.update_geometry(self.pc)
        self.window.poll_events()
        self.window.update_renderer()

    def close(self):
        self.window.destroy_window()