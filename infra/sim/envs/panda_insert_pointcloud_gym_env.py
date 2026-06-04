import mujoco
import fpsample
import numpy as np
import gymnasium as gym
from infra.utils.vision_util import (
    depth_to_point_cloud,
    estimate_point_cloud_normals,
    normalize_point_cloud_workspace,
    PointCloudDisplayer,
)
from infra.utils.transformations import construct_homogeneous_matrix
from typing import Any, Literal, Tuple, Dict
from infra.sim.envs.mujoco_gym_env import GymRenderingSpec
from infra.sim.envs.panda_insert_gym_env import DefaultEnvConfig, PandaPegInsertGymEnv


class PandaPegInsertDepthGymEnv(PandaPegInsertGymEnv):
    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: list[GymRenderingSpec] = [GymRenderingSpec(camera_name="wrist1", mode="depth_array"),
                                               GymRenderingSpec(camera_name="wrist2", mode="depth_array")],
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        config: DefaultEnvConfig = None,
    ):
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
            render_mode=render_mode,
            config=config
        )

        point_cloud_channels = 6 if getattr(self.config, "POINT_CLOUD_WITH_NORMALS", False) else 3
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "point_cloud": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(512, point_cloud_channels),
                            dtype=np.float32,
                        )
                    }
                ),
            }
        )

        self.wrist1_K = self._get_camera_intrinsics("wrist1", 128, 128)
        self.wrist2_K = self._get_camera_intrinsics("wrist2", 128, 128)

        self._pc_displayer = None

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset()
        point_cloud = self._get_pointcloud(obs)
        obs["images"]["point_cloud"] = point_cloud

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, info = super().step(action)
        point_cloud = self._get_pointcloud(obs)

        if self.config.DISPLAY_IMAGE and self._pc_displayer is None:
            self._pc_displayer = PointCloudDisplayer(points=point_cloud)
        if self._pc_displayer is not None:
            self._pc_displayer.display(points=point_cloud) 

        obs["images"]["point_cloud"] = point_cloud
        
        return obs, reward, done, truncated, info

    def _get_camera_intrinsics(self, camera_name: str, height: int, width: int):
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self._model.cam_fovy[cam_id]
        fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        return fx, fy, cx, cy

    def _get_pointcloud(self, obs):
        # 1. reconstruct point cloud
        wrist1_depth = self._depth_buffer_to_meters(obs["images"]["wrist1"])
        wrist2_depth = self._depth_buffer_to_meters(obs["images"]["wrist2"])

        w1_fx, w1_fy, w1_cx, w1_cy = self.wrist1_K
        w2_fx, w2_fy, w2_cx, w2_cy = self.wrist2_K

        wrist1_pc = depth_to_point_cloud(wrist1_depth, w1_fx, w1_fy, w1_cx, w1_cy)
        wrist2_pc = depth_to_point_cloud(wrist2_depth, w2_fx, w2_fy, w2_cx, w2_cy)

        # 2. filter point cloud in world
        cam1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist1")
        cam2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist2")

        pw1 = self._filter_point_cloud(wrist1_pc, self._data, cam1)
        pw2 = self._filter_point_cloud(wrist2_pc, self._data, cam2)

        merged = np.concatenate([pw1, pw2], axis=0).astype(np.float32)
        merged_idx = fpsample.fps_sampling(merged[:, :3], 512)
        merged = merged[merged_idx]

        normals_world = None
        if getattr(self.config, "POINT_CLOUD_WITH_NORMALS", False):
            camera_pos_world = 0.5 * (
                self._data.cam_xpos[cam1].astype(np.float64)
                + self._data.cam_xpos[cam2].astype(np.float64)
            )
            normals_world = estimate_point_cloud_normals(
                merged,
                camera_pos_world,
                radius=getattr(self.config, "NORMAL_ESTIMATION_RADIUS", 0.02),
                max_nn=getattr(self.config, "NORMAL_ESTIMATION_MAX_NN", 30),
            )

        # 3. convert point cloud to tool frame
        T_world_tcp = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        T_tcp_world = np.linalg.inv(T_world_tcp)
        ones = np.ones((merged.shape[0], 1), dtype=np.float32)
        pw_h = np.concatenate([merged.astype(np.float64), ones], axis=1)  # (N,4)
        pe_h = (T_tcp_world @ pw_h.T).T
        point_cloud_tcp = pe_h[:, :3].astype(np.float32)

        if normals_world is not None:
            normals_tcp = (T_tcp_world[:3, :3] @ normals_world.T).T.astype(np.float32)
            point_cloud_tcp = np.concatenate([point_cloud_tcp, normals_tcp], axis=1)

        if getattr(self.config, "POINT_CLOUD_NORMALIZE_WORKSPACE", False):
            point_cloud_tcp = normalize_point_cloud_workspace(
                point_cloud_tcp,
                workspace_low=getattr(
                    self.config,
                    "POINT_CLOUD_WORKSPACE_LOW",
                    [-0.1, -0.1, -0.1],
                ),
                workspace_high=getattr(
                    self.config,
                    "POINT_CLOUD_WORKSPACE_HIGH",
                    [0.1, 0.1, 0.1],
                ),
                clip=getattr(self.config, "POINT_CLOUD_WORKSPACE_CLIP", True),
            )

        return point_cloud_tcp
    
    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        near = self._model.vis.map.znear * self._model.stat.extent
        far = self._model.vis.map.zfar * self._model.stat.extent
        depth_m = near / (1.0 - depth_buffer * (1.0 - near / far))
        return depth_m
    
    def _filter_point_cloud(self, points_cam: np.ndarray, data, cam_id: int) -> np.ndarray:
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        # Back-projection above uses OpenCV-like camera coords:
        # x right, y down, z forward.
        # MuJoCo/OpenGL camera convention differs by a flip on y and z.
        cv_to_mj = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        points_mj_cam = points_cam @ cv_to_mj.T
        points_world = points_mj_cam @ cam_rot.T + cam_pos

        z = points_world[:, 2]
        z_min, z_max = 0.005, 1.2
        mask = np.isfinite(points_world).all(axis=1) & (z >= z_min) & (z <= z_max)
        points_world = points_world[mask]
        return points_world

    def close(self) -> None:
        if self._pc_displayer is not None:
            self._pc_displayer.close()
            self._pc_displayer = None
        super().close()


if __name__ == "__main__":
    env = PandaPegInsertDepthGymEnv(render_mode="human", config=DefaultEnvConfig())
    env.reset()
    env.config.DISPLAY_IMAGE = True
    
    while True:
        frames = env.render()
        import cv2
        if frames and len(frames) > 0:
            combined = np.hstack(frames)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imshow("Cameras", combined_bgr)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('w'):
            env.step([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        elif key == ord('s'):
            env.step([0.0, 0.0, -0.1, 0.0, 0.0, 0.0])
        elif key == ord('a'):
            env.step([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif key == ord('d'):
            env.step([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif key == ord('e'):
            env.step([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        elif key == ord('r'):
            env.step([0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
        elif key == ord('i'):
            env.step([0.0, 0.0, 0.0, 0.0, 0.0, -0.1])
        elif key == ord('k'):
            env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])

        env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    env.close()