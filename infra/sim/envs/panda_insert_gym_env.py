from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np

import cv2
from scipy.spatial.transform import Rotation
from infra.sim.controllers.impedance import impedance_control
from infra.sim.envs.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "assets" / "panda_peg_insert.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))

class DefaultEnvConfig:
    REALSENSE_CAMERAS: Dict = {
        "wrist1": "230322271990",
        "wrist2": "230322272626",
    }
    TARGET_POSE: np.ndarray = [0.5, 0.0, 0.1, -np.pi, 0, 0] # euler
    REWARD_THRESHOLD: np.ndarray = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
    ACTION_SCALE = [0.1, 1]
    RESET_POSE = [0.5, 0.0, 0.3, -np.pi, 0, 0] # euler
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.1
    RANDOM_RX_RANGE = 0.0
    RANDOM_RY_RANGE = 0.0
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([RANDOM_XY_RANGE, RANDOM_XY_RANGE, 0.0, RANDOM_RX_RANGE, RANDOM_RY_RANGE, RANDOM_RZ_RANGE])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([RANDOM_XY_RANGE, RANDOM_XY_RANGE, 0.2, RANDOM_RX_RANGE, RANDOM_RY_RANGE, RANDOM_RZ_RANGE])
    MAX_EPISODE_LENGTH: int = 100
    DISPLAY_IMAGE: bool = True


class PandaPegInsertGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: list[GymRenderingSpec] = [GymRenderingSpec(camera_name="wrist1"),
                                               GymRenderingSpec(camera_name="wrist2")],
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        config: DefaultEnvConfig = None,
    ):
        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
            render_mode=render_mode,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self._action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.resetpos = np.concatenate([config.RESET_POSE[:3], 
                                        Rotation.from_euler("xyz", config.RESET_POSE[3:]).as_quat()])

        self.render_mode = render_mode

        self.config = config
        self.cur_episode_length = 0
        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("actuator8").id

        self._hand_site_id = self._model.site("hand_site").id
        self._end_effector_id = self._model.body("end_effector").mocapid.item()

        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in self.config.REALSENSE_CAMERAS}
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            np.ones((6,), dtype=np.float32) * -1,
            np.ones((6,), dtype=np.float32),
        )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.cur_episode_length = 0
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        reset_pose = self.resetpos.copy()

        if self.config.RANDOM_RESET:
            reset_pose[:2] += np.random.uniform(
                -self.config.RANDOM_XY_RANGE, self.config.RANDOM_XY_RANGE, (2,)
            )

            quat_reset = reset_pose[3:].copy()
            euler_delta = np.array([
                np.random.uniform(-self.config.RANDOM_RX_RANGE, self.config.RANDOM_RX_RANGE),
                np.random.uniform(-self.config.RANDOM_RY_RANGE, self.config.RANDOM_RY_RANGE),
                np.random.uniform(-self.config.RANDOM_RZ_RANGE, self.config.RANDOM_RZ_RANGE)
            ])
            reset_pose[3:] = (Rotation.from_euler("xyz", euler_delta) * Rotation.from_quat(quat_reset)).as_quat()

        reset_quat_scipy = reset_pose[3:7]  # [x, y, z, w]  
        reset_pose[3:7] = np.array([reset_quat_scipy[3], reset_quat_scipy[0], reset_quat_scipy[1], reset_quat_scipy[2]])  # [w, x, y, z]
        self._reset_arm_to_home(reset_pose)
        self._data.mocap_pos[self._end_effector_id] = reset_pose[:3]
        self._data.mocap_quat[self._end_effector_id] = reset_pose[3:7]
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        cur_pos = self._data.mocap_pos[self._end_effector_id]
        cur_quat_mujoco = self._data.mocap_quat[self._end_effector_id]
        cur_quat = [cur_quat_mujoco[1], cur_quat_mujoco[2], cur_quat_mujoco[3], cur_quat_mujoco[0]]

        next_pos = cur_pos + xyz_delta * self._action_scale[0]
        next_quat = (Rotation.from_rotvec(action[3:6] * self._action_scale[1])
                    * Rotation.from_quat(cur_quat)).as_quat()
        next_pos_quat_clip = self._clip_safety_box(np.concatenate([next_pos, next_quat]))
        next_quat_mujoco = [next_pos_quat_clip[6], next_pos_quat_clip[3], next_pos_quat_clip[4], next_pos_quat_clip[5]]

        self._data.mocap_pos[self._end_effector_id] = next_pos_quat_clip[:3]
        self._data.mocap_quat[self._end_effector_id] = next_quat_mujoco

        self.cur_episode_length += 1
        self._servo_Impedance_pose(self._data.mocap_pos[self._end_effector_id], self._data.mocap_quat[self._end_effector_id], num_steps=self._n_substeps)

        obs = self._compute_observation()
        reward = self._compute_reward()
        done = self.cur_episode_length >= self.config.MAX_EPISODE_LENGTH or reward

        if self.render_mode == "human":
            self._viewer.sync()
        
        return obs, int(reward), done, False, {"succeed": reward}

    def _reset_arm_to_home(self, tcp_pos=None):
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qvel[self._panda_dof_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._servo_Impedance_pose(tcp_pos[:3], tcp_pos[3:7])

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        cur_pos = self._data.site_xpos[self._hand_site_id].copy()
        cur_xmat = self._data.site_xmat[self._hand_site_id].reshape((3, 3))
        cur_rot = Rotation.from_matrix(cur_xmat)
        obs["state"]["tcp_pose"] = np.concatenate([cur_pos, cur_rot.as_quat()])

        obs["state"]["tcp_vel"] = self._get_site_twist(self._model, self._data, self._hand_site_id) 
        obs["state"]["tcp_force"] = self._data.sensor("panda/end_effector_force").data
        obs["state"]["tcp_torque"] = self._data.sensor("panda/end_effector_torque").data

        obs["images"] = {}
        obs["images"]["wrist1"], obs["images"]["wrist2"] = self.render()

        return obs

    def _compute_reward(self) -> bool:
        cur_pos = self._data.site_xpos[self._hand_site_id].copy()
        cur_xmat = self._data.site_xmat[self._hand_site_id].reshape((3, 3))
        
        cur_rot = Rotation.from_matrix(cur_xmat)
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot =  target_rot @ cur_rot.as_matrix().T
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([cur_pos - self._TARGET_POSE[:3], diff_euler]))
        
        return np.all(delta < self._REWARD_THRESHOLD)
    
    def _servo_Impedance_pose(self, target_pos, target_quat, num_steps=2000):
        for _ in range(num_steps):
            tau = impedance_control(
                model=self._model,
                data=self._data,
                site_id=self._hand_site_id,
                dof_ids=self._panda_dof_ids,
                pos=target_pos,
                ori=target_quat,
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)
            self._data.ctrl[self._gripper_ctrl_id] = 0.0

        mujoco.mj_forward(self._model, self._data)

    def _get_site_twist(self,model, data, site_id, local=False):
        vel = np.zeros((6, 1), dtype=np.float64)  
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, local)
        return vel.reshape(6,)

    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        
        delta_R = Rotation.from_quat(pose[3:]) * Rotation.from_euler("xyz", self._TARGET_POSE[3:]).inv()
        delta_euler = delta_R.as_euler("xyz")
        delta_euler = np.clip(
            delta_euler,
            [-self.config.RANDOM_RX_RANGE, -self.config.RANDOM_RY_RANGE, -self.config.RANDOM_RZ_RANGE],
            [self.config.RANDOM_RX_RANGE, self.config.RANDOM_RY_RANGE, self.config.RANDOM_RZ_RANGE]
        )
        pose[3:] = (Rotation.from_euler("xyz", delta_euler) * Rotation.from_euler("xyz", self._TARGET_POSE[3:])).as_quat()

        return pose


if __name__ == "__main__":
    env = PandaPegInsertGymEnv(render_mode="human", config=DefaultEnvConfig())
    env.reset()
    
    while True:
        frames = env.render()
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
