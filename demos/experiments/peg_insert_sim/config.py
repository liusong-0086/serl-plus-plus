import numpy as np

from infra.wrappers.robot_pose import Quat2RotvecWrapper
from infra.wrappers.relative_frame import RelativeFrame
from infra.wrappers.intervention import SpacemouseIntervention
from infra.sim.envs.panda_insert_gym_env import DefaultEnvConfig

from launcher.wrappers.serl_obs import SERLObsWrapper
from launcher.wrappers.chunking import ChunkingWrapper

from demos.experiments.config import DefaultTrainingConfig
from infra.sim.envs.panda_insert_gym_env import PandaPegInsertGymEnv

class EnvConfig(DefaultEnvConfig):
    REALSENSE_CAMERAS = {
        "wrist1": {
            "serial_number": "230322276285",
            "dim": (640, 480),
            "exposure": 40000,
        },
        "wrist2": {
            "serial_number": "323622273011",
            "dim": (640, 480),
            "exposure": 40000,
        },
    }

    TARGET_POSE: np.ndarray = [0.5, 0.0, 0.16, -np.pi, 0, 0] # euler
    REWARD_THRESHOLD: np.ndarray = [0.015, 0.015, 0.015, 0.2, 0.2, 0.2]
    ACTION_SCALE = [0.01, 0.02]
    RESET_POSE = [0.5, 0.0, 0.26, -np.pi, 0, 0] # euler
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RX_RANGE = 0.0
    RANDOM_RY_RANGE = 0.0
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([RANDOM_XY_RANGE, RANDOM_XY_RANGE, 0.0, RANDOM_RX_RANGE, RANDOM_RY_RANGE, RANDOM_RZ_RANGE])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([RANDOM_XY_RANGE, RANDOM_XY_RANGE, 0.1, RANDOM_RX_RANGE, RANDOM_RY_RANGE, RANDOM_RZ_RANGE])
    MAX_EPISODE_LENGTH: int = 100
    DISPLAY_IMAGE: bool = True


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist1", "wrist2"]
    classifier_keys = ["wrist1", "wrist2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet18-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        render_mode = "rgb_array" if fake_env else "human"
        env = PandaPegInsertGymEnv(
            config=EnvConfig(),
            render_mode=render_mode,
        )

        # env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2RotvecWrapper(env)   
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        return env
