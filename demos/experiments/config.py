from abc import abstractmethod
from typing import List


class DefaultTrainingConfig:
    """Default training configuration."""

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.97

    max_steps: int = 1000000
    replay_buffer_capacity: int = 100000

    random_steps: int = 500
    training_starts: int = 500
    steps_per_update: int = 30

    log_period: int = 100
    eval_period: int = 2000

    encoder_type: str = "resnet18-pretrained"
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    setup_mode: str = "single-arm-fixed-gripper"

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError

    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
