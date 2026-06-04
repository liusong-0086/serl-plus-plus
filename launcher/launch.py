import torch
import torch.nn as nn
from agentlace.trainer import TrainerConfig

from launcher.common.wandb import WandBLogger
from launcher.agents.sac import SACAgent
from launcher.vision.augmentations import make_batch_augmentation_func


def make_sac_pixel_agent(
    seed: int,
    sample_obs: dict,
    sample_action: torch.Tensor,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet18-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
) -> SACAgent:
    torch.manual_seed(seed)

    agent = SACAgent.create_pixels(
        sample_obs=sample_obs,
        sample_action=sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589) -> TrainerConfig:
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
) -> WandBLogger:
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": project,
        "exp_descriptor": description,
        "tag": description,
    })
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
