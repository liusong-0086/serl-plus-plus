from gymnasium.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="infra.sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="infra.sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPegInsertVision-v0",
    entry_point="infra.sim.envs:PandaPegInsertGymEnv"
)
