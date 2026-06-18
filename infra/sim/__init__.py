from gymnasium.envs.registration import register

register(
    id="PandaPegInsertVision-v0",
    entry_point="infra.sim.envs:PandaPegInsertGymEnv"
)

register(
     id="PandaPegInsertDepth-v0",
    entry_point="infra.sim.envs:PandaPegInsertDepthGymEnv"
)