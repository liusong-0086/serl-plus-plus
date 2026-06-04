import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and optionally the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # Check if environment has images
        has_images = "images" in self.env.observation_space.spaces

        if has_images:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                    **(self.env.observation_space["images"]),
                }
            )
        else:
            # State-only mode
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                }
            )
        self.has_images = has_images

    def observation(self, obs):
        obs_dict = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
        }
        
        # Add images if they exist
        if self.has_images and "images" in obs:
            obs_dict.update(obs["images"])
        
        return obs_dict

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info