import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_name: str = "front"
    mode: Literal["rgb_array", "depth_array", "rgbd_tuple"] = "rgb_array"

class MujocoGymEnv(gym.Env):
    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: list[GymRenderingSpec] = None,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._render_specs = render_spec
        
        self._renderers = [MujocoRenderer(self._model, self._data, camera_name=item.camera_name, width=item.width, 
                                          height=item.height) for item in self._render_specs]
        self._viewer = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def render(self):
        rendered_frames = []
        for i, render_spec in enumerate(self._render_specs):
            # For open3d point cloud viewer (OpenGL backend conflict)
            viewer = self._renderers[i].viewer
            if viewer is not None and hasattr(viewer, "make_context_current"):
                viewer.make_context_current()
            rendered_frames.append(self._renderers[i].render(render_mode=render_spec.mode))
        return rendered_frames

    def close(self) -> None:
        for renderer in self._renderers:
            renderer.close()
        if self._viewer is not None:
            self._viewer.close()

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random