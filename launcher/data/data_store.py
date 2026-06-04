from threading import Lock
from typing import Iterable

import gymnasium as gym
from launcher.data.replay_buffer import ReplayBuffer
from launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        device: str = "cpu"
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity, device=device)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError("TODO")


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        device: str = "cpu",
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, 
            observation_space, 
            action_space, 
            capacity, 
            pixel_keys=image_keys, 
            device=device,
            **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    def latest_data_id(self) -> int:
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError("TODO")