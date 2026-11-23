"""
Gymnasium reinforcement learning environments for attitude stabilization.
"""

from gymnasium.envs.registration import register
from src.envs.spacecraft_env import SpacecraftAttitudeEnv

# Register nominal 3D environment with Gymnasium
register(
    id="SpacecraftAttitude-v0",
    entry_point="src.envs.spacecraft_env:SpacecraftAttitudeEnv",
    kwargs={"include_integral": False},
    max_episode_steps=3000,
)

# Register redesigned 4D environment with Gymnasium (Phase 4D)
register(
    id="SpacecraftAttitude4D-v0",
    entry_point="src.envs.spacecraft_env:SpacecraftAttitudeEnv",
    kwargs={"include_integral": True},
    max_episode_steps=3000,
)

__all__ = ["SpacecraftAttitudeEnv"]
