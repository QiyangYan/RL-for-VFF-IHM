# noqa: D104
from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        register(
            id=f"VariableFriction{suffix}-v1",
            entry_point="gymnasium_robotics.envs.training1.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v2",
            entry_point="gymnasium_robotics.envs.training2.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v3",
            entry_point="gymnasium_robotics.envs.training3.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v4",
            entry_point="gymnasium_robotics.envs.training4.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v5",
            entry_point="gymnasium_robotics.envs.real4.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v6",
            entry_point="gymnasium_robotics.envs.training5.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v7",
            entry_point="gymnasium_robotics.envs.diffusion1.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v8",
            entry_point="gymnasium_robotics.envs.diffusion2.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id=f"VariableFriction{suffix}-v9",
            entry_point="gymnasium_robotics.envs.diffusion3_cross_object.manipulate_block:MujocoHandBlockEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

__version__ = "1.2.4"


try:
    import sys

    from farama_notifications import notifications

    if (
        "gymnasium_robotics" in notifications
        and __version__ in notifications["gymnasium_robotics"]
    ):
        print(notifications["gymnasium_robotics"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
