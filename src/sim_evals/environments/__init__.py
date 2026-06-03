from __future__ import annotations

import os

import gymnasium as gym


def _registered(env_id: str) -> bool:
    return env_id in gym.envs.registry


def register_isaac() -> None:
    if _registered("DROID"):
        return

    from .droid_environment import EnvCfg as DroidEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv

    gym.register(
        id="DROID",
        entry_point=ManagerBasedRLEnv,
        kwargs={
            "env_cfg_entry_point": DroidEnvCfg,
        },
        disable_env_checker=True,
    )


def register_mujoco() -> None:
    if _registered("DROID_MUJOCO"):
        return

    from .mujoco_droid import MujocoDroidEnv, MujocoDroidEnvCfg

    gym.register(
        id="DROID_MUJOCO",
        entry_point=MujocoDroidEnv,
        kwargs={
            "cfg": MujocoDroidEnvCfg(),
        },
        disable_env_checker=True,
    )


try:
    register_isaac()
except Exception:
    pass

if os.environ.get("SIM_EVALS_REGISTER_MUJOCO") == "1":
    register_mujoco()
