import gymnasium as gym

try:
    from .mujoco_droid import MujocoDroidEnv, MujocoDroidEnvCfg
except Exception as e:  # pragma: no cover - optional dependency guard
    MujocoDroidEnv = None
    MujocoDroidEnvCfg = None
    print(f"[DEBUG] Failed to import MujocoDroidEnv or MujocoDroidEnvCfg: {e}")

try:
    from .droid_environment import EnvCfg as DroidEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
except Exception:  # pragma: no cover - optional dependency guard
    DroidEnvCfg = None
    ManagerBasedRLEnv = None

if DroidEnvCfg is not None and ManagerBasedRLEnv is not None:
    gym.register(
        id="DROID",
        entry_point=ManagerBasedRLEnv,
        kwargs={
            "env_cfg_entry_point": DroidEnvCfg,
        },
        disable_env_checker=True,
    )

if MujocoDroidEnv is None or MujocoDroidEnvCfg is None:
    print("[DEBUG] MujocoDroidEnv or MujocoDroidEnvCfg import failed.")
else:
    print("[DEBUG] MujocoDroidEnv and MujocoDroidEnvCfg imported successfully.")

if MujocoDroidEnv is not None and MujocoDroidEnvCfg is not None:
    print("[DEBUG] Registering DROID_MUJOCO environment.")
    gym.register(
        id="DROID_MUJOCO",
        entry_point=MujocoDroidEnv,
        kwargs={
            "cfg": MujocoDroidEnvCfg(),
        },
        disable_env_checker=True,
    )
