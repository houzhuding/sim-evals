from __future__ import annotations


# Use external MJCF file for the robot and scene
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch


# Load MJCF XML from external file
_MJCF_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "MJCF" / "my_droid_fixed.xml"
with open(_MJCF_PATH, "r") as f:
    _MJCF = f.read()


@dataclass
class MujocoDroidEnvCfg:
    width: int = 1280
    height: int = 720
    frame_skip: int = 8
    episode_length_s: float = 30.0
    scene: int = 1

    def set_scene(self, scene_name: int | str) -> None:
        self.scene = int(scene_name)


class MujocoDroidEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, cfg: MujocoDroidEnvCfg | None = None):
        self.cfg = cfg or MujocoDroidEnvCfg()

        self.model = mujoco.MjModel.from_xml_string(_MJCF)
        self.model.opt.timestep = 1.0 / (15.0 * float(self.cfg.frame_skip))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=self.cfg.width, height=self.cfg.height)

        # Cameras matching nvidia_droid environment config (OpenGL convention -> wxyz)
        self.renderer.add_camera(
            name="external_cam", height=720, width=1280,
            pos=(0.05, 0.57, 0.66),
            quat=(0.805, -0.393, -0.195, 0.399),
        )
        self.renderer.add_camera(
            name="external_cam_2", height=720, width=1280,
            pos=(0.05, -0.57, 0.66),
            quat=(-0.393, 0.805, 0.399, -0.195),
        )
        self.renderer.add_camera(
            name="wrist_cam", height=720, width=1280,
            pos=(0.011, -0.031, -0.074),
            quat=(-0.409, -0.420, 0.570, 0.576),
        )

        self.default_qpos = np.array(
            [
                0.0,
                -1.0 / 5.0 * np.pi,
                0.0,
                -4.0 / 5.0 * np.pi,
                0.0,
                3.0 / 5.0 * np.pi,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        self.max_episode_length = int(self.cfg.episode_length_s * 15)
        self.step_count = 0
        self._active_scene = int(self.cfg.scene)

        self.action_space = spaces.Box(
            low=np.array([-np.pi] * 7 + [0.0], dtype=np.float32),
            high=np.array([np.pi] * 7 + [1.0], dtype=np.float32),
            shape=(8,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "policy": spaces.Dict(
                    {
                        "arm_joint_pos": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
                        "gripper_pos": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                        "external_cam": spaces.Box(
                            low=0,
                            high=255,
                            shape=(1, self.cfg.height, self.cfg.width, 3),
                            dtype=np.uint8,
                        ),
                        "external_cam_2": spaces.Box(
                            low=0,
                            high=255,
                            shape=(1, self.cfg.height, self.cfg.width, 3),
                            dtype=np.uint8,
                        ),
                        "wrist_cam": spaces.Box(
                            low=0,
                            high=255,
                            shape=(1, self.cfg.height, self.cfg.width, 3),
                            dtype=np.uint8,
                        ),
                    }
                )
            }
        )

        self._scene_geom_ids = {
            1: ["cube", "bowl"],
            2: ["can", "mug"],
            3: ["banana", "bin"],
        }
        self._all_scene_geoms = ["cube", "bowl", "can", "mug", "banana", "bin"]

        self._reset_state()

    def set_scene(self, scene_name: int | str) -> None:
        self._active_scene = int(scene_name)
        self._apply_scene_visibility()

    def _reset_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:8] = self.default_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_qpos
        self._apply_scene_visibility()
        mujoco.mj_forward(self.model, self.data)

    def _apply_scene_visibility(self) -> None:
        scene_geoms = set(self._scene_geom_ids.get(self._active_scene, self._all_scene_geoms))
        for geom_name in self._all_scene_geoms:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                continue
            self.model.geom_rgba[geom_id, 3] = 1.0 if geom_name in scene_geoms else 0.0

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def _get_obs(self) -> dict[str, Any]:
        arm_qpos = self.data.qpos[:7].astype(np.float32)
        gripper = np.array([self.data.qpos[7] / (np.pi / 4.0)], dtype=np.float32)

        external_cam = self._render_camera("external_cam")
        external_cam_2 = self._render_camera("external_cam_2")
        wrist_cam = self._render_camera("wrist_cam")

        return {
            "policy": {
                "arm_joint_pos": torch.from_numpy(arm_qpos),
                "gripper_pos": torch.from_numpy(np.clip(gripper, 0.0, 1.0)),
                "external_cam": torch.from_numpy(external_cam[None]),
                "external_cam_2": torch.from_numpy(external_cam_2[None]),
                "wrist_cam": torch.from_numpy(wrist_cam[None]),
            }
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "scene" in options:
            self.set_scene(options["scene"])
        self.step_count = 0
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        target_qpos = np.concatenate(
            [
                action[:7].astype(np.float64),
                np.array([float(np.clip(action[7], 0.0, 1.0)) * (np.pi / 4.0)], dtype=np.float64),
            ]
        )
        self.data.ctrl[:] = target_qpos

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_length
        terminated = False
        reward = 0.0
        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None
