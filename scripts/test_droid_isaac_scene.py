#!/usr/bin/env python3
"""Save a stitched DROID Isaac scene image without connecting to a policy server."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DROID Isaac scene smoke test")
    parser.add_argument("--scene", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=360)
    parser.add_argument("--output-dir", type=Path, default=Path("viz_output/isaac_scene_test"))
    args, _ = parser.parse_known_args()
    return args


def to_uint8_rgb(obs: dict, cam_name: str) -> np.ndarray:
    image = obs["policy"][cam_name][0].detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def hold_current_pose_action(obs: dict) -> torch.Tensor:
    policy_obs = obs["policy"]
    arm = policy_obs["arm_joint_pos"].detach().cpu().float()
    gripper = policy_obs["gripper_pos"].detach().cpu().float()
    return torch.cat([arm, gripper], dim=0)[None]


def configure_camera_resolution(env_cfg, width: int, height: int) -> None:
    for cam_name in ("external_cam", "external_cam_2", "wrist_cam"):
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.width = width
        cam_cfg.height = height


def resolve_device(requested_device: str) -> str:
    if not requested_device.startswith("cuda"):
        return requested_device

    if not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    device_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and device_arch not in supported_arches:
        print(
            f"[WARN] CUDA arch {device_arch} is not supported by this PyTorch build "
            f"({sorted(supported_arches)}). Falling back to CPU."
        )
        return "cpu"

    return requested_device


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    out_path = output_dir / f"scene{args.scene}_stitched.png"
    print(f"Will save stitched image to: {out_path}")

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(description="DROID Isaac app launcher")
    AppLauncher.add_app_launcher_args(app_parser)
    args_cli, _ = app_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    args_cli.device = resolve_device(args_cli.device)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import cv2
    import gymnasium as gym
    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    configure_camera_resolution(env_cfg, args.cam_width, args.cam_height)
    env_cfg.set_scene(args.scene)

    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_names = ("external_cam", "external_cam_2", "wrist_cam")

    try:
        action = hold_current_pose_action(obs)
        obs, _, _, _, _ = env.step(action)

        frames = [to_uint8_rgb(obs, cam_name) for cam_name in camera_names]
        stitched = np.concatenate(frames, axis=1)
        ok = cv2.imwrite(str(out_path), cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))
        if not ok or not out_path.exists():
            raise RuntimeError(f"Failed to write stitched image to {out_path}")
        print(f"Saved stitched camera image to {out_path}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
