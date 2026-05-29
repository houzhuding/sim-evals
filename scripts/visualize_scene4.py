#!/usr/bin/env python3
"""Visualize Scene 4 (peg-hole) in the DROID Isaac environment.

Usage:
  python scripts/visualize_scene4.py
  python scripts/visualize_scene4.py --headless
  python scripts/visualize_scene4.py --steps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DROID Scene 4 (peg-hole)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, cuda:0)")
    parser.add_argument("--headless", action="store_true", help="Save one stitched image instead of launching viewer")
    parser.add_argument("--steps", type=int, default=0, help="Steps to run (0 = run until viewer is closed)")
    parser.add_argument("--cam-width", type=int, default=640, help="Camera width")
    parser.add_argument("--cam-height", type=int, default=360, help="Camera height")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("viz_output/isaac_scene_test"),
        help="Output directory for headless image",
    )
    args, _ = parser.parse_known_args()
    return args


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


def configure_camera_resolution(env_cfg, width: int, height: int) -> None:
    for cam_name in ("external_cam", "external_cam_2", "wrist_cam"):
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.width = width
        cam_cfg.height = height


def hold_current_pose_action(obs: dict) -> torch.Tensor:
    policy_obs = obs["policy"]
    arm = policy_obs["arm_joint_pos"].detach().cpu().float()
    gripper = policy_obs["gripper_pos"].detach().cpu().float()
    return torch.cat([arm, gripper], dim=0)[None]


def to_uint8_rgb(obs: dict, cam_name: str) -> np.ndarray:
    image = obs["policy"][cam_name][0].detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(description="Scene 4 Isaac app launcher")
    AppLauncher.add_app_launcher_args(app_parser)
    args_cli, _ = app_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    args_cli.device = resolve_device(args.device)
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
    env_cfg.set_scene(4)

    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()

    print("Loaded Scene 4 (peg-hole).")

    try:
        if args.headless:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / "scene4_stitched.png"
            frames = [
                to_uint8_rgb(obs, "external_cam"),
                to_uint8_rgb(obs, "external_cam_2"),
                to_uint8_rgb(obs, "wrist_cam"),
            ]
            stitched = np.concatenate(frames, axis=1)
            ok = cv2.imwrite(str(out_path), cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write stitched image to {out_path}")
            print(f"Saved {out_path}")
            return

        print("Viewer mode: interact with the Isaac Sim window. Close it to exit.")
        step = 0
        while simulation_app.is_running():
            if args.steps > 0 and step >= args.steps:
                break

            action = hold_current_pose_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
                obs, _ = env.reset()

            step += 1
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
