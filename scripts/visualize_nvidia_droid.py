"""
Visualize the NVIDIA DROID (Isaac Lab) environment.

Usage:
  python scripts/visualize_nvidia_droid.py                  # interactive viewer
  python scripts/visualize_nvidia_droid.py --scene 2        # different scene
  python scripts/visualize_nvidia_droid.py --headless       # save camera images to disk
  python scripts/visualize_nvidia_droid.py --steps 100      # run with random actions
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Visualize NVIDIA DROID environment")
parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, cuda:0)")
parser.add_argument("--scene", type=int, default=1, help="Scene ID (1-3)")
parser.add_argument("--headless", action="store_true", help="Headless mode: save camera images instead of viewer")
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run (0 = run until you close the window)")
parser.add_argument("--output_dir", type=str, default="viz_output", help="Output directory for camera images")
parser.add_argument("--cam_width", type=int, default=640, help="Camera width for rendered observations")
parser.add_argument("--cam_height", type=int, default=360, help="Camera height for rendered observations")
parser.add_argument(
    "--display_camera",
    type=str,
    default="external_cam",
    choices=["external_cam", "external_cam_2", "wrist_cam", "all"],
    help="Which camera to display in OpenCV window",
)
args, unknown = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_parser = argparse.ArgumentParser(description="DROID visualization")
AppLauncher.add_app_launcher_args(app_parser)
args_cli, _ = app_parser.parse_known_args()


def _resolve_device(requested_device: str) -> str:
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


args_cli.device = _resolve_device(args.device)
args_cli.enable_cameras = True
args_cli.headless = args.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sim_evals.environments  # registers DROID
import gymnasium as gym
from sim_evals.environments.droid_environment import EnvCfg


def _opencv_gui_available() -> bool:
    try:
        cv2.namedWindow("__cv2_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv2_test__")
        return True
    except cv2.error:
        return False


def _to_uint8_rgb(obs, cam_name: str) -> np.ndarray:
    img = obs["policy"][cam_name].squeeze(0).cpu().numpy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def main():
    env_cfg = EnvCfg()
    env_cfg.sim.device = args_cli.device

    # Lower camera resolution to reduce CPU/GPU memory use during visualization.
    for cam_name in ["external_cam", "external_cam_2", "wrist_cam"]:
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.width = args.cam_width
        cam_cfg.height = args.cam_height

    env_cfg.set_scene(args.scene)

    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()  # second render cycle for material loading (matching run_eval.py)

    print(f"\nEnvironment: DROID (Isaac Lab)")
    print(f"Scene: {args.scene}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Robot: Franka Panda + Robotiq 2F-85 Gripper")
    print(f"Cameras:")
    print(f"  - external_cam (right side) [{args.cam_width}x{args.cam_height}]")
    print(f"  - external_cam_2 (left side) [{args.cam_width}x{args.cam_height}]")
    print(f"  - wrist_cam (on gripper) [{args.cam_width}x{args.cam_height}]")

    camera_names = ["external_cam", "external_cam_2", "wrist_cam"]

    if args.headless:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_name in camera_names:
            (output_dir / cam_name).mkdir(parents=True, exist_ok=True)

        print(f"\nHeadless mode: saving camera images to {output_dir}/")

        for cam_name in camera_names:
            img = _to_uint8_rgb(obs, cam_name)
            save_path = output_dir / f"{cam_name}_scene{args.scene}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  Saved {save_path} ({img.shape[1]}x{img.shape[0]})")

    use_cv2_window = (not args.headless) and _opencv_gui_available()
    if not args.headless:
        if use_cv2_window:
            print(f"\nStreaming {args.display_camera}. Press 'q' or ESC to quit.")
        else:
            print("\nOpenCV GUI is unavailable in this environment. Continuing with Isaac viewer only.")

    step = 0
    while simulation_app.is_running():
        if args.steps > 0 and step >= args.steps:
            break

        action = torch.zeros(1, 8)
        obs, reward, terminated, truncated, _ = env.step(action)

        if args.headless:
            for cam_name in camera_names:
                img = _to_uint8_rgb(obs, cam_name)
                save_path = output_dir / cam_name / f"step_{step:04d}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        elif use_cv2_window:
            try:
                if args.display_camera == "all":
                    frames = [_to_uint8_rgb(obs, cam_name) for cam_name in camera_names]
                    display_img = np.concatenate(frames, axis=1)
                    window_title = "Cameras: external | external_2 | wrist"
                else:
                    display_img = _to_uint8_rgb(obs, args.display_camera)
                    window_title = f"Camera: {args.display_camera}"

                cv2.imshow(window_title, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            except cv2.error:
                use_cv2_window = False
                print("[WARN] OpenCV GUI failed at runtime. Continuing with Isaac viewer only.")

        if terminated or truncated:
            obs, _ = env.reset()
            obs, _ = env.reset()

        step += 1

    print("\nDone!")
    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()