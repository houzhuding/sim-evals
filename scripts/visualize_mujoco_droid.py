"""
Visualize the DROID environment (MuJoCo backend).

Usage:
  python scripts/visualize_mujoco_droid.py                     # save camera images
  python scripts/visualize_mujoco_droid.py --scene 2           # different scene
  python scripts/visualize_mujoco_droid.py --steps 100         # run with random actions
  python scripts/visualize_mujoco_droid.py --viewer            # launch interactive viewer
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Visualize DROID MuJoCo environment")
parser.add_argument("--scene", type=int, default=1, help="Scene ID (1-3)")
parser.add_argument("--steps", type=int, default=0, help="Number of random action steps (0 = just render)")
parser.add_argument("--viewer", action="store_true", help="Launch interactive MuJoCo viewer")
args = parser.parse_args()

import sim_evals.environments as sim_envs
import gymnasium as gym
import numpy as np
import cv2

sim_envs.register_mujoco()
env = gym.make("DROID_MUJOCO")
env.unwrapped.cfg.scene = args.scene
obs, _ = env.reset()

print(f"\nDROID MuJoCo Environment")
print(f"Scene: {args.scene}")
print(f"Robot: Franka Panda + Robotiq 2F-85 Gripper")
print(f"Cameras: external_cam, external_cam_2, wrist_cam")
print(f"Episode length: {env.unwrapped.cfg.episode_length_s}s")

mj_env = env.unwrapped

if args.viewer:
    try:
        import mujoco.viewer
        print("\nLaunching interactive MuJoCo viewer. Close window to continue...")
        mujoco.viewer.launch(mj_env.model, mj_env.data)
    except Exception as e:
        print(f"Could not launch viewer: {e}")

# Save camera images
output_dir = Path("viz_output")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving camera images to {output_dir}/")

for cam_name in ["external_cam", "external_cam_2", "wrist_cam"]:
    img = obs["policy"][cam_name].squeeze(0).cpu().numpy()
    save_path = output_dir / f"mujoco_{cam_name}_scene{args.scene}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  {save_path} ({img.shape[1]}x{img.shape[0]})")

if args.steps > 0:
    print(f"\nRunning {args.steps} steps with random actions...")
    for step in range(args.steps):
        action = np.random.randn(8) * 0.05
        action[7] = np.clip(action[7], 0, 1)
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))

        if step % 10 == 0:
            ext_img = obs["policy"]["external_cam"].squeeze(0).cpu().numpy()
            save_path = output_dir / f"step_{step:04d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(ext_img, cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

env.close()
print("Done!")
