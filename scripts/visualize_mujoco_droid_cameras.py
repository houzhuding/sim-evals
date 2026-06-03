
# Ensure environment registration before anything else
import sim_evals.environments as sim_envs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

sim_envs.register_mujoco()

def main():
    env = gym.make("DROID_MUJOCO")
    obs, _ = env.reset()

    # --- Show interactive MuJoCo viewer ---
    mj_env = env.unwrapped
    try:
        import mujoco.viewer
        print("Launching MuJoCo interactive viewer. Close the viewer window to continue...")
        mujoco.viewer.launch(mj_env.model, mj_env.data)
    except Exception as e:
        print("Could not launch mujoco.viewer (headless or missing dependency):", e)

    # --- Show camera images ---
    # Try to find all available cameras in the model
    cam_names = []
    for i in range(mj_env.model.ncam):
        name = mujoco.mj_id2name(mj_env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        cam_names.append(name)

    imgs = []
    for name in cam_names:
        try:
            mj_env.renderer.update_scene(mj_env.data, camera=name)
            img = mj_env.renderer.render().copy()
            imgs.append((name, img))
        except Exception as e:
            print(f"Could not render camera {name}: {e}")

    if imgs:
        fig, axs = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 6))
        if len(imgs) == 1:
            axs = [axs]
        for ax, (name, img) in zip(axs, imgs):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("No cameras found in MJCF model.")
    env.close()

if __name__ == "__main__":
    main()
