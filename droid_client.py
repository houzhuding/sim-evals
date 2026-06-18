#!/usr/bin/env python3
"""Run NVIDIA DROID simulation against a remote AR-DROID policy server.

The server is expected to expose the roboarena websocket interface configured by
``droid_server.py``: two external cameras, one wrist camera, joint position
state, and joint-position action chunks.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import importlib
import logging
import math
import shutil
import signal
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import websockets.sync.client

from openpi_client import msgpack_numpy


DEFAULT_PROMPTS = {
    1: "put the cube in the bowl",
    2: "put the can in the mug",
    3: "put banana in the bin",
    4: "pick up the blue peg and insert it into the orange hole, using small wiggle motions to align and seat it fully",
    5: "insert the blue block in the orange block-shaped hole",
}
RELATIVE_FRAME_OFFSETS = [-23, -16, -8, 0]
SCENE5_DEFAULT_HOLE_SIZE_M = 0.04
SCENE5_WALL_THICKNESS_M = 0.0075
SCENE5_BOTTOM_THICKNESS_M = 0.01


class SimDroidPolicyClient:
    """Small stateful client that buffers action chunks from the websocket server."""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        open_loop_horizon: int = 24,
        trace_dir: Path | None = None,
        lock_gripper_close: bool = False,
        latch_gripper_after_close: bool = False,
    ) -> None:
        self.client = RoboarenaWebsocketClient(host=host, port=port)
        self.prompt = prompt
        self.open_loop_horizon = int(open_loop_horizon)
        self.lock_gripper_close = bool(lock_gripper_close)
        self.latch_gripper_after_close = bool(latch_gripper_after_close)
        self.gripper_latched_closed = False
        self.session_id = str(uuid.uuid4())
        self.pred_action_chunk: np.ndarray | None = None
        self.actions_from_chunk_completed = 0
        self.obs_history: list[dict[str, Any]] = []
        self.chunk_index = -1
        self.last_chunk_infer_time_s = math.nan
        self.trace_dir = trace_dir
        self._trace_file = None
        self._trace_writer: csv.DictWriter | None = None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(self.trace_dir / "execution_trace.csv", "w", newline="")
            self._trace_writer = csv.DictWriter(
                self._trace_file,
                fieldnames=execution_trace_fieldnames(),
            )
            self._trace_writer.writeheader()

        metadata = self.client.get_server_metadata()
        logging.info("Server metadata: %s", metadata)
        self._validate_server_config(metadata)

    def _validate_server_config(self, server_config: dict[str, Any]) -> None:
        if server_config.get("n_external_cameras") != 2:
            raise ValueError(
                f"Expected server with 2 external cameras, got "
                f"{server_config.get('n_external_cameras')}"
            )
        if not server_config.get("needs_wrist_camera"):
            raise ValueError("Expected server with wrist camera enabled")
        if server_config.get("action_space") != "joint_position":
            raise ValueError(
                f"Expected joint_position action space, got "
                f"{server_config.get('action_space')}"
            )

    def reset(self) -> None:
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.obs_history = []
        self.session_id = str(uuid.uuid4())
        self.chunk_index = -1
        self.last_chunk_infer_time_s = math.nan
        self.gripper_latched_closed = False
        self.client.reset()

    def close(self) -> None:
        if self._trace_file is not None:
            self._trace_file.flush()
            self._trace_file.close()
            self._trace_file = None
        self.client.close()

    def infer(self, sim_obs: dict[str, Any]) -> dict[str, Any]:
        """Return one action for the current simulation observation."""
        pre_server_obs = extract_server_fields(sim_obs)
        self.obs_history.append(pre_server_obs)
        history_limit = max(abs(min(RELATIVE_FRAME_OFFSETS)) + 1, self.open_loop_horizon + 1)
        if len(self.obs_history) > history_limit:
            self.obs_history = self.obs_history[-history_limit:]
        if (
            self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= self.open_loop_horizon
            or self.actions_from_chunk_completed >= len(self.pred_action_chunk)
        ):
            request = make_server_observation(
                self._select_request_frames(),
                prompt=self.prompt,
                session_id=self.session_id,
            )
            t0 = time.time()
            response = self.client.infer(request)
            dt = time.time() - t0
            self.pred_action_chunk = extract_action_chunk(response)
            self.actions_from_chunk_completed = 0
            self.chunk_index += 1
            self.last_chunk_infer_time_s = dt
            logging.info(
                "Received action chunk %s in %.2fs, range [%.4f, %.4f]",
                self.pred_action_chunk.shape,
                dt,
                float(self.pred_action_chunk.min()),
                float(self.pred_action_chunk.max()),
            )

        horizon_index = self.actions_from_chunk_completed
        raw_action = self.pred_action_chunk[horizon_index].astype(np.float32)
        action = raw_action.copy()
        self.actions_from_chunk_completed += 1

        model_gripper_close = bool(action[-1] > 0.5)
        if self.lock_gripper_close:
            action[-1] = 1.0
            self.gripper_latched_closed = True
        elif self.latch_gripper_after_close:
            if model_gripper_close:
                self.gripper_latched_closed = True
            action[-1] = 1.0 if self.gripper_latched_closed else 0.0
        else:
            action[-1] = 1.0 if model_gripper_close else 0.0
        return {
            "action": action,
            "viz": make_viz_image(sim_obs),
            "trace": {
                "session_id": self.session_id,
                "chunk_index": self.chunk_index,
                "horizon_index": horizon_index,
                "chunk_infer_time_s": self.last_chunk_infer_time_s,
                "lock_gripper_close": self.lock_gripper_close,
                "latch_gripper_after_close": self.latch_gripper_after_close,
                "gripper_latched_closed": self.gripper_latched_closed,
                "pre_obs": pre_server_obs,
                "raw_action": raw_action,
                "executed_action": action,
            },
        }

    def record_execution(
        self,
        *,
        episode: int,
        env_step: int,
        trace: dict[str, Any],
        post_obs: dict[str, Any],
        step_wall_time_s: float,
    ) -> None:
        if self._trace_writer is None:
            return
        row = {
            "wall_time_s": time.time(),
            "episode": episode,
            "env_step": env_step,
            "session_id": trace["session_id"],
            "chunk_index": trace["chunk_index"],
            "horizon_index": trace["horizon_index"],
            "chunk_infer_time_s": trace["chunk_infer_time_s"],
            "step_wall_time_s": step_wall_time_s,
            "lock_gripper_close": bool(trace.get("lock_gripper_close", False)),
            "latch_gripper_after_close": bool(trace.get("latch_gripper_after_close", False)),
            "gripper_latched_closed": bool(trace.get("gripper_latched_closed", False)),
        }
        add_vector_fields(row, "pre_joint_q", trace["pre_obs"].get("observation/joint_position"), 7)
        add_vector_fields(row, "pre_gripper", trace["pre_obs"].get("observation/gripper_position"), 1)
        add_vector_fields(row, "pre_wrench", trace["pre_obs"].get("observation/wrist_wrench"), 6)
        add_vector_fields(row, "raw_action", trace["raw_action"], 8)
        add_vector_fields(row, "executed_action", trace["executed_action"], 8)
        post_server_obs = extract_server_fields(post_obs)
        add_vector_fields(row, "post_joint_q", post_server_obs.get("observation/joint_position"), 7)
        add_vector_fields(row, "post_gripper", post_server_obs.get("observation/gripper_position"), 1)
        add_vector_fields(row, "post_wrench", post_server_obs.get("observation/wrist_wrench"), 6)
        self._trace_writer.writerow(row)
        if self._trace_file is not None:
            self._trace_file.flush()

    def _select_request_frames(self) -> list[dict[str, Any]]:
        if self.pred_action_chunk is None:
            return [self.obs_history[-1]]

        anchor = len(self.obs_history) - 1
        return [
            self.obs_history[max(anchor + offset, 0)]
            for offset in RELATIVE_FRAME_OFFSETS
        ]


class RoboarenaWebsocketClient:
    """Minimal client for eval_utils.policy_server's endpoint-routed protocol."""

    def __init__(self, host: str, port: int) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def infer(self, obs: dict[str, Any]) -> Any:
        request = dict(obs)
        request["endpoint"] = "infer"
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        self._ws.send(self._packer.pack({"endpoint": "reset"}))
        try:
            response = self._ws.recv(timeout=5)
        except TimeoutError:
            logging.warning("Timed out waiting for reset acknowledgment; continuing.")
            return
        if isinstance(response, str):
            raise RuntimeError(f"Error resetting inference server:\n{response}")

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            logging.debug("Ignoring websocket close error", exc_info=True)

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=60,
                    ping_timeout=600,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)


def tensor_image_to_numpy(value: Any) -> np.ndarray:
    """Convert an Isaac/MuJoCo image tensor shaped (1, H, W, C) to uint8 RGB."""
    if is_torch_tensor(value):
        image = value[0].detach().cpu().numpy()
    else:
        image = np.asarray(value)[0]

    if image.dtype != np.uint8:
        image = np.asarray(image, dtype=np.float32)
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def tensor_state_to_numpy(value: Any) -> np.ndarray:
    if is_torch_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def vector_or_none(value: Any, width: int) -> np.ndarray | None:
    if value is None:
        return None
    array = tensor_state_to_numpy(value).reshape(-1)
    if array.size < width:
        padded = np.full(width, np.nan, dtype=np.float32)
        padded[: array.size] = array
        return padded
    return array[:width].astype(np.float32)


def add_vector_fields(row: dict[str, Any], prefix: str, value: Any, width: int) -> None:
    vector = vector_or_none(value, width)
    if vector is None:
        vector = np.full(width, np.nan, dtype=np.float32)
    for index in range(width):
        row[f"{prefix}_{index}"] = float(vector[index])


def execution_trace_fieldnames() -> list[str]:
    fields = [
        "wall_time_s",
        "episode",
        "env_step",
        "session_id",
        "chunk_index",
        "horizon_index",
        "chunk_infer_time_s",
        "step_wall_time_s",
        "lock_gripper_close",
        "latch_gripper_after_close",
        "gripper_latched_closed",
    ]
    for prefix, width in (
        ("pre_joint_q", 7),
        ("pre_gripper", 1),
        ("pre_wrench", 6),
        ("raw_action", 8),
        ("executed_action", 8),
        ("post_joint_q", 7),
        ("post_gripper", 1),
        ("post_wrench", 6),
    ):
        fields.extend(f"{prefix}_{index}" for index in range(width))
    return fields


def is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


def extract_server_fields(sim_obs: dict[str, Any]) -> dict[str, Any]:
    policy_obs = sim_obs["policy"]
    fields = {
        "observation/exterior_image_0_left": tensor_image_to_numpy(
            policy_obs["external_cam"]
        ),
        "observation/exterior_image_1_left": tensor_image_to_numpy(
            policy_obs["external_cam_2"]
        ),
        "observation/wrist_image_left": tensor_image_to_numpy(policy_obs["wrist_cam"]),
        "observation/joint_position": tensor_state_to_numpy(
            policy_obs["arm_joint_pos"]
        ),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": tensor_state_to_numpy(
            policy_obs["gripper_pos"]
        ),
    }
    for wrench_key in (
        "wrist_wrench",
        "wrist_force_torque",
        "wrist_ft",
        "force_torque",
    ):
        if wrench_key in policy_obs:
            wrench = vector_or_none(policy_obs[wrench_key], 6)
            if wrench is not None:
                fields["observation/wrist_wrench"] = wrench
            break
    return fields


def make_server_observation(
    frame_obs: list[dict[str, Any]],
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    current_obs = frame_obs[-1]
    request = {
        "endpoint": "infer",
        "observation/joint_position": current_obs["observation/joint_position"],
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": current_obs["observation/gripper_position"],
        "prompt": prompt,
        "session_id": session_id,
    }
    if "observation/wrist_wrench" in current_obs:
        request["observation/wrist_wrench"] = current_obs["observation/wrist_wrench"]

    for image_key in (
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        images = [obs[image_key] for obs in frame_obs]
        request[image_key] = images[0] if len(images) == 1 else np.stack(images, axis=0)

    return request


def extract_action_chunk(response: Any) -> np.ndarray:
    """Normalize supported server response shapes to (T, 8)."""
    if isinstance(response, dict):
        if "actions" in response:
            response = response["actions"]
        elif "action" in response:
            response = response["action"]

    actions = np.asarray(response, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2 or actions.shape[-1] != 8:
        raise ValueError(f"Expected action chunk shaped (T, 8), got {actions.shape}")
    return actions


def make_viz_image(sim_obs: dict[str, Any]) -> np.ndarray:
    policy_obs = sim_obs["policy"]
    frames = [
        tensor_image_to_numpy(policy_obs["external_cam"]),
        tensor_image_to_numpy(policy_obs["external_cam_2"]),
        tensor_image_to_numpy(policy_obs["wrist_cam"]),
    ]
    min_h = min(frame.shape[0] for frame in frames)
    if any(frame.shape[0] != min_h for frame in frames):
        import cv2

        frames = [
            cv2.resize(
                frame,
                (int(frame.shape[1] * min_h / frame.shape[0]), min_h),
                interpolation=cv2.INTER_AREA,
            )
            for frame in frames
        ]
    return np.concatenate(frames, axis=1)


def configure_camera_resolution(env_cfg: Any, width: int, height: int) -> None:
    for cam_name in ("external_cam", "external_cam_2", "wrist_cam"):
        cam_cfg = getattr(env_cfg.scene, cam_name, None)
        if cam_cfg is None:
            continue
        cam_cfg.width = width
        cam_cfg.height = height


def resolve_hole_size_m(args: argparse.Namespace) -> float | None:
    if args.hole_size is not None and args.hole_size_mm is not None:
        raise ValueError("Use only one of --hole-size or --hole-size-mm.")

    if args.hole_size_mm is not None:
        hole_size_m = float(args.hole_size_mm) / 1000.0
    elif args.hole_size is not None:
        raw_size = float(args.hole_size)
        # The scene is authored in meters, but this task is usually described in mm.
        hole_size_m = raw_size / 1000.0 if raw_size > 1.0 else raw_size
    else:
        return None

    if not math.isfinite(hole_size_m) or hole_size_m <= 0:
        raise ValueError(f"Hole size must be positive, got {hole_size_m!r} m")
    if hole_size_m < 0.005 or hole_size_m > 0.20:
        raise ValueError(
            f"Hole size {hole_size_m:.4f} m is outside the expected range "
            "0.005-0.20 m."
        )
    return hole_size_m


def create_scene5_with_hole_size(hole_size_m: float) -> Path:
    """Create a temporary scene-5 USD with a square hole opening of A x A x A."""
    from pxr import Gf, Usd

    assets_dir = Path(__file__).resolve().parent / "assets"
    source_path = assets_dir / "scene5.usd"
    if not source_path.exists():
        source_path = assets_dir / "scene4.usd"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source scene USD: {source_path}")

    hole_size_mm = int(round(hole_size_m * 1000.0))
    output_path = Path(tempfile.gettempdir()) / f"sim_evals_scene5_hole_{hole_size_mm}mm.usd"
    shutil.copy2(source_path, output_path)

    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated scene: {output_path}")

    wall_t = SCENE5_WALL_THICKNESS_M
    bottom_t = SCENE5_BOTTOM_THICKNESS_M
    outer_size = hole_size_m + 2.0 * wall_t
    wall_center = hole_size_m / 2.0 + wall_t / 2.0
    wall_z = bottom_t + hole_size_m / 2.0

    def set_vec3(path: str, attr_name: str, value: tuple[float, float, float], vec_type: Any) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Missing prim in scene5 USD: {path}")
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            raise RuntimeError(f"Missing attribute {attr_name!r} on {path}")
        attr.Set(vec_type(*value))

    set_vec3("/World/hole/bottom", "xformOp:scale", (outer_size, outer_size, bottom_t), Gf.Vec3f)
    set_vec3("/World/hole/bottom", "xformOp:translate", (0.0, 0.0, bottom_t / 2.0), Gf.Vec3d)

    set_vec3("/World/hole/wall_pos_x", "xformOp:scale", (wall_t, outer_size, hole_size_m), Gf.Vec3f)
    set_vec3("/World/hole/wall_pos_x", "xformOp:translate", (wall_center, 0.0, wall_z), Gf.Vec3d)
    set_vec3("/World/hole/wall_neg_x", "xformOp:scale", (wall_t, outer_size, hole_size_m), Gf.Vec3f)
    set_vec3("/World/hole/wall_neg_x", "xformOp:translate", (-wall_center, 0.0, wall_z), Gf.Vec3d)

    set_vec3("/World/hole/wall_pos_y", "xformOp:scale", (hole_size_m, wall_t, hole_size_m), Gf.Vec3f)
    set_vec3("/World/hole/wall_pos_y", "xformOp:translate", (0.0, wall_center, wall_z), Gf.Vec3d)
    set_vec3("/World/hole/wall_neg_y", "xformOp:scale", (hole_size_m, wall_t, hole_size_m), Gf.Vec3f)
    set_vec3("/World/hole/wall_neg_y", "xformOp:translate", (0.0, -wall_center, wall_z), Gf.Vec3d)

    stage.Save()
    return output_path


def resolve_device(requested_device: str) -> str:
    if not requested_device.startswith("cuda"):
        return requested_device

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    device_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and device_arch not in supported_arches:
        logging.warning(
            "CUDA arch %s is not supported by this PyTorch build (%s). "
            "Falling back to CPU.",
            device_arch,
            sorted(supported_arches),
        )
        return "cpu"

    return requested_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NVIDIA DROID sim with a remote AR-DROID websocket policy."
    )
    parser.add_argument("--port", type=int, default=50532, help="Policy server port")
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollouts")
    parser.add_argument("--scene", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--hole-size",
        type=float,
        default=None,
        help=(
            "Scene-5 square hole size A. Values <= 1 are meters, values > 1 "
            "are interpreted as millimeters. Example: 0.06 or 60 for a 60 mm hole."
        ),
    )
    parser.add_argument(
        "--hole-size-mm",
        type=float,
        default=None,
        help="Scene-5 square hole size A in millimeters.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Language instruction. Defaults to the standard prompt for --scene.",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=24,
        help="Number of returned chunk actions to execute before querying again.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-episode step cap. Defaults to env max_episode_length.",
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=320,
        help="Rendered camera width sent to the policy server.",
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=180,
        help="Rendered camera height sent to the policy server.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory for rollout videos. Defaults to runs/<date>/<time>.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Directory for execution_trace.csv. Defaults to <video-dir>/trace.",
    )
    parser.add_argument(
        "--lock-gripper-close",
        action="store_true",
        help="Force the executed gripper action to close while preserving raw model output in logs.",
    )
    parser.add_argument(
        "--latch-gripper-after-close",
        action="store_true",
        help="Use model gripper output until first close command, then keep the gripper closed.",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable writing rollout videos.",
    )
    parser.add_argument(
        "--pre-infer-view-seconds",
        type=float,
        default=0.0,
        help=(
            "Before inference starts, keep the Isaac Sim viewer interactive for this many "
            "seconds so you can adjust camera/viewpoint (only when not headless)."
        ),
    )
    args, _ = parser.parse_known_args()
    return args


def wait_for_view_adjustment(simulation_app: Any, duration_s: float) -> None:
    """Keep the viewer responsive to allow manual camera adjustments before rollout."""
    if duration_s <= 0:
        return

    deadline = time.monotonic() + duration_s
    logging.info(
        "Pre-inference viewer adjustment: %.1fs. Rotate/zoom/pan now; rollout starts automatically.",
        duration_s,
    )
    while time.monotonic() < deadline:
        if hasattr(simulation_app, "is_running") and not simulation_app.is_running():
            break
        simulation_app.update()
        time.sleep(1.0 / 60.0)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import torch

    hole_size_m = resolve_hole_size_m(args)
    if hole_size_m is not None and args.scene != 5:
        logging.warning(
            "--hole-size was provided with --scene %d; using scene 5, the scene-4 copy "
            "with a resizable hole.",
            args.scene,
        )
        args.scene = 5

    prompt = args.prompt or DEFAULT_PROMPTS[args.scene]
    cv2 = None
    gui_enabled = not args.headless
    client: SimDroidPolicyClient | None = None
    env = None
    simulation_app = None
    did_cleanup = False
    if not args.headless:
        import cv2

    def cleanup() -> None:
        nonlocal did_cleanup
        if did_cleanup:
            return
        did_cleanup = True

        if cv2 is not None and gui_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                logging.debug("Ignoring OpenCV shutdown error", exc_info=True)

        if client is not None:
            try:
                client.close()
            except Exception:
                logging.debug("Ignoring policy client shutdown error", exc_info=True)

        if env is not None:
            try:
                env.close()
            except Exception:
                logging.warning("Failed to close gym environment cleanly", exc_info=True)

        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                logging.warning("Failed to close Isaac Sim app cleanly", exc_info=True)
            # Best-effort fallback for rare cases where close() does not tear down the Kit app.
            try:
                kit_app_module = importlib.import_module("omni.kit.app")
                kit_app = kit_app_module.get_app()
                if kit_app is not None:
                    kit_app.post_quit()
            except Exception:
                logging.debug("Ignoring Kit post_quit fallback failure", exc_info=True)

    def _handle_termination(signum: int, _frame: Any) -> None:
        logging.warning("Received signal %s; shutting down Isaac Sim...", signum)
        cleanup()
        raise KeyboardInterrupt

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)
    atexit.register(cleanup)

    video_dir = args.video_dir
    if video_dir is None:
        now = datetime.now()
        video_dir = Path("runs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    trace_dir = args.trace_dir or (video_dir / "trace")

    client = SimDroidPolicyClient(
        host="localhost",
        port=args.port,
        prompt=prompt,
        open_loop_horizon=args.open_loop_horizon,
        trace_dir=trace_dir,
        lock_gripper_close=args.lock_gripper_close,
        latch_gripper_after_close=args.latch_gripper_after_close,
    )
    logging.info("Execution trace CSV: %s", trace_dir / "execution_trace.csv")
    if args.lock_gripper_close:
        logging.info("Gripper override enabled: executed action[-1] is forced to 1.0 (close)")
    elif args.latch_gripper_after_close:
        logging.info("Gripper latch enabled: model controls gripper until first close, then close is held")

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(description="DROID Isaac app launcher")
    AppLauncher.add_app_launcher_args(app_parser)
    args_cli, _ = app_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    args_cli.device = resolve_device(args_cli.device)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

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
    scene_path = None
    if args.scene == 5 and hole_size_m is not None:
        scene_path = create_scene5_with_hole_size(hole_size_m)
        logging.info(
            "Using generated scene 5 with hole size %.1f mm: %s",
            hole_size_m * 1000.0,
            scene_path,
        )
    elif args.scene == 5:
        logging.info(
            "Using scene 5 default hole size %.1f mm. Pass --hole-size to resize it.",
            SCENE5_DEFAULT_HOLE_SIZE_M * 1000.0,
        )
    env_cfg.set_scene(args.scene, scene_path=scene_path)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()

    if not args.headless:
        wait_for_view_adjustment(simulation_app, args.pre_infer_view_seconds)

    if not args.no_save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or env.env.max_episode_length
    logging.info(
        "Starting rollouts: episodes=%d scene=%d hole_size_mm=%s prompt=%r max_steps=%d",
        args.episodes,
        args.scene,
        None if hole_size_m is None else round(hole_size_m * 1000.0, 3),
        prompt,
        max_steps,
    )

    try:
        with torch.no_grad():
            for ep in range(args.episodes):
                video: list[np.ndarray] = []
                for env_step in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{args.episodes}"):
                    ret = client.infer(obs)
                    video.append(ret["viz"])

                    if gui_enabled and cv2 is not None:
                        try:
                            cv2.imshow(
                                "DROID cameras: external | external_2 | wrist",
                                cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR),
                            )
                            cv2.waitKey(1)
                        except Exception:
                            gui_enabled = False
                            logging.warning(
                                "OpenCV GUI display is unavailable; disabling live preview. "
                                "Use --headless to suppress this warning.",
                                exc_info=True,
                            )

                    action = torch.tensor(ret["action"], dtype=torch.float32)[None]
                    step_t0 = time.time()
                    obs, _, terminated, truncated, _ = env.step(action)
                    step_dt = time.time() - step_t0
                    client.record_execution(
                        episode=ep,
                        env_step=env_step,
                        trace=ret["trace"],
                        post_obs=obs,
                        step_wall_time_s=step_dt,
                    )
                    if terminated or truncated:
                        break

                client.reset()
                if not args.no_save_video and video:
                    import mediapy

                    output_path = video_dir / f"episode_{ep}.mp4"
                    mediapy.write_video(output_path, video, fps=15)
                    logging.info("Wrote %s", output_path)

                if ep + 1 < args.episodes:
                    obs, _ = env.reset()
                    obs, _ = env.reset()
    finally:
        cleanup()
        atexit.unregister(cleanup)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
