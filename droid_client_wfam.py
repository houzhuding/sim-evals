#!/usr/bin/env python3
"""Run NVIDIA DROID simulation against a WFAM websocket policy server.

WFAM uses the DROID three-camera observation layout, but its policy state/action
contract is different from the original joint-position client:

Observation sent to server:
  - observation/ee_pose9d_rot6d: (9,)
  - observation/gripper_position: (1,)
  - observation/wrist_wrench: (6,)

Action chunk received from server:
  - (T, 16) = [ee_pose9d_rot6d(9), gripper(1), predicted_wrench(6)]

The sim-evals DROID environment executes joint-position actions. The default
execution mode is therefore ``ik-joint``: run WFAM inference, solve a small
damped-least-squares Franka IK problem from the predicted EE pose, and execute
the resulting 7D joint target plus gripper.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import importlib
import json
import logging
import math
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import droid_client as droid_base


RELATIVE_FRAME_OFFSETS = droid_base.RELATIVE_FRAME_OFFSETS
DEFAULT_PROMPTS = droid_base.DEFAULT_PROMPTS
WFAM_ACTION_WIDTH = 16
WFAM_CONTROL_WIDTH = 10
WFAM_WRENCH_WIDTH = 6
PANDA_JOINT_LOW = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
PANDA_JOINT_HIGH = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector * 0.0
    return vector / norm


def pose9d_rot6d_to_matrix(pose9d: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = droid_base.vector_or_none(pose9d, 9)
    if pose is None:
        raise ValueError("Expected a 9D pose vector")
    position = pose[:3].astype(np.float64)
    col0 = normalize_vector(pose[3:6].astype(np.float64))
    col1_raw = pose[6:9].astype(np.float64)
    col1 = normalize_vector(col1_raw - np.dot(col0, col1_raw) * col0)
    if np.linalg.norm(col1) < 1e-8:
        col1 = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        col1 = normalize_vector(col1 - np.dot(col0, col1) * col0)
    col2 = normalize_vector(np.cross(col0, col1))
    rotation = np.stack([col0, col1, col2], axis=1)
    return position, rotation


def make_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def pose9d_rot6d_to_transform(pose9d: Any) -> np.ndarray:
    position, rotation = pose9d_rot6d_to_matrix(pose9d)
    return make_transform(position, rotation)


def matrix_to_rpy_zyx(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def vector_to_json(value: Any, digits: int = 6) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), digits) for item in array]


def matrix_to_json(value: Any, digits: int = 6) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float64)
    return [
        [round(float(item), digits) for item in row]
        for row in array.tolist()
    ]


def format_vector(value: Any, digits: int = 2) -> str:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{item:.{digits}f}" for item in array) + "]"


def format_matrix4(value: Any, digits: int = 4) -> str:
    array = np.asarray(value, dtype=np.float64).reshape(4, 4)
    return "\n".join(
        "  [" + ", ".join(f"{item:.{digits}f}" for item in row) + "]"
        for row in array
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.asarray(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class FrankaPose9DIKSolver:
    """Tiny dependency-free DLS IK solver for the Franka/Panda arm.

    The solver uses a Franka-style DH chain to estimate the local Jacobian, but
    computes pose error against the measured sim observation. This calibration
    step makes the solver tolerant to small tool-frame differences between the
    analytical chain and the Isaac asset.
    """

    def __init__(
        self,
        *,
        max_iters: int = 12,
        damping: float = 0.05,
        step_scale: float = 0.8,
        max_joint_step: float = 0.08,
        max_target_delta_m: float = 0.06,
        orientation_weight: float = 0.35,
        position_tolerance: float = 0.004,
        orientation_tolerance: float = 0.05,
        tool_z_offset: float = 0.107,
    ) -> None:
        self.max_iters = int(max_iters)
        self.damping = float(damping)
        self.step_scale = float(step_scale)
        self.max_joint_step = float(max_joint_step)
        self.max_target_delta_m = float(max_target_delta_m)
        self.orientation_weight = float(orientation_weight)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.tool_z_offset = float(tool_z_offset)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        # Standard Franka/Panda DH approximation to the hand/flange frame.
        params = (
            (0.0, 0.0, 0.333, q[0]),
            (0.0, -math.pi / 2.0, 0.0, q[1]),
            (0.0, math.pi / 2.0, 0.316, q[2]),
            (0.0825, math.pi / 2.0, 0.0, q[3]),
            (-0.0825, -math.pi / 2.0, 0.384, q[4]),
            (0.0, math.pi / 2.0, 0.0, q[5]),
            (0.088, math.pi / 2.0, 0.0, q[6]),
            (0.0, 0.0, self.tool_z_offset, 0.0),
        )
        transform = np.eye(4, dtype=np.float64)
        for a, alpha, d, theta in params:
            transform = transform @ dh_transform(a, alpha, d, theta)
        return transform

    def _estimate_world_pose(self, q: np.ndarray, current_q: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        fk_current = self.forward_kinematics(current_q)
        fk_q = self.forward_kinematics(q)
        return current_pose @ np.linalg.inv(fk_current) @ fk_q

    def _pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target[:3, 3] - current[:3, 3]
        rot_error = matrix_to_rotvec(target[:3, :3] @ current[:3, :3].T)
        return np.concatenate([pos_error, self.orientation_weight * rot_error], axis=0)

    def _numeric_jacobian(
        self,
        q: np.ndarray,
        current_q: np.ndarray,
        measured_current_pose: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        base_pose = self._estimate_world_pose(q, current_q, measured_current_pose)
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for joint_index in range(7):
            q_eps = q.copy()
            q_eps[joint_index] += eps
            pose_eps = self._estimate_world_pose(q_eps, current_q, measured_current_pose)
            jacobian[:3, joint_index] = (pose_eps[:3, 3] - base_pose[:3, 3]) / eps
            delta_rot = matrix_to_rotvec(pose_eps[:3, :3] @ base_pose[:3, :3].T)
            jacobian[3:, joint_index] = self.orientation_weight * delta_rot / eps
        return jacobian

    def solve(
        self,
        current_q: Any,
        current_pose9d: Any,
        target_pose9d: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        current_q_arr = droid_base.vector_or_none(current_q, 7)
        if current_q_arr is None:
            current_q_arr = np.zeros(7, dtype=np.float32)
        current_q_arr = np.asarray(current_q_arr, dtype=np.float64)
        current_pos, current_rot = pose9d_rot6d_to_matrix(current_pose9d)
        target_pos, target_rot = pose9d_rot6d_to_matrix(target_pose9d)

        target_delta = target_pos - current_pos
        target_delta_norm = float(np.linalg.norm(target_delta))
        if self.max_target_delta_m > 0 and target_delta_norm > self.max_target_delta_m:
            target_pos = current_pos + target_delta * (self.max_target_delta_m / target_delta_norm)

        measured_current_pose = make_transform(current_pos, current_rot)
        target_pose = make_transform(target_pos, target_rot)
        q = np.clip(current_q_arr.copy(), PANDA_JOINT_LOW, PANDA_JOINT_HIGH)

        info: dict[str, Any] = {
            "success": False,
            "iters": 0,
            "pos_error_m": math.nan,
            "rot_error_rad": math.nan,
            "clipped_target_delta_m": target_delta_norm,
        }
        for iteration in range(max(1, self.max_iters)):
            estimated_pose = self._estimate_world_pose(q, current_q_arr, measured_current_pose)
            raw_rot_error = matrix_to_rotvec(target_pose[:3, :3] @ estimated_pose[:3, :3].T)
            error = self._pose_error(estimated_pose, target_pose)
            pos_error = float(np.linalg.norm(error[:3]))
            rot_error = float(np.linalg.norm(raw_rot_error))
            info.update(
                {
                    "iters": iteration + 1,
                    "pos_error_m": pos_error,
                    "rot_error_rad": rot_error,
                    "success": pos_error <= self.position_tolerance and rot_error <= self.orientation_tolerance,
                }
            )
            if info["success"]:
                break

            jacobian = self._numeric_jacobian(q, current_q_arr, measured_current_pose)
            lhs = jacobian @ jacobian.T + (self.damping ** 2) * np.eye(6, dtype=np.float64)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq * self.step_scale, -self.max_joint_step, self.max_joint_step)
            q = np.clip(q + dq, PANDA_JOINT_LOW, PANDA_JOINT_HIGH)

        return q.astype(np.float32), info


class SimDroidWFAMPolicyClient:
    """Stateful WFAM client that buffers action chunks from the websocket server."""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        open_loop_horizon: int = 24,
        trace_dir: Path | None = None,
        lock_gripper_close: bool = False,
        latch_gripper_after_close: bool = False,
        execution_mode: str = "ik-joint",
        allow_missing_ee_pose: bool = False,
        ik_solver: FrankaPose9DIKSolver | None = None,
        dump_action_chunks: bool = False,
        print_action_chunk_matrices: bool = False,
        action_chunk_dump_dir: Path | None = None,
    ) -> None:
        if execution_mode not in {"ik-joint", "hold-joint", "pose9d-direct"}:
            raise ValueError(f"Unsupported execution_mode: {execution_mode}")
        self.client = droid_base.RoboarenaWebsocketClient(host=host, port=port)
        self.prompt = prompt
        self.open_loop_horizon = int(open_loop_horizon)
        self.lock_gripper_close = bool(lock_gripper_close)
        self.latch_gripper_after_close = bool(latch_gripper_after_close)
        self.execution_mode = execution_mode
        self.allow_missing_ee_pose = bool(allow_missing_ee_pose)
        self.ik_solver = ik_solver or FrankaPose9DIKSolver()
        self.dump_action_chunks = bool(dump_action_chunks)
        self.print_action_chunk_matrices = bool(print_action_chunk_matrices)
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
            self._trace_file = open(self.trace_dir / "execution_trace_wfam.csv", "w", newline="")
            self._trace_writer = csv.DictWriter(
                self._trace_file,
                fieldnames=execution_trace_fieldnames(),
            )
            self._trace_writer.writeheader()
        if action_chunk_dump_dir is not None:
            self.action_chunk_dump_dir = action_chunk_dump_dir
        elif self.trace_dir is not None:
            self.action_chunk_dump_dir = self.trace_dir / "action_chunks"
        else:
            self.action_chunk_dump_dir = None

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
        action_space = server_config.get("action_space")
        if action_space not in {"ee_pose9d_wrench", "wfam_pose9d_wrench"}:
            raise ValueError(f"Expected WFAM action space, got {action_space}")

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
        pre_server_obs = extract_wfam_server_fields(
            sim_obs,
            allow_missing_ee_pose=self.allow_missing_ee_pose,
        )
        self.obs_history.append(pre_server_obs)
        history_limit = max(abs(min(RELATIVE_FRAME_OFFSETS)) + 1, self.open_loop_horizon + 1)
        if len(self.obs_history) > history_limit:
            self.obs_history = self.obs_history[-history_limit:]

        if (
            self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= self.open_loop_horizon
            or self.actions_from_chunk_completed >= len(self.pred_action_chunk)
        ):
            request = make_wfam_server_observation(
                self._select_request_frames(),
                prompt=self.prompt,
                session_id=self.session_id,
            )
            t0 = time.time()
            response = self.client.infer(request)
            dt = time.time() - t0
            self.pred_action_chunk = extract_wfam_action_chunk(response)
            self.actions_from_chunk_completed = 0
            self.chunk_index += 1
            self.last_chunk_infer_time_s = dt
            logging.info(
                "Received WFAM action chunk %s in %.2fs, range [%.4f, %.4f]",
                self.pred_action_chunk.shape,
                dt,
                float(self.pred_action_chunk.min()),
                float(self.pred_action_chunk.max()),
            )
            self._dump_action_chunk_debug(pre_server_obs)

        horizon_index = self.actions_from_chunk_completed
        raw_action = self.pred_action_chunk[horizon_index].astype(np.float32)
        control_pose9d = raw_action[:WFAM_CONTROL_WIDTH].copy()
        predicted_wrench = raw_action[WFAM_CONTROL_WIDTH:WFAM_ACTION_WIDTH].copy()
        self.actions_from_chunk_completed += 1

        model_gripper_close = bool(control_pose9d[9] > 0.5)
        if self.lock_gripper_close:
            control_pose9d[9] = 1.0
            self.gripper_latched_closed = True
        elif self.latch_gripper_after_close:
            if model_gripper_close:
                self.gripper_latched_closed = True
            control_pose9d[9] = 1.0 if self.gripper_latched_closed else 0.0
        else:
            control_pose9d[9] = 1.0 if model_gripper_close else 0.0

        ik_info = {
            "success": False,
            "iters": 0,
            "pos_error_m": math.nan,
            "rot_error_rad": math.nan,
            "clipped_target_delta_m": math.nan,
        }
        joint_q = droid_base.vector_or_none(
            pre_server_obs.get("observation/joint_position"),
            7,
        )
        if joint_q is None or np.any(~np.isfinite(joint_q)):
            joint_q = np.zeros(7, dtype=np.float32)

        if self.execution_mode == "pose9d-direct":
            executed_action = control_pose9d
        elif self.execution_mode == "ik-joint":
            ik_q, ik_info = self.ik_solver.solve(
                joint_q,
                pre_server_obs["observation/ee_pose9d_rot6d"],
                control_pose9d[:9],
            )
            executed_action = np.concatenate(
                [ik_q.astype(np.float32), control_pose9d[9:10]],
                axis=0,
            ).astype(np.float32)
        else:
            executed_action = np.concatenate(
                [joint_q.astype(np.float32), control_pose9d[9:10]],
                axis=0,
            ).astype(np.float32)

        return {
            "action": executed_action,
            "viz": droid_base.make_viz_image(sim_obs),
            "trace": {
                "session_id": self.session_id,
                "chunk_index": self.chunk_index,
                "horizon_index": horizon_index,
                "chunk_infer_time_s": self.last_chunk_infer_time_s,
                "execution_mode": self.execution_mode,
                "lock_gripper_close": self.lock_gripper_close,
                "latch_gripper_after_close": self.latch_gripper_after_close,
                "gripper_latched_closed": self.gripper_latched_closed,
                "pre_obs": pre_server_obs,
                "raw_wfam_action": raw_action,
                "control_pose9d": control_pose9d,
                "predicted_wrench": predicted_wrench,
                "executed_action": executed_action,
                "ik_info": ik_info,
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
            "execution_mode": trace["execution_mode"],
            "lock_gripper_close": bool(trace.get("lock_gripper_close", False)),
            "latch_gripper_after_close": bool(trace.get("latch_gripper_after_close", False)),
            "gripper_latched_closed": bool(trace.get("gripper_latched_closed", False)),
        }
        ik_info = trace.get("ik_info", {}) or {}
        row["ik_success"] = bool(ik_info.get("success", False))
        row["ik_iters"] = int(ik_info.get("iters", 0))
        row["ik_pos_error_m"] = float(ik_info.get("pos_error_m", math.nan))
        row["ik_rot_error_rad"] = float(ik_info.get("rot_error_rad", math.nan))
        row["ik_clipped_target_delta_m"] = float(ik_info.get("clipped_target_delta_m", math.nan))
        droid_base.add_vector_fields(row, "pre_joint_q", trace["pre_obs"].get("observation/joint_position"), 7)
        droid_base.add_vector_fields(row, "pre_ee_pose9d", trace["pre_obs"].get("observation/ee_pose9d_rot6d"), 9)
        droid_base.add_vector_fields(row, "pre_gripper", trace["pre_obs"].get("observation/gripper_position"), 1)
        droid_base.add_vector_fields(row, "pre_wrench", trace["pre_obs"].get("observation/wrist_wrench"), 6)
        droid_base.add_vector_fields(row, "raw_wfam_action", trace["raw_wfam_action"], 16)
        droid_base.add_vector_fields(row, "control_pose9d", trace["control_pose9d"], 10)
        droid_base.add_vector_fields(row, "predicted_wrench", trace["predicted_wrench"], 6)
        droid_base.add_vector_fields(row, "executed_action", trace["executed_action"], len(trace["executed_action"]))
        post_server_obs = extract_wfam_server_fields(
            post_obs,
            allow_missing_ee_pose=self.allow_missing_ee_pose,
        )
        droid_base.add_vector_fields(row, "post_joint_q", post_server_obs.get("observation/joint_position"), 7)
        droid_base.add_vector_fields(row, "post_ee_pose9d", post_server_obs.get("observation/ee_pose9d_rot6d"), 9)
        droid_base.add_vector_fields(row, "post_gripper", post_server_obs.get("observation/gripper_position"), 1)
        droid_base.add_vector_fields(row, "post_wrench", post_server_obs.get("observation/wrist_wrench"), 6)
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

    def _dump_action_chunk_debug(self, pre_server_obs: dict[str, Any]) -> None:
        if self.pred_action_chunk is None:
            return
        if not (self.dump_action_chunks or self.print_action_chunk_matrices):
            return

        current_pose9d = pre_server_obs["observation/ee_pose9d_rot6d"]
        current_transform = pose9d_rot6d_to_transform(current_pose9d)
        current_inverse = np.linalg.inv(current_transform)
        current_joint_q = droid_base.vector_or_none(
            pre_server_obs.get("observation/joint_position"),
            7,
        )
        if current_joint_q is None:
            current_joint_q = np.zeros(7, dtype=np.float32)
        current_joint_q = np.asarray(current_joint_q, dtype=np.float64)
        current_joint_deg = np.rad2deg(current_joint_q)
        current_xyz_mm = current_transform[:3, 3] * 1000.0
        current_rpy_deg = np.rad2deg(matrix_to_rpy_zyx(current_transform[:3, :3]))

        logging.info(
            "WFAM chunk %04d debug: current_xyz_mm=%s current_rpy_deg=%s current_joint_deg=%s",
            self.chunk_index,
            format_vector(current_xyz_mm),
            format_vector(current_rpy_deg),
            format_vector(current_joint_deg),
        )

        rows = []
        for horizon_index, raw_action in enumerate(self.pred_action_chunk):
            target_pose9d = raw_action[:9]
            target_transform = pose9d_rot6d_to_transform(target_pose9d)
            delta_transform = current_inverse @ target_transform
            target_xyz_mm = target_transform[:3, 3] * 1000.0
            target_rpy_deg = np.rad2deg(matrix_to_rpy_zyx(target_transform[:3, :3]))
            delta_xyz_world_mm = (target_transform[:3, 3] - current_transform[:3, 3]) * 1000.0
            delta_xyz_local_mm = delta_transform[:3, 3] * 1000.0
            delta_rpy_deg = np.rad2deg(matrix_to_rpy_zyx(delta_transform[:3, :3]))
            target_delta_m = float(np.linalg.norm(target_transform[:3, 3] - current_transform[:3, 3]))

            ik_q, ik_info = self.ik_solver.solve(
                current_joint_q,
                current_pose9d,
                target_pose9d,
            )
            ik_joint_deg = np.rad2deg(np.asarray(ik_q, dtype=np.float64))
            ik_delta_joint_deg = ik_joint_deg - current_joint_deg
            predicted_wrench = raw_action[WFAM_CONTROL_WIDTH:WFAM_ACTION_WIDTH]

            logging.info(
                (
                    "WFAM chunk %04d h%02d | abs_xyz_mm=%s abs_rpy_deg=%s | "
                    "delta_world_mm=%s delta_local_mm=%s delta_rpy_deg=%s | "
                    "grip_raw=%.3f wrench=%s | ik_q_deg=%s ik_dq_deg=%s | "
                    "ik_pos_err=%.2fmm ik_rot_err=%.2fdeg success=%s"
                ),
                self.chunk_index,
                horizon_index,
                format_vector(target_xyz_mm),
                format_vector(target_rpy_deg),
                format_vector(delta_xyz_world_mm),
                format_vector(delta_xyz_local_mm),
                format_vector(delta_rpy_deg),
                float(raw_action[9]),
                format_vector(predicted_wrench, digits=3),
                format_vector(ik_joint_deg),
                format_vector(ik_delta_joint_deg),
                float(ik_info.get("pos_error_m", math.nan)) * 1000.0,
                math.degrees(float(ik_info.get("rot_error_rad", math.nan))),
                bool(ik_info.get("success", False)),
            )

            if self.print_action_chunk_matrices:
                logging.info(
                    (
                        "WFAM chunk %04d h%02d target_T_world:\n%s\n"
                        "WFAM chunk %04d h%02d delta_T_current_to_target:\n%s"
                    ),
                    self.chunk_index,
                    horizon_index,
                    format_matrix4(target_transform),
                    self.chunk_index,
                    horizon_index,
                    format_matrix4(delta_transform),
                )

            rows.append(
                {
                    "session_id": self.session_id,
                    "chunk_index": self.chunk_index,
                    "horizon_index": horizon_index,
                    "current_xyz_mm": vector_to_json(current_xyz_mm),
                    "current_rpy_deg": vector_to_json(current_rpy_deg),
                    "current_joint_deg": vector_to_json(current_joint_deg),
                    "raw_action": vector_to_json(raw_action),
                    "target_T_world": matrix_to_json(target_transform),
                    "target_xyz_mm": vector_to_json(target_xyz_mm),
                    "target_rpy_deg": vector_to_json(target_rpy_deg),
                    "delta_T_current_to_target": matrix_to_json(delta_transform),
                    "delta_xyz_world_mm": vector_to_json(delta_xyz_world_mm),
                    "delta_xyz_local_mm": vector_to_json(delta_xyz_local_mm),
                    "delta_rpy_deg": vector_to_json(delta_rpy_deg),
                    "target_delta_m": round(target_delta_m, 6),
                    "gripper_raw": round(float(raw_action[9]), 6),
                    "predicted_wrench": vector_to_json(predicted_wrench),
                    "ik_joint_deg": vector_to_json(ik_joint_deg),
                    "ik_delta_joint_deg": vector_to_json(ik_delta_joint_deg),
                    "ik_info": {
                        "success": bool(ik_info.get("success", False)),
                        "iters": int(ik_info.get("iters", 0)),
                        "pos_error_mm": round(float(ik_info.get("pos_error_m", math.nan)) * 1000.0, 6),
                        "rot_error_deg": round(math.degrees(float(ik_info.get("rot_error_rad", math.nan))), 6),
                        "clipped_target_delta_mm": round(
                            float(ik_info.get("clipped_target_delta_m", math.nan)) * 1000.0,
                            6,
                        ),
                    },
                }
            )

        if self.dump_action_chunks:
            dump_dir = self.action_chunk_dump_dir
            if dump_dir is None:
                dump_dir = Path("runs") / "wfam_action_chunks"
                self.action_chunk_dump_dir = dump_dir
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"chunk_{self.chunk_index:04d}.jsonl"
            with open(dump_path, "w") as file:
                for row in rows:
                    file.write(json.dumps(row, sort_keys=True) + "\n")
            logging.info("Wrote WFAM decoded action chunk debug: %s", dump_path)


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
        "execution_mode",
        "ik_success",
        "ik_iters",
        "ik_pos_error_m",
        "ik_rot_error_rad",
        "ik_clipped_target_delta_m",
        "lock_gripper_close",
        "latch_gripper_after_close",
        "gripper_latched_closed",
    ]
    for prefix, width in (
        ("pre_joint_q", 7),
        ("pre_ee_pose9d", 9),
        ("pre_gripper", 1),
        ("pre_wrench", 6),
        ("raw_wfam_action", 16),
        ("control_pose9d", 10),
        ("predicted_wrench", 6),
        ("executed_action", 10),
        ("post_joint_q", 7),
        ("post_ee_pose9d", 9),
        ("post_gripper", 1),
        ("post_wrench", 6),
    ):
        fields.extend(f"{prefix}_{index}" for index in range(width))
    return fields


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rotvec, axis=1)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None], rotvec.shape[0], axis=0)
    nonzero = theta > 1e-8
    if not np.any(nonzero):
        return matrices
    axis = np.zeros_like(rotvec)
    axis[nonzero] = rotvec[nonzero] / theta[nonzero, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    k_mat = np.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros],
        axis=1,
    ).reshape(-1, 3, 3)
    sin_t = np.sin(theta)[:, None, None]
    cos_t = np.cos(theta)[:, None, None]
    matrices = matrices + sin_t * k_mat + (1.0 - cos_t) * np.matmul(k_mat, k_mat)
    matrices[~nonzero] = np.eye(3, dtype=np.float64)
    return matrices


def pose6_rotvec_to_pose9d_rot6d(pose6: Any) -> np.ndarray:
    pose6_array = droid_base.vector_or_none(pose6, 6)
    if pose6_array is None:
        return np.zeros(9, dtype=np.float32)
    matrix = rotvec_to_matrix(pose6_array[3:6][None])[0]
    rot6d = np.concatenate([matrix[:, 0], matrix[:, 1]], axis=0)
    return np.concatenate([pose6_array[:3], rot6d], axis=0).astype(np.float32)


def quat_xyzw_to_pose9d_rot6d(position: Any, quat_xyzw: Any) -> np.ndarray:
    pos = droid_base.vector_or_none(position, 3)
    quat = droid_base.vector_or_none(quat_xyzw, 4)
    if pos is None or quat is None:
        return np.zeros(9, dtype=np.float32)
    quat = quat.astype(np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    x, y, z, w = quat
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    rot6d = np.concatenate([matrix[:, 0], matrix[:, 1]], axis=0)
    return np.concatenate([pos, rot6d], axis=0).astype(np.float32)


def extract_ee_pose9d(policy_obs: dict[str, Any], allow_missing: bool) -> np.ndarray:
    for key in ("ee_pose9d_rot6d", "ee_pose9d", "cartesian_pose9d_rot6d"):
        if key in policy_obs:
            pose = droid_base.vector_or_none(policy_obs[key], 9)
            if pose is not None:
                return pose.astype(np.float32)

    for key in ("cartesian_position", "ee_pose6d", "ee_pose_rotvec"):
        if key in policy_obs:
            return pose6_rotvec_to_pose9d_rot6d(policy_obs[key])

    if "ee_pos" in policy_obs and "ee_quat_xyzw" in policy_obs:
        return quat_xyzw_to_pose9d_rot6d(policy_obs["ee_pos"], policy_obs["ee_quat_xyzw"])

    if allow_missing:
        logging.warning("Missing ee_pose9d_rot6d; using zeros because --allow-missing-ee-pose is set.")
        return np.zeros(9, dtype=np.float32)
    raise KeyError(
        "WFAM inference needs policy obs key 'ee_pose9d_rot6d'. "
        "Update the sim environment or pass --allow-missing-ee-pose for smoke tests."
    )


def extract_wfam_server_fields(sim_obs: dict[str, Any], *, allow_missing_ee_pose: bool = False) -> dict[str, Any]:
    policy_obs = sim_obs["policy"]
    fields = {
        "observation/exterior_image_0_left": droid_base.tensor_image_to_numpy(
            policy_obs["external_cam"]
        ),
        "observation/exterior_image_1_left": droid_base.tensor_image_to_numpy(
            policy_obs["external_cam_2"]
        ),
        "observation/wrist_image_left": droid_base.tensor_image_to_numpy(policy_obs["wrist_cam"]),
        "observation/joint_position": droid_base.tensor_state_to_numpy(
            policy_obs.get("arm_joint_pos", np.zeros(7, dtype=np.float32))
        ),
        "observation/ee_pose9d_rot6d": extract_ee_pose9d(
            policy_obs,
            allow_missing=allow_missing_ee_pose,
        ),
        "observation/gripper_position": droid_base.tensor_state_to_numpy(
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
            wrench = droid_base.vector_or_none(policy_obs[wrench_key], 6)
            if wrench is not None:
                fields["observation/wrist_wrench"] = wrench
            break
    if "observation/wrist_wrench" not in fields:
        fields["observation/wrist_wrench"] = np.zeros(6, dtype=np.float32)
    return fields


def make_wfam_server_observation(
    frame_obs: list[dict[str, Any]],
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    current_obs = frame_obs[-1]
    request = {
        "endpoint": "infer",
        "observation/joint_position": current_obs["observation/joint_position"],
        "observation/ee_pose9d_rot6d": current_obs["observation/ee_pose9d_rot6d"],
        "observation/gripper_position": current_obs["observation/gripper_position"],
        "observation/wrist_wrench": current_obs["observation/wrist_wrench"],
        "prompt": prompt,
        "session_id": session_id,
    }

    for image_key in (
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        images = [obs[image_key] for obs in frame_obs]
        request[image_key] = images[0] if len(images) == 1 else np.stack(images, axis=0)

    return request


def as_action_2d(value: Any, width: int) -> np.ndarray:
    if value is None:
        return np.zeros((1, width), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[-1] < width:
        padded = np.zeros((array.shape[0], width), dtype=np.float32)
        padded[:, : array.shape[-1]] = array
        return padded
    return array[:, :width].astype(np.float32, copy=False)


def extract_wfam_action_chunk(response: Any) -> np.ndarray:
    """Normalize server responses to ``(T, 16) = pose9d + gripper + wrench``."""
    if isinstance(response, dict):
        if "actions" in response:
            response = response["actions"]
        elif "action" in response:
            response = response["action"]
        elif "action.ee_pose9d_rot6d" in response:
            pose = as_action_2d(response["action.ee_pose9d_rot6d"], 9)
            horizon = pose.shape[0]
            gripper = as_action_2d(response.get("action.gripper_position"), 1)
            wrench = as_action_2d(response.get("action.wrist_wrench"), 6)
            if gripper.shape[0] != horizon:
                gripper = np.repeat(gripper[:1], horizon, axis=0)
            if wrench.shape[0] != horizon:
                wrench = np.repeat(wrench[:1], horizon, axis=0)
            return np.concatenate([pose, gripper, wrench], axis=-1).astype(np.float32)

    actions = np.asarray(response, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2:
        raise ValueError(f"Expected WFAM action chunk shaped (T, D), got {actions.shape}")
    if actions.shape[-1] == WFAM_CONTROL_WIDTH:
        zeros = np.zeros((actions.shape[0], WFAM_WRENCH_WIDTH), dtype=np.float32)
        actions = np.concatenate([actions, zeros], axis=-1)
    if actions.shape[-1] != WFAM_ACTION_WIDTH:
        raise ValueError(f"Expected WFAM action chunk shaped (T, 16), got {actions.shape}")
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NVIDIA DROID sim with a remote WFAM websocket policy."
    )
    parser.add_argument("--port", type=int, default=50532, help="Policy server port")
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollouts")
    parser.add_argument("--scene", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--hole-size", type=float, default=None)
    parser.add_argument("--hole-size-mm", type=float, default=None)
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--open-loop-horizon", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cam-width", type=int, default=320)
    parser.add_argument("--cam-height", type=int, default=160)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument(
        "--dump-action-chunks",
        action="store_true",
        help=(
            "Decode every received WFAM chunk into 24 human-readable actions. "
            "Writes JSONL under trace/action_chunks and logs a compact mm/degree table."
        ),
    )
    parser.add_argument(
        "--action-chunk-dump-dir",
        type=Path,
        default=None,
        help="Override the directory for --dump-action-chunks JSONL files.",
    )
    parser.add_argument(
        "--print-action-chunk-matrices",
        action="store_true",
        help="Also print full 4x4 target and delta transformation matrices for every action.",
    )
    parser.add_argument("--lock-gripper-close", action="store_true")
    parser.add_argument("--latch-gripper-after-close", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--pre-infer-view-seconds", type=float, default=0.0)
    parser.add_argument(
        "--execution-mode",
        choices=["ik-joint", "hold-joint", "pose9d-direct"],
        default="ik-joint",
        help=(
            "ik-joint solves WFAM ee_pose9d to a 7D joint target for the current DROID env. "
            "hold-joint only logs WFAM inference. pose9d-direct requires a compatible EE-pose env."
        ),
    )
    parser.add_argument("--ik-max-iters", type=int, default=12)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--ik-step-scale", type=float, default=0.8)
    parser.add_argument("--ik-max-joint-step", type=float, default=0.08)
    parser.add_argument(
        "--ik-max-target-delta-m",
        type=float,
        default=0.06,
        help="Clamp each WFAM target pose to this translational delta from the current EE pose.",
    )
    parser.add_argument("--ik-orientation-weight", type=float, default=0.35)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.004)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--ik-tool-z-offset",
        type=float,
        default=0.107,
        help="Approximate Franka flange-to-tool z offset used by the internal Jacobian model.",
    )
    parser.add_argument(
        "--allow-missing-ee-pose",
        action="store_true",
        help="Use zero ee pose if the environment does not expose ee_pose9d_rot6d.",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import torch

    hole_size_m = droid_base.resolve_hole_size_m(args)
    if hole_size_m is not None and args.scene != 5:
        logging.warning("--hole-size was provided with --scene %d; using scene 5.", args.scene)
        args.scene = 5

    prompt = args.prompt or DEFAULT_PROMPTS[args.scene]
    cv2 = None
    gui_enabled = not args.headless
    client: SimDroidWFAMPolicyClient | None = None
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
    ik_solver = FrankaPose9DIKSolver(
        max_iters=args.ik_max_iters,
        damping=args.ik_damping,
        step_scale=args.ik_step_scale,
        max_joint_step=args.ik_max_joint_step,
        max_target_delta_m=args.ik_max_target_delta_m,
        orientation_weight=args.ik_orientation_weight,
        position_tolerance=args.ik_position_tolerance,
        orientation_tolerance=args.ik_orientation_tolerance,
        tool_z_offset=args.ik_tool_z_offset,
    )

    client = SimDroidWFAMPolicyClient(
        host="localhost",
        port=args.port,
        prompt=prompt,
        open_loop_horizon=args.open_loop_horizon,
        trace_dir=trace_dir,
        lock_gripper_close=args.lock_gripper_close,
        latch_gripper_after_close=args.latch_gripper_after_close,
        execution_mode=args.execution_mode,
        allow_missing_ee_pose=args.allow_missing_ee_pose,
        ik_solver=ik_solver,
        dump_action_chunks=args.dump_action_chunks,
        print_action_chunk_matrices=args.print_action_chunk_matrices,
        action_chunk_dump_dir=args.action_chunk_dump_dir,
    )
    logging.info("WFAM execution trace CSV: %s", trace_dir / "execution_trace_wfam.csv")
    if args.dump_action_chunks:
        logging.info(
            "WFAM decoded action chunk dumps: %s",
            args.action_chunk_dump_dir or (trace_dir / "action_chunks"),
        )
    if args.execution_mode == "hold-joint":
        logging.warning(
            "WFAM execution-mode=hold-joint: model outputs are logged, but the old "
            "DROID env is commanded to hold current joints. Use pose9d-direct only "
            "with a compatible EE-pose controller."
        )
    elif args.execution_mode == "ik-joint":
        logging.info(
            "WFAM execution-mode=ik-joint: DLS IK enabled "
            "(iters=%d damping=%.4f max_joint_step=%.3f max_target_delta_m=%.3f).",
            args.ik_max_iters,
            args.ik_damping,
            args.ik_max_joint_step,
            args.ik_max_target_delta_m,
        )

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(description="DROID Isaac app launcher")
    AppLauncher.add_app_launcher_args(app_parser)
    args_cli, _ = app_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    args_cli.device = droid_base.resolve_device(args_cli.device)
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
    droid_base.configure_camera_resolution(env_cfg, args.cam_width, args.cam_height)
    scene_path = None
    if args.scene == 5 and hole_size_m is not None:
        scene_path = droid_base.create_scene5_with_hole_size(hole_size_m)
        logging.info("Using generated scene 5 with hole size %.1f mm: %s", hole_size_m * 1000.0, scene_path)
    elif args.scene == 5:
        logging.info(
            "Using scene 5 default hole size %.1f mm. Pass --hole-size to resize it.",
            droid_base.SCENE5_DEFAULT_HOLE_SIZE_M * 1000.0,
        )
    env_cfg.set_scene(args.scene, scene_path=scene_path)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()

    if not args.headless:
        droid_base.wait_for_view_adjustment(simulation_app, args.pre_infer_view_seconds)

    if not args.no_save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or env.env.max_episode_length
    logging.info(
        "Starting WFAM rollouts: episodes=%d scene=%d prompt=%r max_steps=%d execution_mode=%s",
        args.episodes,
        args.scene,
        prompt,
        max_steps,
        args.execution_mode,
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
                                "WFAM DROID cameras: external | external_2 | wrist",
                                cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR),
                            )
                            cv2.waitKey(1)
                        except Exception:
                            gui_enabled = False
                            logging.warning("OpenCV GUI unavailable; disabling preview.", exc_info=True)

                    action = torch.tensor(ret["action"], dtype=torch.float32)[None]
                    expected_shape = getattr(env.action_space, "shape", None)
                    if expected_shape and action.shape[-1] != expected_shape[-1]:
                        raise RuntimeError(
                            f"WFAM client produced action dim {action.shape[-1]}, but env expects "
                            f"{expected_shape[-1]}. Use --execution-mode hold-joint with the current "
                            "joint-position env, or switch to an EE-pose env/controller."
                        )
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
