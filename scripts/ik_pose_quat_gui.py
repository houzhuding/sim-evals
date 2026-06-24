#!/usr/bin/env python3
"""Small standalone Franka IK pose test with a matplotlib 3D GUI.

Input target pose:

    x y z qx qy qz qw

The solver is intentionally simple and dependency-light. It uses the same
Franka-style DH approximation and damped least-squares numerical Jacobian idea
as ``droid_client_wfam.py``. This is useful for sanity-checking pose targets and
joint outputs before sending them to the DROID environment.

Examples:

    python third_party/sim-evals/scripts/ik_pose_quat_gui.py

    python third_party/sim-evals/scripts/ik_pose_quat_gui.py \
      0.335511 -0.026887 0.504454 0 1 0 0

    python third_party/sim-evals/scripts/ik_pose_quat_gui.py \
      0.335511 -0.026887 0.504454 0 1 0 0 \
      --max-target-delta-m 0.06 --save runs/ik_debug.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PANDA_JOINT_LOW = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
PANDA_JOINT_HIGH = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)
PANDA_HOME_Q = np.asarray(
    # [0.0, -math.pi / 5.0, 0.0, -4.0 * math.pi / 5.0, 0.0, 3.0 * math.pi / 5.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector * 0.0
    return vector / norm


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    x, y, z, w = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
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


def make_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def pose_xyz_quat_to_transform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(7)
    return make_transform(values[:3], quat_xyzw_to_matrix(values[3:7]))


def rpy_zyx_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


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


def transform_to_xyzrpy(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate([transform[:3, 3], matrix_to_rpy_zyx(transform[:3, :3])], axis=0)


def xyzrpy_to_transform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(6)
    return make_transform(values[:3], rpy_zyx_to_matrix(values[3:6]))


def sanitize_joint_position(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64).reshape(7), PANDA_JOINT_LOW, PANDA_JOINT_HIGH)


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


def franka_dh_params(q: np.ndarray, tool_z_offset: float) -> tuple[tuple[float, float, float, float], ...]:
    q = np.asarray(q, dtype=np.float64).reshape(7)
    return (
        (0.0, 0.0, 0.333, q[0]),
        (0.0, -math.pi / 2.0, 0.0, q[1]),
        (0.0, math.pi / 2.0, 0.316, q[2]),
        (0.0825, math.pi / 2.0, 0.0, q[3]),
        (-0.0825, -math.pi / 2.0, 0.384, q[4]),
        (0.0, math.pi / 2.0, 0.0, q[5]),
        (0.088, math.pi / 2.0, 0.0, q[6]),
        (0.0, 0.0, tool_z_offset, 0.0),
    )


def link_transforms(q: np.ndarray, tool_z_offset: float) -> list[np.ndarray]:
    transform = np.eye(4, dtype=np.float64)
    transforms = [transform.copy()]
    for a, alpha, d, theta in franka_dh_params(q, tool_z_offset):
        transform = transform @ dh_transform(a, alpha, d, theta)
        transforms.append(transform.copy())
    return transforms


def forward_kinematics(q: np.ndarray, tool_z_offset: float) -> np.ndarray:
    return link_transforms(q, tool_z_offset)[-1]


@dataclass
class IkResult:
    q: np.ndarray
    success: bool
    iters: int
    pos_error_m: float
    rot_error_rad: float
    target_delta_m: float


class SimpleFrankaDlsIk:
    def __init__(
        self,
        *,
        max_iters: int = 120,
        damping: float = 0.05,
        step_scale: float = 0.8,
        max_joint_step: float = 0.08,
        max_target_delta_m: float = 0.0,
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

    def pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target[:3, 3] - current[:3, 3]
        rot_error = matrix_to_rotvec(target[:3, :3] @ current[:3, :3].T)
        return np.concatenate([pos_error, self.orientation_weight * rot_error], axis=0)

    def numeric_jacobian(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        base_pose = forward_kinematics(q, self.tool_z_offset)
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for joint_index in range(7):
            q_eps = q.copy()
            q_eps[joint_index] += eps
            pose_eps = forward_kinematics(q_eps, self.tool_z_offset)
            jacobian[:3, joint_index] = (pose_eps[:3, 3] - base_pose[:3, 3]) / eps
            delta_rot = matrix_to_rotvec(pose_eps[:3, :3] @ base_pose[:3, :3].T)
            jacobian[3:, joint_index] = self.orientation_weight * delta_rot / eps
        return jacobian

    def solve(self, current_q: np.ndarray, target_pose: np.ndarray) -> IkResult:
        current_q = np.asarray(current_q, dtype=np.float64).reshape(7)
        q = np.clip(current_q.copy(), PANDA_JOINT_LOW, PANDA_JOINT_HIGH)
        current_pose = forward_kinematics(q, self.tool_z_offset)

        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(4, 4).copy()
        target_delta = target_pose[:3, 3] - current_pose[:3, 3]
        target_delta_m = float(np.linalg.norm(target_delta))
        if self.max_target_delta_m > 0.0 and target_delta_m > self.max_target_delta_m:
            target_pose[:3, 3] = current_pose[:3, 3] + target_delta * (
                self.max_target_delta_m / target_delta_m
            )

        pos_error = math.inf
        rot_error = math.inf
        success = False
        iters = 0
        for iteration in range(max(1, self.max_iters)):
            pose = forward_kinematics(q, self.tool_z_offset)
            raw_rot_error = matrix_to_rotvec(target_pose[:3, :3] @ pose[:3, :3].T)
            error = self.pose_error(pose, target_pose)
            pos_error = float(np.linalg.norm(error[:3]))
            rot_error = float(np.linalg.norm(raw_rot_error))
            iters = iteration + 1
            success = pos_error <= self.position_tolerance and rot_error <= self.orientation_tolerance
            if success:
                break

            jacobian = self.numeric_jacobian(q)
            lhs = jacobian @ jacobian.T + (self.damping ** 2) * np.eye(6, dtype=np.float64)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq * self.step_scale, -self.max_joint_step, self.max_joint_step)
            q = np.clip(q + dq, PANDA_JOINT_LOW, PANDA_JOINT_HIGH)

        return IkResult(
            q=q,
            success=success,
            iters=iters,
            pos_error_m=pos_error,
            rot_error_rad=rot_error,
            target_delta_m=target_delta_m,
        )


def draw_frame(axis, transform: np.ndarray, *, label: str, length: float = 0.06) -> None:
    origin = transform[:3, 3]
    colors = ("tab:red", "tab:green", "tab:blue")
    for dim, color in enumerate(colors):
        vec = transform[:3, dim] * length
        axis.quiver(*origin, *vec, color=color, linewidth=2)
    axis.text(*origin, label)


def set_axes_equal(axis) -> None:
    limits = np.asarray([axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d()], dtype=np.float64)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    radius = max(radius, 0.1)
    axis.set_xlim3d(centers[0] - radius, centers[0] + radius)
    axis.set_ylim3d(centers[1] - radius, centers[1] + radius)
    axis.set_zlim3d(max(0.0, centers[2] - radius), centers[2] + radius)


def print_ik_result(result: IkResult, target_pose: np.ndarray, tool_z_offset: float) -> None:
    solution_pose = forward_kinematics(result.q, tool_z_offset)
    pos_error = np.linalg.norm(target_pose[:3, 3] - solution_pose[:3, 3])
    rot_error = np.linalg.norm(matrix_to_rotvec(target_pose[:3, :3] @ solution_pose[:3, :3].T))

    print("IK result")
    print(f"  success: {result.success}")
    print(f"  iterations: {result.iters}")
    print(f"  target_delta_from_current_m: {result.target_delta_m:.6f}")
    print(f"  final_position_error_m: {pos_error:.6f}")
    print(f"  final_rotation_error_deg: {math.degrees(rot_error):.3f}")
    print("  joint_position_rad:")
    print("   ", " ".join(f"{v:.8f}" for v in result.q))
    print("  joint_position_deg:")
    print("   ", " ".join(f"{v:.3f}" for v in np.rad2deg(result.q)))


def plot_static_result(
    current_q: np.ndarray,
    result_q: np.ndarray,
    target_pose: np.ndarray,
    tool_z_offset: float,
    save: Path | None,
    show: bool,
) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current_links = link_transforms(current_q, tool_z_offset)
    result_links = link_transforms(result_q, tool_z_offset)
    current_points = np.stack([t[:3, 3] for t in current_links], axis=0)
    result_points = np.stack([t[:3, 3] for t in result_links], axis=0)

    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(
        current_points[:, 0],
        current_points[:, 1],
        current_points[:, 2],
        "o--",
        color="0.55",
        label="current chain",
    )
    axis.plot(
        result_points[:, 0],
        result_points[:, 1],
        result_points[:, 2],
        "o-",
        color="tab:blue",
        label="IK solution chain",
    )
    axis.scatter(*target_pose[:3, 3], color="tab:red", s=80, marker="x", label="target")
    draw_frame(axis, current_links[-1], label="current")
    draw_frame(axis, result_links[-1], label="solution")
    draw_frame(axis, target_pose, label="target")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="upper right")
    set_axes_equal(axis)
    fig.tight_layout()
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=160)
        print(f"Saved visualization: {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


class FrankaIkWorkbench:
    """A compact matplotlib GUI for moving an EE target and solving IK."""

    CONTROL_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(
        self,
        *,
        current_q: np.ndarray,
        target_pose: np.ndarray,
        solver: SimpleFrankaDlsIk,
        xyz_ranges: dict[str, tuple[float, float]],
        rpy_range_deg: tuple[float, float],
        save: Path | None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        self.plt = plt
        self.Button = Button
        self.Slider = Slider
        self.TextBox = TextBox
        self.current_q = sanitize_joint_position(current_q)
        self.solution_q: np.ndarray | None = None
        self.solver = solver
        self.xyz_ranges = xyz_ranges
        self.rpy_range_deg = rpy_range_deg
        self.save = save
        self._updating_controls = False

        initial_xyzrpy = transform_to_xyzrpy(target_pose)
        initial_display = initial_xyzrpy.copy()
        initial_display[3:6] = np.rad2deg(initial_display[3:6])

        self.fig = plt.figure(figsize=(14.5, 8.2))
        self.axis = self.fig.add_axes([0.04, 0.10, 0.58, 0.84], projection="3d")
        self.status_axis = self.fig.add_axes([0.66, 0.06, 0.31, 0.10])
        self.status_axis.axis("off")
        self.status_text = self.status_axis.text(
            0.0,
            1.0,
            "",
            va="top",
            family="monospace",
            fontsize=9,
        )

        self.current_boxes = {}
        current_box_specs = [
            ("x", 0.66, 0.905),
            ("y", 0.765, 0.905),
            ("z", 0.870, 0.905),
            ("roll", 0.66, 0.850),
            ("pitch", 0.765, 0.850),
            ("yaw", 0.870, 0.850),
        ]
        for name, x0, y0 in current_box_specs:
            box = TextBox(self.fig.add_axes([x0, y0, 0.080, 0.035]), name, initial="")
            box.set_active(False)
            self.current_boxes[name] = box

        self.sliders = {}
        self.text_boxes = {}
        slider_specs = [
            ("x", xyz_ranges["x"], "m"),
            ("y", xyz_ranges["y"], "m"),
            ("z", xyz_ranges["z"], "m"),
            ("roll", rpy_range_deg, "deg"),
            ("pitch", rpy_range_deg, "deg"),
            ("yaw", rpy_range_deg, "deg"),
        ]
        y0 = 0.765
        dy = 0.064
        for index, (name, limits, unit) in enumerate(slider_specs):
            y = y0 - index * dy
            slider_axis = self.fig.add_axes([0.70, y, 0.20, 0.030])
            text_axis = self.fig.add_axes([0.915, y - 0.004, 0.065, 0.038])
            value = float(initial_display[index])
            value = float(np.clip(value, limits[0], limits[1]))
            slider = Slider(slider_axis, f"{name} [{unit}]", limits[0], limits[1], valinit=value)
            text_box = TextBox(text_axis, "", initial=f"{value:.4f}")
            slider.on_changed(lambda _value, key=name: self._on_slider_changed(key))
            text_box.on_submit(lambda text, key=name: self._on_text_submitted(key, text))
            self.sliders[name] = slider
            self.text_boxes[name] = text_box

        self.compute_button = Button(self.fig.add_axes([0.66, 0.25, 0.13, 0.045]), "Compute IK")
        self.use_solution_button = Button(self.fig.add_axes([0.82, 0.25, 0.15, 0.045]), "Use Solution")
        self.target_current_button = Button(self.fig.add_axes([0.66, 0.19, 0.13, 0.045]), "Target=Current")
        self.reset_button = Button(self.fig.add_axes([0.82, 0.19, 0.15, 0.045]), "Reset Joints")

        self.compute_button.on_clicked(self._on_compute_clicked)
        self.use_solution_button.on_clicked(self._on_use_solution_clicked)
        self.target_current_button.on_clicked(self._on_target_current_clicked)
        self.reset_button.on_clicked(self._on_reset_clicked)

        self.fig.suptitle("Franka IK Workbench: edit target XYZ/RPY, compute IK, inspect FK", fontsize=13)
        self._draw()
        self._on_compute_clicked(None)
        if self.save is not None:
            self.save.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self.save, dpi=160)
            print(f"Saved visualization: {self.save}")

    def show(self) -> None:
        self.plt.show()

    def control_values(self) -> np.ndarray:
        values = np.asarray([self.sliders[name].val for name in self.CONTROL_NAMES], dtype=np.float64)
        values[3:6] = np.deg2rad(values[3:6])
        return values

    def target_pose(self) -> np.ndarray:
        return xyzrpy_to_transform(self.control_values())

    def current_pose(self) -> np.ndarray:
        return forward_kinematics(self.current_q, self.solver.tool_z_offset)

    def _set_controls_from_transform(self, transform: np.ndarray) -> None:
        values = transform_to_xyzrpy(transform)
        values[3:6] = np.rad2deg(values[3:6])
        self._updating_controls = True
        try:
            for index, name in enumerate(self.CONTROL_NAMES):
                value = float(values[index])
                self.sliders[name].set_val(value)
                self.text_boxes[name].set_val(f"{value:.4f}")
        finally:
            self._updating_controls = False
        self._draw()

    def _on_slider_changed(self, key: str) -> None:
        if self._updating_controls:
            return
        value = float(self.sliders[key].val)
        self._updating_controls = True
        try:
            self.text_boxes[key].set_val(f"{value:.4f}")
        finally:
            self._updating_controls = False
        self._draw()

    def _on_text_submitted(self, key: str, text: str) -> None:
        if self._updating_controls:
            return
        try:
            value = float(text)
        except ValueError:
            value = float(self.sliders[key].val)
        value = float(np.clip(value, self.sliders[key].valmin, self.sliders[key].valmax))
        self._updating_controls = True
        try:
            self.sliders[key].set_val(value)
            self.text_boxes[key].set_val(f"{value:.4f}")
        finally:
            self._updating_controls = False
        self._draw()

    def _on_compute_clicked(self, _event) -> None:
        target_pose = self.target_pose()
        result = self.solver.solve(self.current_q, target_pose)
        self.solution_q = result.q
        print_ik_result(result, target_pose, self.solver.tool_z_offset)
        self._draw(result)

    def _on_use_solution_clicked(self, _event) -> None:
        if self.solution_q is None:
            return
        self.current_q = self.solution_q.copy()
        self.solution_q = None
        self._set_controls_from_transform(self.current_pose())

    def _on_target_current_clicked(self, _event) -> None:
        self._set_controls_from_transform(self.current_pose())

    def _on_reset_clicked(self, _event) -> None:
        self.current_q = sanitize_joint_position(PANDA_HOME_Q)
        self.solution_q = None
        self._set_controls_from_transform(self.current_pose())

    def _draw(self, result: IkResult | None = None) -> None:
        self.axis.cla()
        current_links = link_transforms(self.current_q, self.solver.tool_z_offset)
        current_points = np.stack([t[:3, 3] for t in current_links], axis=0)
        target_pose = self.target_pose()

        self.axis.plot(
            current_points[:, 0],
            current_points[:, 1],
            current_points[:, 2],
            "o--",
            color="0.50",
            label="current chain",
        )
        draw_frame(self.axis, current_links[-1], label="current")

        if self.solution_q is not None:
            solution_links = link_transforms(self.solution_q, self.solver.tool_z_offset)
            solution_points = np.stack([t[:3, 3] for t in solution_links], axis=0)
            self.axis.plot(
                solution_points[:, 0],
                solution_points[:, 1],
                solution_points[:, 2],
                "o-",
                color="tab:blue",
                label="IK solution chain",
            )
            draw_frame(self.axis, solution_links[-1], label="solution")

        self.axis.scatter(*target_pose[:3, 3], color="tab:red", s=80, marker="x", label="target")
        draw_frame(self.axis, target_pose, label="target")
        self.axis.set_xlabel("x [m]")
        self.axis.set_ylabel("y [m]")
        self.axis.set_zlabel("z [m]")
        self.axis.legend(loc="upper right")
        set_axes_equal(self.axis)
        self._update_status(result)
        self.fig.canvas.draw_idle()

    def _update_status(self, result: IkResult | None) -> None:
        current_xyzrpy = transform_to_xyzrpy(self.current_pose())
        current_xyzrpy[3:6] = np.rad2deg(current_xyzrpy[3:6])
        target_xyzrpy = self.control_values()
        target_xyzrpy[3:6] = np.rad2deg(target_xyzrpy[3:6])
        lines = [
            "Current FK xyz/rpy:",
            "  " + " ".join(f"{value: .4f}" for value in current_xyzrpy[:3])
            + " m | "
            + " ".join(f"{value: .1f}" for value in current_xyzrpy[3:6])
            + " deg",
            "Target xyz/rpy:",
            "  " + " ".join(f"{value: .4f}" for value in target_xyzrpy[:3])
            + " m | "
            + " ".join(f"{value: .1f}" for value in target_xyzrpy[3:6])
            + " deg",
        ]
        if result is not None:
            lines.extend(
                [
                    f"IK success={result.success} iters={result.iters}",
                    f"pos_err={result.pos_error_m * 1000.0:.2f} mm "
                    f"rot_err={math.degrees(result.rot_error_rad):.2f} deg",
                    "q rad: " + " ".join(f"{value:.3f}" for value in result.q),
                ]
            )
        elif self.solution_q is not None:
            lines.append("Last solution is shown in blue.")
        else:
            lines.append("Move sliders, then press Compute IK.")
        self.status_text.set_text("\n".join(lines))
        self._update_current_boxes(current_xyzrpy)

    def _update_current_boxes(self, current_xyzrpy: np.ndarray) -> None:
        values = current_xyzrpy.copy()
        names = self.CONTROL_NAMES
        self._updating_controls = True
        try:
            for index, name in enumerate(names):
                precision = 4 if index < 3 else 1
                self.current_boxes[name].set_val(f"{values[index]:.{precision}f}")
        finally:
            self._updating_controls = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Franka IK test for x y z qx qy qz qw target pose.")
    parser.add_argument(
        "target",
        nargs="*",
        type=float,
        help="Optional target pose: x y z qx qy qz qw. If omitted, starts at current FK pose.",
    )
    parser.add_argument(
        "--current-q",
        nargs=7,
        type=float,
        default=PANDA_HOME_Q.tolist(),
        help="Current 7D Franka joint position in radians. Defaults to PANDA_HOME_Q in this script.",
    )
    parser.add_argument("--max-iters", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--step-scale", type=float, default=0.8)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument(
        "--max-target-delta-m",
        type=float,
        default=0.0,
        help="Optional translational clamp from current FK pose. Use 0.06 to mirror droid_client_wfam.",
    )
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--position-tolerance", type=float, default=0.004)
    parser.add_argument("--orientation-tolerance", type=float, default=0.05)
    parser.add_argument("--tool-z-offset", type=float, default=0.107)
    parser.add_argument("--x-range", nargs=2, type=float, default=(-0.9, 0.9), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.9, 0.9), metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.0, 1.2), metavar=("MIN", "MAX"))
    parser.add_argument("--rpy-range-deg", nargs=2, type=float, default=(-180.0, 180.0), metavar=("MIN", "MAX"))
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the 3D figure.")
    parser.add_argument("--no-gui", action="store_true", help="Do not open the matplotlib GUI.")
    parser.add_argument("--static", action="store_true", help="Use the old static plot instead of the interactive workbench.")
    return parser.parse_args()


def target_pose_from_args(args: argparse.Namespace, current_q: np.ndarray) -> np.ndarray:
    if len(args.target) == 0:
        return forward_kinematics(current_q, args.tool_z_offset)
    if len(args.target) != 7:
        raise SystemExit("ERROR: target must be omitted or exactly: x y z qx qy qz qw")
    return pose_xyz_quat_to_transform(np.asarray(args.target, dtype=np.float64))


def validate_range(name: str, limits: tuple[float, float]) -> tuple[float, float]:
    low, high = float(limits[0]), float(limits[1])
    if not low < high:
        raise SystemExit(f"ERROR: {name} range must satisfy MIN < MAX, got {limits}")
    return low, high


def main() -> int:
    args = parse_args()
    current_q = sanitize_joint_position(args.current_q)
    target_pose = target_pose_from_args(args, current_q)
    solver = SimpleFrankaDlsIk(
        max_iters=args.max_iters,
        damping=args.damping,
        step_scale=args.step_scale,
        max_joint_step=args.max_joint_step,
        max_target_delta_m=args.max_target_delta_m,
        orientation_weight=args.orientation_weight,
        position_tolerance=args.position_tolerance,
        orientation_tolerance=args.orientation_tolerance,
        tool_z_offset=args.tool_z_offset,
    )

    if args.no_gui or args.static:
        result = solver.solve(current_q, target_pose)
        print_ik_result(result, target_pose, args.tool_z_offset)
        if not args.no_gui or args.save is not None:
            plot_static_result(
                current_q,
                result.q,
                target_pose,
                args.tool_z_offset,
                args.save,
                show=not args.no_gui,
            )
        return 0 if result.success else 2

    workbench = FrankaIkWorkbench(
        current_q=current_q,
        target_pose=target_pose,
        solver=solver,
        xyz_ranges={
            "x": validate_range("x", tuple(args.x_range)),
            "y": validate_range("y", tuple(args.y_range)),
            "z": validate_range("z", tuple(args.z_range)),
        },
        rpy_range_deg=validate_range("rpy", tuple(args.rpy_range_deg)),
        save=args.save,
    )
    workbench.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
