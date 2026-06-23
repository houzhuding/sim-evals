#!/usr/bin/env python3
"""Local smoke test for droid_client_wfam.py without Isaac or a policy server."""

from __future__ import annotations

import uuid
from pathlib import Path
import sys
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if "websockets" not in sys.modules:
    websockets_mod = types.ModuleType("websockets")
    websockets_mod.__path__ = []
    websockets_sync_mod = types.ModuleType("websockets.sync")
    websockets_sync_mod.__path__ = []
    websockets_client_mod = types.ModuleType("websockets.sync.client")
    websockets_sync_mod.client = websockets_client_mod
    websockets_mod.sync = websockets_sync_mod
    sys.modules["websockets"] = websockets_mod
    sys.modules["websockets.sync"] = websockets_sync_mod
    sys.modules["websockets.sync.client"] = websockets_client_mod

if "openpi_client" not in sys.modules:
    openpi_client_mod = types.ModuleType("openpi_client")
    msgpack_numpy_mod = types.ModuleType("openpi_client.msgpack_numpy")

    class _FakePacker:
        def pack(self, value):
            return value

    msgpack_numpy_mod.Packer = _FakePacker
    msgpack_numpy_mod.unpackb = lambda value: value
    openpi_client_mod.msgpack_numpy = msgpack_numpy_mod
    sys.modules["openpi_client"] = openpi_client_mod
    sys.modules["openpi_client.msgpack_numpy"] = msgpack_numpy_mod

from droid_client_wfam import FrankaPose9DIKSolver, SimDroidWFAMPolicyClient  # noqa: E402


def valid_pose(offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [0.35 + offset, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )


def valid_joint_q() -> np.ndarray:
    return np.asarray([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.0], dtype=np.float32)


class FakeWFAMPolicyServerClient:
    def __init__(self) -> None:
        self.requests = []
        self.num_infer_calls = 0

    def infer(self, request: dict) -> dict[str, np.ndarray]:
        self.requests.append(request)
        self.num_infer_calls += 1
        base = float(self.num_infer_calls)
        pose = np.stack([valid_pose(0.005 * base)] * 4, axis=0)
        gripper = np.asarray([[0.0], [1.0], [0.25], [0.75]], dtype=np.float32)
        wrench = np.full((4, 6), base + 10.0, dtype=np.float32)
        return {
            "action.ee_pose9d_rot6d": pose,
            "action.gripper_position": gripper,
            "action.wrist_wrench": wrench,
        }

    def reset(self) -> None:
        pass


def make_fake_obs(step: int) -> dict:
    h, w = 180, 320

    def image(offset: int) -> np.ndarray:
        return np.full((1, h, w, 3), step + offset, dtype=np.uint8)

    return {
        "policy": {
            "external_cam": image(0),
            "external_cam_2": image(10),
            "wrist_cam": image(20),
            "arm_joint_pos": valid_joint_q(),
            "ee_pose9d_rot6d": valid_pose(0.001 * step),
            "gripper_pos": np.array([step % 2], dtype=np.float32),
            "wrist_wrench": np.arange(6, dtype=np.float32) + 200 + step,
        }
    }


def make_client(fake_server: FakeWFAMPolicyServerClient) -> SimDroidWFAMPolicyClient:
    client = SimDroidWFAMPolicyClient.__new__(SimDroidWFAMPolicyClient)
    client.client = fake_server
    client.prompt = "insert the blue peg"
    client.open_loop_horizon = 4
    client.lock_gripper_close = False
    client.latch_gripper_after_close = False
    client.execution_mode = "ik-joint"
    client.allow_missing_ee_pose = False
    client.ik_solver = FrankaPose9DIKSolver(max_iters=2)
    client.dump_action_chunks = False
    client.print_action_chunk_matrices = False
    client.action_chunk_dump_dir = None
    client.gripper_latched_closed = False
    client.session_id = str(uuid.uuid4())
    client.pred_action_chunk = None
    client.actions_from_chunk_completed = 0
    client.obs_history = []
    client.chunk_index = -1
    client.last_chunk_infer_time_s = float("nan")
    client.trace_dir = None
    client._trace_file = None
    client._trace_writer = None
    return client


def main() -> None:
    solver = FrankaPose9DIKSolver(max_iters=2)
    q0 = valid_joint_q()
    same_q, info = solver.solve(q0, valid_pose(), valid_pose())
    assert same_q.shape == (7,)
    assert np.all(np.isfinite(same_q))
    assert info["success"]

    fake_server = FakeWFAMPolicyServerClient()
    client = make_client(fake_server)

    returned_actions = []
    predicted_wrenches = []
    for step in range(6):
        result = client.infer(make_fake_obs(step))
        returned_actions.append(result["action"])
        predicted_wrenches.append(result["trace"]["predicted_wrench"])
        assert result["action"].shape == (8,)
        assert result["viz"].shape == (180, 960, 3)

    assert fake_server.num_infer_calls == 2

    first_request = fake_server.requests[0]
    assert first_request["endpoint"] == "infer"
    assert first_request["observation/exterior_image_0_left"].shape == (180, 320, 3)
    assert first_request["observation/exterior_image_1_left"].shape == (180, 320, 3)
    assert first_request["observation/wrist_image_left"].shape == (180, 320, 3)
    assert first_request["observation/joint_position"].shape == (7,)
    assert first_request["observation/ee_pose9d_rot6d"].shape == (9,)
    assert first_request["observation/gripper_position"].shape == (1,)
    assert first_request["observation/wrist_wrench"].shape == (6,)
    assert first_request["prompt"] == "insert the blue peg"

    second_request = fake_server.requests[1]
    assert second_request["observation/exterior_image_0_left"].shape == (4, 180, 320, 3)
    assert second_request["observation/exterior_image_1_left"].shape == (4, 180, 320, 3)
    assert second_request["observation/wrist_image_left"].shape == (4, 180, 320, 3)

    assert returned_actions[0][-1] == 0.0
    assert returned_actions[1][-1] == 1.0
    assert returned_actions[2][-1] == 0.0
    assert returned_actions[3][-1] == 1.0
    assert predicted_wrenches[0].shape == (6,)
    assert np.allclose(predicted_wrenches[0], np.full(6, 11.0, dtype=np.float32))

    old_session_id = client.session_id
    client.reset()
    assert client.session_id != old_session_id
    assert client.pred_action_chunk is None
    assert client.actions_from_chunk_completed == 0
    assert client.obs_history == []

    print("droid_client_wfam local smoke test passed")


if __name__ == "__main__":
    main()
