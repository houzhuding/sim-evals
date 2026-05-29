#!/usr/bin/env python3
"""Local smoke test for droid_client.py without Isaac or a policy server."""

from __future__ import annotations

import uuid
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from droid_client import SimDroidPolicyClient  # noqa: E402


class FakePolicyServerClient:
    def __init__(self) -> None:
        self.requests = []
        self.num_infer_calls = 0

    def infer(self, request: dict) -> np.ndarray:
        self.requests.append(request)
        self.num_infer_calls += 1
        base = float(self.num_infer_calls)
        actions = np.zeros((4, 8), dtype=np.float32)
        actions[:, :7] = base
        actions[:, 7] = [0.0, 1.0, 0.25, 0.75]
        return actions

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
            "arm_joint_pos": np.arange(7, dtype=np.float32) + step,
            "gripper_pos": np.array([step % 2], dtype=np.float32),
        }
    }


def make_client(fake_server: FakePolicyServerClient) -> SimDroidPolicyClient:
    client = SimDroidPolicyClient.__new__(SimDroidPolicyClient)
    client.client = fake_server
    client.prompt = "put the cube in the bowl"
    client.open_loop_horizon = 4
    client.session_id = str(uuid.uuid4())
    client.pred_action_chunk = None
    client.actions_from_chunk_completed = 0
    client.obs_history = []
    return client


def main() -> None:
    fake_server = FakePolicyServerClient()
    client = make_client(fake_server)

    returned_actions = []
    for step in range(6):
        result = client.infer(make_fake_obs(step))
        returned_actions.append(result["action"])
        assert result["action"].shape == (8,)
        assert result["viz"].shape == (180, 960, 3)

    assert fake_server.num_infer_calls == 2

    first_request = fake_server.requests[0]
    assert first_request["endpoint"] == "infer"
    assert first_request["observation/exterior_image_0_left"].shape == (180, 320, 3)
    assert first_request["observation/exterior_image_1_left"].shape == (180, 320, 3)
    assert first_request["observation/wrist_image_left"].shape == (180, 320, 3)
    assert first_request["observation/joint_position"].shape == (7,)
    assert first_request["observation/gripper_position"].shape == (1,)
    assert first_request["prompt"] == "put the cube in the bowl"

    second_request = fake_server.requests[1]
    assert second_request["observation/exterior_image_0_left"].shape == (4, 180, 320, 3)
    assert second_request["observation/exterior_image_1_left"].shape == (4, 180, 320, 3)
    assert second_request["observation/wrist_image_left"].shape == (4, 180, 320, 3)

    # The fake action chunk has gripper values [0, 1, .25, .75]; client binarizes them.
    assert returned_actions[0][-1] == 0.0
    assert returned_actions[1][-1] == 1.0
    assert returned_actions[2][-1] == 0.0
    assert returned_actions[3][-1] == 1.0

    old_session_id = client.session_id
    client.reset()
    assert client.session_id != old_session_id
    assert client.pred_action_chunk is None
    assert client.actions_from_chunk_completed == 0
    assert client.obs_history == []

    print("droid_client local smoke test passed")


if __name__ == "__main__":
    main()
