import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np

from typing import Any, List
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg

from .nvidia_droid import NVIDIA_DROID

DATA_PATH = Path(__file__).parent / "../../../assets/"
_WRIST_FT_BODY_INDEX_CACHE: dict[int, tuple[int, str]] = {}


def _resolve_wrist_ft_body_index(robot: Any) -> tuple[int, str]:
    cache_key = id(robot)
    cached = _WRIST_FT_BODY_INDEX_CACHE.get(cache_key, None)
    if cached is not None:
        return cached

    names = [str(name) for name in getattr(robot, "body_names", [])]
    names_l = [name.lower() for name in names]

    # body_incoming_joint_wrench_b is indexed by child body. Prefer the gripper
    # base so the reported wrench corresponds to the fixed wrist/gripper joint.
    preferred_token_sets = (
        ("robotiq", "base"),
        ("gripper", "base"),
        ("base_link",),
        ("panda_hand",),
        ("panda_link8",),
        ("hand",),
        ("wrist",),
    )
    for tokens in preferred_token_sets:
        for index, name in enumerate(names_l):
            if all(token in name for token in tokens):
                out = (int(index), names[index])
                _WRIST_FT_BODY_INDEX_CACHE[cache_key] = out
                return out

    fallback = (len(names) - 1, names[-1]) if names else (-1, "")
    _WRIST_FT_BODY_INDEX_CACHE[cache_key] = fallback
    return fallback


def wrist_wrench_local(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    body_index, body_name = _resolve_wrist_ft_body_index(robot)
    if body_index < 0:
        return torch.zeros(6, device=robot.device, dtype=torch.float32)

    wrench = getattr(robot.data, "body_incoming_joint_wrench_b", None)
    if wrench is None:
        return torch.zeros(6, device=robot.device, dtype=torch.float32)

    if not getattr(wrist_wrench_local, "_source_printed", False):
        print(
            "[DROID] wrist wrench source: "
            f"body_incoming_joint_wrench_b body='{body_name}' index={body_index}",
            flush=True,
        )
        wrist_wrench_local._source_printed = True

    return wrench[0, body_index, :6].to(dtype=torch.float32)

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = NVIDIA_DROID

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )

    external_cam_2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_2",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    def dynamic_scene(self, scene_name: str, scene_path: str | Path | None = None):
        if scene_path is not None:
            environment_path = Path(scene_path)
        else:
            environment_path = DATA_PATH / f"scene{scene_name}.usd"
        if not environment_path.exists() and str(scene_name) == "5":
            environment_path = DATA_PATH / "scene4.usd"
        scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/scene",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=str(environment_path),
                    ),
                )
        self.scene = scene

        stage = Usd.Stage.Open(
            str(environment_path)
        )
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            # if rigid body
            if not UsdPhysics.RigidBodyAPI(child):
                continue

            name = child.GetName()
            print(f"Found rigid body: {name}")
            pos_attr = child.GetAttribute("xformOp:translate")
            pos = pos_attr.Get() if pos_attr and pos_attr.IsValid() else None
            if pos is None:
                pos = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()

            rot_attr = child.GetAttribute("xformOp:orient")
            rot = rot_attr.Get() if rot_attr and rot_attr.IsValid() else None
            if rot is None:
                rot = (1.0, 0.0, 0.0, 0.0)
            else:
                rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])

            pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            asset = RigidObjectCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                        spawn=None,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=pos,
                            rot=rot,
                        ),
                    )
            setattr(self, name, asset)


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction

@configclass
class ActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        wrist_wrench = ObsTerm(func=wrist_wrench_local)
        external_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        external_cam_2 = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_2"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 8
        self.sim.dt = 1 / (15 * 8)
        self.sim.render_interval = self.decimation

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.rerender_on_reset = True

    
    def set_scene(self, scene_name: str, scene_path: str | Path | None = None):
        self.scene.dynamic_scene(scene_name, scene_path=scene_path)
