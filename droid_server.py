import dataclasses
import logging
import socket
import asyncio
import os
import http
import logging
import time
import traceback
import json
import copy
from pathlib import Path
import torch
import tyro
from einops import rearrange
import datetime

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig

logger = logging.getLogger(__name__)

CAMERA_MODEL_ORDER = (
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
)

CAMERA_INPUT_MAPPING = {
    "observation/exterior_image_0_left": "video.exterior_image_1_left",
    "observation/exterior_image_1_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}

CAMERA_DEBUG_LABELS = {
    "video.exterior_image_1_left": "view0_exterior_image_1",
    "video.exterior_image_2_left": "view1_exterior_image_2",
    "video.wrist_image_left": "view2_wrist",
}


def _safe_debug_name(value: str | None) -> str:
    if not value:
        return "no_session"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return cleaned[:120] or "no_session"


def _as_numpy(value):
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _clone_debug_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _clone_debug_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_debug_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_debug_value(v) for v in value)
    return copy.deepcopy(value)


def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and np.max(image) <= 1.0 and np.min(image) >= 0.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _save_image(path: Path, array: np.ndarray) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    image = _to_uint8_image(array)
    Image.fromarray(image).save(path)
    return True


def _save_contact_sheet(path: Path, frames: dict[str, np.ndarray]) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    processed = []
    for name, value in frames.items():
        image = _to_uint8_image(value)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        processed.append((name, Image.fromarray(image)))
    if not processed:
        return False
    width = sum(img.width for _, img in processed)
    height = max(img.height for _, img in processed) + 24
    canvas = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    offset_x = 0
    for name, img in processed:
        canvas.paste(img, (offset_x, 24))
        draw.text((offset_x + 4, 4), name, fill=(255, 255, 255))
        offset_x += img.width
    canvas.save(path)
    return True


def _make_droid_wan_stitch_layout(converted_obs: dict) -> np.ndarray | None:
    frames = []
    for key in CAMERA_MODEL_ORDER:
        value = converted_obs.get(key)
        if value is None:
            return None
        image = _to_uint8_image(value)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        frames.append(image)
    exterior_1, exterior_2, wrist = frames
    if exterior_1.shape != exterior_2.shape or exterior_1.shape[:2] != wrist.shape[:2]:
        return None
    height, width = exterior_1.shape[:2]
    canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    wrist_wide = np.repeat(wrist, 2, axis=1)
    canvas[:height, :] = wrist_wide[:, : width * 2]
    canvas[height:, :width] = exterior_1
    canvas[height:, width:] = exterior_2
    return canvas


@dataclasses.dataclass
class Args:
    port: int = 50532
    timeout_seconds: int = 50000
    model_path: str = "/team/datasets/DreamZero/DreamZero-DROID"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None
    frames_per_chunk: int = 4
    num_inference_steps: int | None = None
    num_dit_steps: int | None = None
    wan_cfg_scale: float | None = None
    wan_seed: int | None = None
    attention_backend: str | None = "TE"
    dynamic_cache_schedule: bool = False
    save_video_pred: bool = False
    video_output_dir: str | None = None
    wan_debug_dump_dir: str | None = None
    wan_debug_max_steps: int | None = None
    wan_debug_every_k: int | None = None
    decode_video_pred_debug: bool = False


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.
    
    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """
    
    # Number of frames to accumulate after the first call
    FRAMES_PER_CHUNK = 4
    
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        frames_per_chunk: int = 4,
        save_video_pred: bool = False,
        video_output_dir: str | None = None,
        debug_dump_dir: str | None = None,
        debug_max_steps: int | None = None,
        decode_video_pred_debug: bool = False,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir
        self._frames_per_chunk = max(1, int(frames_per_chunk))
        self._save_video_pred = save_video_pred
        self._video_output_dir = video_output_dir or output_dir
        self._debug_dump_dir = debug_dump_dir
        self._debug_max_steps = debug_max_steps if debug_max_steps is not None else 0
        self._decode_video_pred_debug = decode_video_pred_debug

        self._frame_buffers: dict[str, list[np.ndarray]] = {key: [] for key in CAMERA_MODEL_ORDER}
        self._call_count = 0
        self._is_first_call = True
        self._current_session_id: str | None = None
        self._current_session_debug_dir: Path | None = None
        self._current_session_debug_stamp: str | None = None
        self._session_index = 0
        self._session_step_index = 0
        self.video_across_time: list[torch.Tensor] = []
        self._msg_index = 0
        self._debug_step_index = 0

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
        if self._video_output_dir:
            os.makedirs(self._video_output_dir, exist_ok=True)
        if self._debug_dump_dir:
            os.makedirs(self._debug_dump_dir, exist_ok=True)
    
    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.
        
        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str
        
        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}
        
        # Map image keys (roboarena uses 0-indexed, AR_droid uses 1-indexed)
        image_key_mapping = CAMERA_INPUT_MAPPING
        
        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self._frames_per_chunk
        
        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
        
        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
        
        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""
        
        return converted
    
    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.
        
        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        
        return action
    
    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle
        
        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)
        
        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        
        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)
    
    def _decode_video_latents(self, latents: torch.Tensor) -> np.ndarray:
        action_head = self._policy.trained_model.action_head
        with torch.no_grad():
            frames = action_head.vae.decode(
                latents,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        return ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

    def _save_predicted_video(self) -> None:
        if not self._save_video_pred or not self.video_across_time or not self._video_output_dir:
            return
        try:
            latents = torch.cat(self.video_across_time, dim=2)
            frames = self._decode_video_latents(latents)
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            existing = [f for f in os.listdir(self._video_output_dir) if f.endswith(".mp4")]
            safe_session = _safe_debug_name(self._current_session_id)
            output_path = os.path.join(
                self._video_output_dir,
                f"{len(existing):06}_{safe_session}_{timestamp}.mp4",
            )
            try:
                imageio.mimsave(output_path, list(frames), fps=5, codec='libx264')
                logger.info("Saved video prediction (%d frames) to %s", len(frames), output_path)
            except Exception as exc:
                logger.warning("Failed to save mp4: %s; falling back to PNGs", exc)
                png_dir = os.path.join(self._video_output_dir, f"{len(existing):06}_{safe_session}_{timestamp}_frames")
                os.makedirs(png_dir, exist_ok=True)
                for idx, frame in enumerate(frames):
                    _save_image(Path(png_dir) / f"frame_{idx:04d}.png", frame)
                logger.info("Saved video prediction (%d frames) as PNGs to %s", len(frames), png_dir)
            contact_frames = {f"f{idx:02d}": frames[idx] for idx in range(min(len(frames), 24))}
            _save_contact_sheet(Path(self._video_output_dir) / f"{len(existing):06}_{safe_session}_{timestamp}_contact.png", contact_frames)
        except Exception as exc:
            logger.warning("Failed to save video prediction: %s", exc, exc_info=True)

    def _decode_prompt_tokens(self, token_ids, attention_mask, tokenizer=None) -> list[dict]:
        if tokenizer is None:
            tokenizer_wrapper = getattr(self._policy.eval_transform, "tokenizer", None)
            tokenizer = getattr(tokenizer_wrapper, "tokenizer", None)
        token_ids_np = _as_numpy(token_ids)
        attention_mask_np = _as_numpy(attention_mask)
        if tokenizer is None or token_ids_np is None or attention_mask_np is None:
            return []

        decoded = []
        for row_ids, row_mask in zip(token_ids_np, attention_mask_np):
            valid_length = int(np.asarray(row_mask).sum())
            valid_ids = [int(token) for token in np.asarray(row_ids)[:valid_length].tolist()]
            decoded.append(
                {
                    "token_count": valid_length,
                    "token_ids": valid_ids,
                    "decoded_text": tokenizer.decode(valid_ids, skip_special_tokens=True),
                    "decoded_with_special_tokens": tokenizer.decode(valid_ids, skip_special_tokens=False),
                }
            )
        return decoded

    def _find_language_transform(self, transform):
        if hasattr(transform, "_prepare_language") and hasattr(transform, "tokenizer"):
            return transform
        child_transforms = getattr(transform, "transforms", None)
        if child_transforms:
            for child in child_transforms:
                match = self._find_language_transform(child)
                if match is not None:
                    return match
        return None

    def _collect_text_encoder_prompt_debug(self, converted_obs: dict) -> dict:
        eval_transform = self._policy.eval_transform
        language_transform = self._find_language_transform(eval_transform)
        raw_user_prompt = converted_obs.get("annotation.language.action_text", "")

        positive_prompt_text = raw_user_prompt
        final_positive_prompt_text = raw_user_prompt
        negative_prompt_text = (
            "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, three legs, many people in the background, walking backwards."
        )
        is_lapa_instance = False
        is_dream_instance = False
        is_cotrain_instance = False
        tokenizer_wrapper = getattr(language_transform, "tokenizer", None) if language_transform is not None else None

        if language_transform is not None:
            positive_prompt_text, is_lapa_instance, is_dream_instance, is_cotrain_instance = language_transform._prepare_language(
                converted_obs
            )
            if hasattr(language_transform, "format_prompt_for_text_encoder"):
                final_positive_prompt_text = language_transform.format_prompt_for_text_encoder(positive_prompt_text)
            else:
                final_positive_prompt_text = positive_prompt_text

        positive_ids = positive_mask = negative_ids = negative_mask = None
        if tokenizer_wrapper is not None:
            positive_ids, positive_mask = tokenizer_wrapper(
                [final_positive_prompt_text], return_mask=True, add_special_tokens=True
            )
            negative_ids, negative_mask = tokenizer_wrapper(
                [negative_prompt_text], return_mask=True, add_special_tokens=True
            )
        tokenizer = getattr(tokenizer_wrapper, "tokenizer", None)

        action_head = self._policy.trained_model.action_head
        cfg_scale = float(getattr(action_head, "cfg_scale", 1.0))
        ip_size = int(getattr(action_head, "ip_size", 1))
        ip_rank = int(getattr(action_head, "ip_rank", 0))

        positive = self._decode_prompt_tokens(positive_ids, positive_mask, tokenizer=tokenizer)
        negative = self._decode_prompt_tokens(negative_ids, negative_mask, tokenizer=tokenizer)

        if ip_size > 1:
            effective_inputs = ["positive"] if ip_rank == 0 else ["negative"]
        else:
            effective_inputs = ["positive"]
            if cfg_scale != 1.0 and negative:
                effective_inputs.append("negative")

        return {
            "raw_user_prompt": raw_user_prompt,
            "prepared_positive_prompt_text": positive_prompt_text,
            "final_positive_prompt_text": final_positive_prompt_text,
            "prepared_negative_prompt_text": negative_prompt_text,
            "is_lapa_instance": bool(is_lapa_instance),
            "is_dream_instance": bool(is_dream_instance),
            "is_cotrain_instance": bool(is_cotrain_instance),
            "language_transform_class": type(language_transform).__name__ if language_transform is not None else None,
            "cfg_scale": cfg_scale,
            "ip_size": ip_size,
            "ip_rank": ip_rank,
            "effective_text_encoder_inputs": effective_inputs,
            "positive_prompt_inputs": positive,
            "negative_prompt_inputs": negative,
        }

    def _get_or_create_session_debug_dir(self) -> Path | None:
        if not self._debug_dump_dir:
            return None
        if self._current_session_debug_dir is not None:
            return self._current_session_debug_dir

        session_stamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        safe_session = _safe_debug_name(self._current_session_id)
        session_dir = Path(self._debug_dump_dir) / (
            f"session_{self._session_index:06d}_{session_stamp}_{safe_session}"
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        self._current_session_debug_stamp = session_stamp
        self._current_session_debug_dir = session_dir

        action_head = getattr(getattr(self._policy, "trained_model", None), "action_head", None)
        if action_head is not None:
            action_head.wan_debug_dir = str(session_dir)
            action_head._wan_debug_step_index = 0
            action_head._wan_infer_step_index = 0
        return session_dir

    def _start_new_session_debug_dir(self) -> None:
        self._current_session_debug_dir = None
        self._current_session_debug_stamp = None
        self._session_step_index = 0
        self._get_or_create_session_debug_dir()

    def _save_debug_step(self, converted_obs: dict, video_pred: torch.Tensor | None) -> None:
        if not self._debug_dump_dir:
            return
        if self._debug_max_steps and self._debug_step_index >= self._debug_max_steps:
            return
        session_dir = self._get_or_create_session_debug_dir()
        if session_dir is None:
            return
        step_dir = session_dir / f"step_{self._session_step_index:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        ordered_frames: dict[str, np.ndarray] = {}
        for key in CAMERA_MODEL_ORDER:
            value = converted_obs.get(key)
            if value is None:
                continue
            label = CAMERA_DEBUG_LABELS[key]
            ordered_frames[label] = value
            _save_image(step_dir / f"{label}.png", value)
        if ordered_frames:
            _save_contact_sheet(step_dir / "camera_contact_sheet.png", ordered_frames)
        stitched_preview = _make_droid_wan_stitch_layout(converted_obs)
        if stitched_preview is not None:
            _save_image(step_dir / "droid_wan_stitch_layout.png", stitched_preview)

        prompt_debug = self._collect_text_encoder_prompt_debug(converted_obs)
        summary = {
            "debug_step": self._debug_step_index,
            "session_timestamp": self._current_session_debug_stamp,
            "msg_index": self._msg_index,
            "session_id": self._current_session_id,
            "camera_model_order": list(CAMERA_MODEL_ORDER),
            "camera_input_mapping": CAMERA_INPUT_MAPPING,
            "wan_layout": {
                "top_row": "wrist duplicated across width",
                "bottom_left": "video.exterior_image_1_left",
                "bottom_right": "video.exterior_image_2_left",
            },
            "prompt": converted_obs.get("annotation.language.action_text", ""),
            "text_encoder_prompt_debug": prompt_debug,
        }
        with open(step_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        with open(step_dir / "text_encoder_inputs.json", "w") as handle:
            json.dump(prompt_debug, handle, indent=2)
        effective_inputs = prompt_debug.get("effective_text_encoder_inputs", [])
        with open(step_dir / "text_encoder_effective_inputs.txt", "w") as handle:
            for branch in effective_inputs:
                handle.write(f"[{branch}]\n")
                for item in prompt_debug.get(f"{branch}_prompt_inputs", []):
                    handle.write(item.get("decoded_text", "") + "\n\n")

        if video_pred is not None:
            video_pred_np = _as_numpy(video_pred)
            if video_pred_np is not None:
                np.savez_compressed(step_dir / "video_pred_latent.npz", video_pred=video_pred_np)
            if self._decode_video_pred_debug:
                try:
                    frames = self._decode_video_latents(video_pred)
                    batch_frames = {}
                    for idx, frame in enumerate(frames):
                        _save_image(step_dir / f"video_pred_decoded_frame_{idx:02d}.png", frame)
                        if idx < 24:
                            batch_frames[f"f{idx:02d}"] = frame
                    if batch_frames:
                        _save_contact_sheet(step_dir / "video_pred_decoded_contact_sheet.png", batch_frames)
                    imageio.mimsave(step_dir / "video_pred_decoded.mp4", list(frames), fps=5, codec='libx264')
                except Exception as exc:
                    logger.warning("Failed to decode/save video_pred debug: %s", exc, exc_info=True)
        self._debug_step_index += 1
        self._session_step_index += 1

    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations.
        
        Args:
            obs: Observation dict in roboarena format
            
        Returns:
            action: (N, 8) action array
        """
        # Check for session change - reset state if new session
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id
            self._session_index += 1
            self._start_new_session_debug_dir()
        elif session_id is None and self._current_session_id is None and self._current_session_debug_dir is None:
            self._current_session_id = f"auto_session_{self._session_index:06d}"
            self._session_index += 1
            self._start_new_session_debug_dir()
        
        self._msg_index += 1
        self._call_count += 1
        
        # Convert observation format
        converted_obs = self._convert_observation(obs)
        
        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)
        
        # Broadcast obs to workers
        self._broadcast_batch_to_workers(converted_obs)
        
        # Create batch for policy
        batch = Batch(obs=converted_obs)
        
        # Distributed forward pass
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()
        
        # Store video predictions for potential saving
        if video_pred is not None:
            self.video_across_time.append(video_pred)
        self._save_debug_step(converted_obs, video_pred)
        
        # Extract and convert action
        action_chunk_dict = result_batch.act
        
        # Convert Batch to dict
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        
        action = self._convert_action(action_dict)
        
        # Update first call flag
        if self._is_first_call:
            self._is_first_call = False
        
        return action
    
    def _reset_state(self, save_video: bool = True) -> None:
        """Internal method to reset policy state.

        Args:
            save_video: Whether to save accumulated video before reset.
        """
        if save_video:
            self._save_predicted_video()

        for key in self._frame_buffers:
            self._frame_buffers[key] = []

        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.
        
        Clears frame buffers and resets call count.
        """
        self._reset_state(save_video=True)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []
        self._msg_index = 0
        self._signal_group = signal_group
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)
    
    def _save_input_obs(self, obs: dict) -> None:
        """Save incoming observation images per message.
        
        Expected format: THWC (Time, Height, Width, Channel) with 4 frames.
        Saves each frame as a separate PNG image: HWC format (uint8).
        
        Directory structure:
        output_dir/inputs/{msg_index:06d}_{timestamp}/{obs_key}/f{frame_idx:02d}.png
        """
        if not self._output_dir:
            return
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        base_dir = os.path.join(self._output_dir, "inputs", f"{self._msg_index:06d}_{timestamp}")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            return

        for key in ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"):
            if key not in obs:
                continue
            value = obs[key]
            try:
                # Convert to numpy if tensor
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = np.asarray(value)
                
                # Expected format: THWC (Time, Height, Width, Channel)
                if arr.ndim != 4:
                    logger.warning(f"obs key '{key}' has shape {arr.shape}, expected 4D (T,H,W,C)")
                    continue
                
                # arr is (T, H, W, C)
                T, H, W, C = arr.shape
                
                # Normalize to uint8
                if arr.dtype == np.uint8:
                    frames_u8 = arr
                else:
                    f = arr.astype(np.float32)
                    # Common conventions: [-1,1] or [0,1]
                    min_val = float(np.nanmin(f))
                    max_val = float(np.nanmax(f))
                    if min_val >= -1.1 and max_val <= 1.1:
                        # Assume [-1,1] range
                        frames_u8 = ((f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        # Min-max scaling
                        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
                        frames_u8 = ((f - min_val) / denom * 255.0).clip(0, 255).astype(np.uint8)
                
                # Save each frame: frames_u8[i] is (H, W, C)
                key_dir = os.path.join(base_dir, key.replace("/", "_"))
                os.makedirs(key_dir, exist_ok=True)
                for frame_idx in range(T):
                    frame = frames_u8[frame_idx]  # (H, W, C)
                    # Handle grayscale (H, W) -> (H, W, 1)
                    if frame.ndim == 2:
                        frame = np.expand_dims(frame, axis=-1)
                    imageio.imwrite(os.path.join(key_dir, f"f{frame_idx:02d}.png"), frame)
                    
            except Exception as e:
                logger.warning(f"Failed to save obs key '{key}': {e}")
                continue



    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            # Non-rank-0 processes run a worker loop
            await self._worker_loop()

    async def _worker_loop(self):
        """Worker loop for non-rank-0 processes to participate in distributed inference."""
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                # Wait for obs broadcast from rank 0
                # Create a dummy obs dict structure - will be filled by broadcast
                # obs = {}

                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break

                # --- ADD THIS ELIF BLOCK ---
                elif signal == 2:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    # Loop back to the top and wait for the next signal
                    continue

                # Receive the batch data via broadcast/gather mechanism
                # This is a simplified version - the actual obs structure needs to be broadcasted
                batch = self._receive_batch_from_rank0()
                # Participate in distributed forward pass
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        """Receive batch data from rank 0 using torch.distributed primitives."""
        import pickle

        # Receive the size of the pickled data first
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        # Receive the actual data
        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        # Deserialize
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    def _broadcast_batch_to_workers(self, obs):
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle

        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        try:
            while True:
                try:
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    obs = msgpack_numpy.unpackb(data)
                    print(f"Wait Time: {recv_done - start_time:.2f} seconds")
                    self._msg_index += 1

                    infer_start_time = time.perf_counter()

                    # Signal other ranks to continue (0 = continue)
                    signal_tensor.zero_() 
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group) # <-- USE GLOO GROUP

                    # Broadcast the obs to all ranks for distributed inference
                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    # All ranks need to participate in the forward pass
                    dist.barrier()
                    forward_start_time = time.perf_counter()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Forward Time: {time.perf_counter() - forward_start_time:.2f} seconds")

                    action_chunk_dict = result_batch.act
                    video_chunk = video_pred

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")

                    self.video_across_time.append(video_chunk)

                    if len(self.video_across_time) > 10:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                        self.video_across_time = []
                    elif self._policy.trained_model.action_head.current_start_frame == 1 + self._policy.trained_model.action_head.num_frame_per_block and len(self.video_across_time) > 1:
                        print("current_start_frame == 1 + num_frame_per_block and len(self.video_across_time) > 1")
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time[:-1], dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        self.video_across_time = [video_chunk]

                    
                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out
                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    if len(self.video_across_time) > 0:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                    self.video_across_time = []
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            logger.info(f"Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)  # Set tensor value to 2
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            # When connection closes, signal other ranks to continue waiting for next connection
            # (or implement proper shutdown if needed)


def init_mesh() -> DeviceMesh:
    # env vars set by torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def main(args: Args) -> None:
    # Disable CUDA graphs to prevent cudagraph_trees allocator crash:
    # RuntimeError: Expected curr_block->next == nullptr
    # This crash occurs on the second+ VAE encode call when torch.compile uses
    # mode="reduce-overhead" (which enables CUDA graphs by default).
    import torch._inductor.config as inductor_cfg
    inductor_cfg.triton.cudagraphs = False

    # Set environment variable for DIT cache.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"

    if args.num_inference_steps is not None:
        os.environ["WAN_NUM_INFERENCE_STEPS"] = str(args.num_inference_steps)
    if args.num_dit_steps is not None:
        os.environ["NUM_DIT_STEPS"] = str(args.num_dit_steps)
    if args.wan_cfg_scale is not None:
        os.environ["WAN_CFG_SCALE"] = str(args.wan_cfg_scale)
    if args.wan_seed is not None:
        os.environ["WAN_SEED"] = str(args.wan_seed)
    if args.dynamic_cache_schedule:
        os.environ["DYNAMIC_CACHE_SCHEDULE"] = "True"
    if args.wan_debug_dump_dir:
        os.environ["WAN_DEBUG_DUMP_DIR"] = args.wan_debug_dump_dir
    if args.wan_debug_max_steps is not None:
        os.environ["WAN_DEBUG_MAX_STEPS"] = str(args.wan_debug_max_steps)
    if args.wan_debug_every_k is not None:
        os.environ["WAN_DEBUG_EVERY_K"] = str(args.wan_debug_every_k)

    attention_backend = args.attention_backend or "TE"
    os.environ["ATTENTION_BACKEND"] = attention_backend

    # Increase the recompile limit to 100 for inference due
    # to autoregressive nature of the model (several possible shapes).
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "oxe_droid"
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_name = os.path.basename(model_path)
    parent_dir = os.path.dirname(model_path)
    base_output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
    resolved_video_output_dir = args.video_output_dir or os.path.join(base_output_dir, "video_pred_output_droid")
    resolved_debug_dump_dir = args.wan_debug_dump_dir or os.path.join(base_output_dir, "debug_output_droid")
    os.environ.setdefault("WAN_DEBUG_DUMP_DIR", resolved_debug_dump_dir)

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    # Create server for all ranks - rank 0 handles websocket, others run worker loop
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        output_dir = base_output_dir
        os.makedirs(output_dir, exist_ok=True)
        video_output_dir = resolved_video_output_dir
        debug_dump_dir = resolved_debug_dump_dir
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(debug_dump_dir, exist_ok=True)
        logging.info("DROID output_dir=%s", output_dir)
        logging.info("DROID video_output_dir=%s", video_output_dir)
        logging.info("DROID debug_dump_dir=%s", debug_dump_dir)
        logging.info("DROID camera order verified: %s", CAMERA_MODEL_ORDER)
        logging.info("DROID camera input mapping: %s", CAMERA_INPUT_MAPPING)
    else:
        output_dir = None
        video_output_dir = None
        debug_dump_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")

    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
        frames_per_chunk=args.frames_per_chunk,
        save_video_pred=args.save_video_pred,
        video_output_dir=video_output_dir,
        debug_dump_dir=debug_dump_dir,
        debug_max_steps=args.wan_debug_max_steps,
        decode_video_pred_debug=args.decode_video_pred_debug,
    )
    
    # Configure server for AR_droid (2 external cameras, wrist camera, joint position actions)
    server_config = PolicyServerConfig(
        image_resolution=(180, 320),  # AR_droid expects 180x320 images
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # Track session to reset state for new clients
        action_space="joint_position",
    )
    
    if rank == 0:
        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0 processes need to run worker loop for distributed inference
        # We'll use the existing WebsocketPolicyServer's worker loop mechanism
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
