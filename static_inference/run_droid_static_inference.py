#!/usr/bin/env python3
"""Run native OXE-DROID static inference for GR00T-N1.5 on one trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
GR00T_ROOT = REPO_ROOT / "models/gr00t-n1.5"
STATIC_ROOT = REPO_ROOT / "static-inference"
sys.path[:0] = [str(STATIC_ROOT), str(REPO_ROOT), str(GR00T_ROOT)]

from droid.archive import DroidTrajectory, manifest_entry
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform, collate

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/gr00t/gr00t-n1.5"
EMBODIMENT_TAG = EmbodimentTag.OXE_DROID
HORIZON = 16
NUM_STEPS = 4
CHECKPOINT_IMAGE_SIZE = (256, 256)


def resize_for_checkpoint(image: np.ndarray) -> np.ndarray:
    """Match the square cached-image input used by the OXE-DROID checkpoint."""
    return cv2.resize(image, CHECKPOINT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--no-gradnorm", action="store_true")
    return parser.parse_args()


def build_transform(config, metadata):
    transform = config.transform()
    transform.set_metadata(metadata)
    transform.eval()
    groot_transform = next(value for value in transform.transforms if isinstance(value, GR00TTransform))
    groot_transform.training = True
    return transform, groot_transform


def main():
    args = parse_args()
    entry = manifest_entry(args.manifest, args.trajectory_index)
    config = load_data_config("oxe_droid")
    modality_config = config.modality_config()
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=config.transform(),
        device=args.device,
    )
    transform, groot_transform = build_transform(config, policy.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses = [[] for _ in range(NUM_STEPS)]
    cosines = [[] for _ in range(NUM_STEPS)]
    gradnorms = [[] for _ in range(NUM_STEPS)]
    targets, velocities = [], [[] for _ in range(NUM_STEPS)]

    with DroidTrajectory(entry["archive_path"]) as trajectory:
        frame_count = max(0, len(trajectory) - HORIZON + 1)
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)
        for frame in range(frame_count):
            cartesian_state = trajectory.arrays["observation_cartesian_position"][frame]
            action_chunk = trajectory.arrays["action_cartesian_velocity"][frame : frame + HORIZON]
            data = {
                "video.exterior_image_1": resize_for_checkpoint(
                    trajectory.image("exterior_image_1_left", frame)
                )[None],
                "video.exterior_image_2": resize_for_checkpoint(
                    trajectory.image("exterior_image_2_left", frame)
                )[None],
                "video.wrist_image": resize_for_checkpoint(
                    trajectory.image("wrist_image_left", frame)
                )[None],
                "state.eef_position": cartesian_state[:3][None],
                "state.eef_rotation": cartesian_state[3:6][None],
                "state.gripper_position": trajectory.arrays["observation_gripper_position"][frame][None],
                "action.eef_position_delta": action_chunk[:, :3],
                "action.eef_rotation_delta": action_chunk[:, 3:6],
                "action.gripper_position": trajectory.arrays["action_gripper_position"][frame : frame + HORIZON],
                "annotation.language.language_instruction": trajectory.prompt,
            }
            sample = transform(data)
            inputs = collate([sample], groot_transform.eagle_processor)
            result = policy.model.static_inference(inputs, compute_gradnorm=not args.no_gradnorm)
            for step in range(NUM_STEPS):
                losses[step].append(result["final_loss"][step])
                cosines[step].append(result["cosine"][step])
                gradnorms[step].append(result["gradnorm_vision"][step])
            if args.save_meta:
                mask = result["action_mask"][0]
                real_h = int(mask.any(dim=1).sum().item())
                real_d = int(mask.any(dim=0).sum().item())
                targets.append(result["u"][0, :real_h, :real_d].numpy())
                for step in range(NUM_STEPS):
                    velocities[step].append(result["v"][step][0, :real_h, :real_d].numpy())

    for step in range(NUM_STEPS):
        np.save(args.output_dir / f"final_loss_{step}.npy", np.asarray(losses[step], dtype=np.float32))
        np.save(args.output_dir / f"cosine_{step}.npy", np.asarray(cosines[step], dtype=np.float32))
        np.save(args.output_dir / f"gradnorm_vision_step_{step}.npy", np.asarray(gradnorms[step], dtype=np.float32))
    if args.save_meta:
        meta = args.output_dir / "meta"
        meta.mkdir(exist_ok=True)
        np.save(meta / "u.npy", np.stack(targets).astype(np.float32))
        for step in range(NUM_STEPS):
            np.save(meta / f"v_{step}.npy", np.stack(velocities[step]).astype(np.float32))
    (args.output_dir / "trajectory_meta.json").write_text(json.dumps({
        "model": "gr00t-n1.5",
        "embodiment": "oxe_droid",
        "trajectory_index": args.trajectory_index,
        "trajectory_id": entry["trajectory_id"],
        "source_length": entry["length"],
        "action_horizon": HORIZON,
        "num_frames_used": frame_count,
        "discarded_tail": min(HORIZON - 1, entry["length"]),
        "action_source": "cartesian_velocity + gripper_position",
        "language_field": "language_instruction",
        "source_image_resolution": [180, 320],
        "checkpoint_image_resolution": [256, 256],
        "checkpoint_image_resize": "cv2.INTER_LINEAR",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
