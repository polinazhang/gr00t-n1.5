"""Run static inference for one RoboCasa atomic seen task with GR00T-N1.5.

Static inference is analysis-only: for each frame of each demonstration episode,
the model runs its standard 4-step flow-matching denoising loop starting from a
single noise tensor, and the per-step velocity predictions are compared against
the ground-truth action chunk (u = gt_action - noise). Per-frame scalars are
stacked per episode and saved as npy files.

Output layout (see README.md for full semantics):

    <output_root>/<timestamp>/task_<i>/episode_<j:06d>/
        final_loss_{n}.npy             (T_ep,)  always saved, n in 0..3
        cosine_{n}.npy                 (T_ep,)  always saved
        gradnorm_vision_step_{n}.npy   (T_ep,)  always saved
        meta/u.npy                     (T_ep, 16, 12)  only with --save_meta
        meta/v_{n}.npy                 (T_ep, 16, 12)  only with --save_meta
    <output_root>/<timestamp>/task_<i>/summary.json

Frames per episode: t = 0 .. ep_len - 16 (full 16-step GT action chunk available,
matching training's valid steps, no end padding). Axis 0 of every file of an
episode is this frame index.

Example:
    python static_inference/run_static_inference.py --task_id 2 --save_meta
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
GR00T_ROOT = REPO_ROOT / "models" / "gr00t-n1.5"
INFERENCE_MODELS_DIR = REPO_ROOT / "inference" / "models"  # for n15_data_config
sys.path.insert(0, str(INFERENCE_MODELS_DIR))
sys.path.insert(0, str(GR00T_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform, collate

DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "robocasa365" / "atomic-seen-splits"
# Fixed base model. No fallback of any kind: static inference must run on this
# exact checkpoint so all 18 tasks are compared against a common base model.
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "gr00t" / "gr00t-n1.5"
DEFAULT_OUTPUT_ROOT = Path("/coc/testnvme/xzhang3205/static/gr00t-n1.5")

EMBODIMENT_TAG = EmbodimentTag.NEW_EMBODIMENT
NUM_DENOISE_STEPS = 4  # gr00t default num_inference_timesteps
ACTION_HORIZON = 16  # real action horizon (robocasa action delta indices 0..15)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task_id", type=int, default=None, help="Task id, 1..18 (not needed with --dataset_dir)")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Direct LeRobot task dir; overrides dataset_root/task_{task_id}_demo_50",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint dir. Default (and intended): the fixed local base model "
        f"{DEFAULT_CHECKPOINT}. No fallback exists.",
    )
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--save_meta", action="store_true", help="Also save meta/u.npy and meta/v_{n}.npy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=None, help="Debug: cap number of episodes")
    parser.add_argument("--max_frames", type=int, default=None, help="Debug: cap frames per episode")
    parser.add_argument("--timestamp", type=str, default=None, help="Override output timestamp dir name")
    parser.add_argument("--no_gradnorm", action="store_true", help="Skip vision grad norm computation")
    return parser.parse_args()


def build_frame_transform(cfg, metadata):
    """Build the runner's OWN transform instance (separate from the policy's).

    The whole chain is first set to eval mode (deterministic preprocessing:
    center crop instead of random crop, no color jitter — identical to what
    Gr00tPolicy.__init__ does via `self._modality_transform.eval()` in normal
    inference). Then ONLY the final GR00TTransform is switched back to training
    mode so the ground-truth `action`/`action_mask` flow through (in eval mode
    it drops them). GR00TTransform.training gates nothing else here: language
    dropout prob is 0 in this data config, and StateActionTransform
    normalization is mode-independent.
    """
    transform = cfg.transform()
    transform.set_metadata(metadata)
    transform.eval()  # deterministic video/state preprocessing, as in normal inference
    groot_tf = None
    for t in transform.transforms:
        if isinstance(t, GR00TTransform):
            groot_tf = t
    assert groot_tf is not None, "GR00TTransform not found in the transform chain"
    groot_tf.training = True
    return transform, groot_tf


def build_frame_inputs(transform, groot_tf, data_point):
    """Run one raw dataset step through the transform chain and collate to a
    batch of 1 (mirrors GR00TTransform.apply_batch). GR00T_N1_5.prepare_input
    handles device/dtype inside model.static_inference."""
    sample = transform(data_point)
    collated = collate([sample], groot_tf.eagle_processor)
    return collated


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset_dir is None:
        assert args.task_id is not None and 1 <= args.task_id <= 18, f"task_id must be in 1..18, got {args.task_id}"
        dataset_path = Path(args.dataset_root) / f"task_{args.task_id}_demo_50"
    else:
        dataset_path = Path(args.dataset_dir)
    assert dataset_path.is_dir(), f"Dataset not found: {dataset_path}"

    assert Path(args.checkpoint).is_dir(), (
        f"Checkpoint not found: {args.checkpoint}. Static inference requires the "
        "fixed local base model; no download/fallback is permitted."
    )
    logging.info(f"Checkpoint: {args.checkpoint}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dataset_dir is not None and args.task_id is None:
        # --dataset_dir mode: output_root/timestamp already identifies the task.
        task_output_dir = Path(args.output_root) / timestamp
    else:
        task_output_dir = Path(args.output_root) / timestamp / f"task_{args.task_id}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir: {task_output_dir}")

    cfg = load_data_config("n15_data_config:PandaOmronDataConfig")
    modality_configs = cfg.modality_config()
    assert "action" in modality_configs, "Modality configs must include the action modality"

    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_configs,
        modality_transform=cfg.transform(),
        device=args.device,
    )
    # Normalization statistics come from the checkpoint's
    # experiment_cfg/metadata.json (loaded by Gr00tPolicy._load_metadata).
    # NEVER recompute statistics.
    transform, groot_tf = build_frame_transform(cfg, policy.metadata)

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=EMBODIMENT_TAG,
        video_backend="decord",  # torchcodec is not installed in the n1.5 venv; decord is the gr00t default
        transforms=None,  # raw per-step data; the runner applies its own transform
    )
    num_episodes = len(dataset.trajectory_ids)
    if args.max_episodes is not None:
        num_episodes = min(num_episodes, args.max_episodes)
    logging.info(f"Dataset: {dataset_path} ({len(dataset.trajectory_ids)} episodes, running {num_episodes})")

    # Per-task aggregation over all frames of all episodes.
    task_final_loss = [[] for _ in range(NUM_DENOISE_STEPS)]
    task_cosine = [[] for _ in range(NUM_DENOISE_STEPS)]
    task_gradnorm = [[] for _ in range(NUM_DENOISE_STEPS)]
    total_frames = 0

    for ep_idx in range(num_episodes):
        traj_id = int(dataset.trajectory_ids[ep_idx])
        ep_len = int(dataset.trajectory_lengths[ep_idx])
        # t = 0 .. ep_len - 16 inclusive: full GT action chunk available.
        frame_indices = list(range(0, ep_len - ACTION_HORIZON + 1))
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]
        if not frame_indices:
            logging.warning(f"Episode {ep_idx} (traj {traj_id}): len {ep_len} < {ACTION_HORIZON}, skipping")
            continue

        ep_final_loss = [[] for _ in range(NUM_DENOISE_STEPS)]
        ep_cosine = [[] for _ in range(NUM_DENOISE_STEPS)]
        ep_gradnorm = [[] for _ in range(NUM_DENOISE_STEPS)]
        ep_u = []
        ep_v = [[] for _ in range(NUM_DENOISE_STEPS)]

        for t in frame_indices:
            data_point = dataset.get_step_data(traj_id, t)
            collated = build_frame_inputs(transform, groot_tf, data_point)
            result = policy.model.static_inference(
                collated, compute_gradnorm=not args.no_gradnorm
            )

            # Derive real action dims from the mask (horizon 16 x dim 12 for robocasa).
            mask = result["action_mask"][0]  # (action_horizon, max_action_dim), 0/1
            real_h = int(mask.any(dim=1).sum().item())
            real_d = int(mask.any(dim=0).sum().item())

            for n in range(NUM_DENOISE_STEPS):
                ep_final_loss[n].append(result["final_loss"][n])
                ep_cosine[n].append(result["cosine"][n])
                ep_gradnorm[n].append(result["gradnorm_vision"][n])
            if args.save_meta:
                ep_u.append(result["u"][0, :real_h, :real_d].numpy())
                for n in range(NUM_DENOISE_STEPS):
                    ep_v[n].append(result["v"][n][0, :real_h, :real_d].numpy())

            if t % 50 == 0:
                logging.info(
                    f"task {args.task_id} ep {ep_idx} frame {t}/{frame_indices[-1]} "
                    f"cos={['%.4f' % result['cosine'][n] for n in range(NUM_DENOISE_STEPS)]}"
                )

        ep_dir = task_output_dir / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        for n in range(NUM_DENOISE_STEPS):
            np.save(ep_dir / f"final_loss_{n}.npy", np.asarray(ep_final_loss[n], dtype=np.float32))
            np.save(ep_dir / f"cosine_{n}.npy", np.asarray(ep_cosine[n], dtype=np.float32))
            np.save(
                ep_dir / f"gradnorm_vision_step_{n}.npy",
                np.asarray(ep_gradnorm[n], dtype=np.float32),
            )
        if args.save_meta:
            meta_dir = ep_dir / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            np.save(meta_dir / "u.npy", np.stack(ep_u).astype(np.float32))
            for n in range(NUM_DENOISE_STEPS):
                np.save(meta_dir / f"v_{n}.npy", np.stack(ep_v[n]).astype(np.float32))

        for n in range(NUM_DENOISE_STEPS):
            task_final_loss[n].extend(ep_final_loss[n])
            task_cosine[n].extend(ep_cosine[n])
            task_gradnorm[n].extend(ep_gradnorm[n])
        total_frames += len(frame_indices)
        logging.info(
            f"Episode {ep_idx} done: {len(frame_indices)} frames, "
            f"mean cosine step0={np.mean(ep_cosine[0]):.4f}"
        )

    summary = {
        "task_id": args.task_id,
        "dataset": str(dataset_path),
        "checkpoint": str(args.checkpoint),
        "timestamp": timestamp,
        "num_episodes": num_episodes,
        "total_frames": total_frames,
        "num_denoise_steps": NUM_DENOISE_STEPS,
        "save_meta": args.save_meta,
        "mean_final_loss": {str(n): float(np.mean(task_final_loss[n])) for n in range(NUM_DENOISE_STEPS)},
        "mean_cosine": {str(n): float(np.mean(task_cosine[n])) for n in range(NUM_DENOISE_STEPS)},
        "mean_gradnorm_vision": {
            str(n): float(np.nanmean(task_gradnorm[n])) for n in range(NUM_DENOISE_STEPS)
        },
    }
    with open(task_output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Task {args.task_id} done. Summary: {json.dumps(summary['mean_cosine'])}")


if __name__ == "__main__":
    main()
