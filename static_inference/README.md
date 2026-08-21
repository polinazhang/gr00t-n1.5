# Static Inference for GR00T-N1.5

Analysis-only mode (see `prompts/static-inference-context.md`): given RoboCasa
demonstration frames, the model runs its standard 4-step flow-matching denoising
loop starting from a single noise tensor (drawn exactly like
`FlowmatchingActionHead.get_action`) and never executes actions. The per-step
velocity predictions are compared against the ground-truth action chunk.

- Model: `/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/gr00t/gr00t-n1.5`
  — the fixed local base model, always. There is NO fallback of any kind (no HF
  download, no alternate checkpoint): all 18 tasks must be compared against the
  same base model. The runner hard-fails if this directory is missing.
  The checkpoint's own `experiment_cfg/metadata.json`
  provides the normalization statistics (loaded by `Gr00tPolicy._load_metadata`)
  — statistics are NEVER recomputed. Embodiment `NEW_EMBODIMENT`
  (`new_embodiment`).
- Datasets:
  `/coc/testnvme/xzhang3205/vla-adaptation/datasets/robocasa365/atomic-seen-splits/task_{i}_demo_50`,
  `i` in 1..18.
- Padding: actions are padded to `action_horizon=16`, `max_action_dim=32` by
  `GR00TTransform`; the real action chunk is horizon 16 x dim 12 (robocasa
  action keys), covered by `action_mask`. All losses/cosines/grad norms are
  masked to the real dims.

## Running

```bash
# single task locally
python static_inference/run_static_inference.py --task_id 2 --save_meta

# debug: 1 episode, 2 frames
python static_inference/run_static_inference.py --task_id 2 \
    --max_episodes 1 --max_frames 2 --save_meta

# slurm: one job per task (1 node, 32 cpus, 1x A40, qos long, 3 days, kira-lab;
# 1 GPU because the runner is a single-process, single-GPU workload)
python static_inference/launch_static_inference.py --task_id 2 --extra_args "--save_meta"
python static_inference/launch_static_inference.py --all --extra_args "--save_meta"
```

Slurm logs: `/coc/testnvme/xzhang3205/vla-adaptation/slurms/`. Generated sbatch
scripts: `static_inference/generated_sbatch/`.

## Output layout

```
<coc/testnvme/xzhang3205/static/gr00t-n1.5>/<timestamp>/task_<i>/
    summary.json                       per-task means over all frames
    episode_<j:06d>/                   j = episode index (order of meta/episodes.jsonl)
        final_loss_{n}.npy             (T_ep,) float32, ALWAYS saved
        cosine_{n}.npy                 (T_ep,) float32, ALWAYS saved
        gradnorm_vision_step_{n}.npy   (T_ep,) float32, ALWAYS saved
        meta/                          only with --save_meta
            u.npy                      (T_ep, 16, 12) float32
            v_{n}.npy                  (T_ep, 16, 12) float32
```

`n` in 0..3 is the denoising step index; `<timestamp>` is `%Y%m%d_%H%M%S` of the
run start (one run = one task). `--save_meta` gates ONLY `meta/u.npy` and
`meta/v_{n}.npy`; the scalar files are always written.

## Stacking semantics

- Frames per episode: `t = 0 .. ep_len - 16` inclusive (the full 16-step
  ground-truth action chunk `[t, t+16)` is available — matching training's
  valid steps, no end padding). `T_ep = ep_len - 15`.
- **Axis 0 of every file in an episode directory is the frame index `t`**, and is
  aligned across all files of that episode (`final_loss_2.npy[k]`,
  `cosine_2.npy[k]`, `gradnorm_vision_step_2.npy[k]`, `meta/u.npy[k]`,
  `meta/v_2.npy[k]` all describe frame `t = k`).
- Index `n` is the n-th denoising step at `t_flow = n/4` in gr00t's convention
  (`num_inference_timesteps = 4`, `t_flow = 0, 0.25, 0.5, 0.75`). The latent at
  step 0 is pure noise.
- Per frame, ONE noise tensor `eps ~ N(0, I)` of shape `(16, 32)` (padded
  action space) is drawn exactly as in `get_action` and reused as the `t=0`
  latent for all 4 steps and all downstream computations.
- `u = gt_action - noise` (gr00t's velocity convention `v = actions - noise`),
  computed in the padded `(16, 32)` normalized action space and cropped to the
  real dims `(16, 12)` for storage.
- `v_{n}` is the model's velocity prediction at denoising step `n` for the same
  noise/latent trajectory, likewise cropped to `(16, 12)`.
- `final_loss_{n}` is the masked MSE between `v_{n}` and `u` over valid elements
  only — `sum((v_n - u)^2 * mask) / mask.sum()` — identical semantics to the
  n1.5 training loss in `FlowmatchingActionHead.forward` (note: n1.5's training
  loss has NO `+1e-6` in the denominator, so the static path drops it too and
  asserts `mask.sum() > 0` instead).
- `cosine_{n}` is the cosine similarity between `v_{n}` and `u` (see
  `prompts/cosine-similarity.md`: the doc's `u_t = eps - A*` is the negative of
  gr00t's `v = actions - noise`, so `cos(v_n, u)` here equals the doc's
  `cos(-v_n, u_t)` and is +1 for a perfect prediction), computed over all valid
  (masked) elements pooled into a single inner product:
  `<v_n*mask, u*mask> / (||v_n*mask|| * ||u*mask|| + 1e-6)`.
- `gradnorm_vision_step_{n}` is `||grad_{h_v} L^{(n)}||_2`, the local
  sensitivity of the step-`n` masked loss to the vision embedding `h_v` (output
  of the vision tower + projector `mlp1`, in LLM embedding space, before the
  language model; all camera views concatenated). Implemented via the additive
  reparameterization `h_v + delta`, `delta = zeros_like(h_v).requires_grad_(True)`,
  `torch.autograd.grad(loss, delta)` — equal to `dL/dh_v` at `delta = 0` (see
  `prompts/vision-grad-norm.md`). All other inputs, including the step-`n`
  latent from the no-grad pass, are held fixed.

## Data pipeline (n1.5-specific)

The runner uses the official `LeRobotSingleDataset` (transforms=None) for raw
per-step data, and feeds each frame through its OWN instance of the
`PandaOmronDataConfig` transform chain (`inference/models/n15_data_config.py`).
The chain is first set to eval mode with `transform.eval()` — exactly what
`Gr00tPolicy.__init__` does for normal inference — so video preprocessing is
deterministic (center crop, no color jitter; verified: run-to-run pixel diff
0.0, while the default training-mode chain differs run to run). Then ONLY the
final `GR00TTransform` is switched back to training mode so
`action`/`action_mask` flow through (its `training` flag gates only action
inclusion and language dropout — prob 0 here — so preprocessing stays
identical to the eval path). The sample is then batched with `collate([sample],
transform.eagle_processor)`, mirroring `GR00TTransform.apply_batch`, and passed
to `GR00T_N1_5.static_inference`, which handles device/dtype via
`prepare_input`.

## Code structure (additive-only changes)

- `gr00t/model/backbone/eagle_backbone.py`
  - `EagleBackbone.extract_vision_embeddings(vl_input)` -> `h_v`
    (`[num_img_tokens, C]`), the output of `eagle_model.extract_feature`
    (vision tower + pixel shuffle + `mlp1`) reshaped as the scatter consumes it.
  - `EagleBackbone.forward_with_vision_embeds(vl_input, vision_embeds)` -> same
    `BatchFeature` as `forward()`, replicating `Eagle2_5_VL.forward`'s
    embedding-scatter + language-model call (plus `select_layer` hidden state
    and `eagle_linear`) with the provided embeddings.
- `gr00t/model/action_head/flow_matching_action_head.py`
  - `FlowmatchingActionHead.static_denoise_step(...)` -> one denoising loop
    iteration (velocity prediction), grad-capable, mirroring `get_action`
    (future_tokens included, no `encoder_attention_mask`).
- `gr00t/model/gr00t_n1.py`
  - `GR00T_N1_5.static_inference(inputs, num_inference_steps=None,
    compute_gradnorm=True)` -> per-step dict `{u, v, final_loss, cosine,
    gradnorm_vision, action_mask}` (CPU tensors / Python floats).
  - `GR00T_N1_5._static_masked_mse(...)` -> masked MSE with exact n1.5
    training-loss semantics.

None of the existing methods are modified or call the new ones; training and
regular inference paths are untouched.
