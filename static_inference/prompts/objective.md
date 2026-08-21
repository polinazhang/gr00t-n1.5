# Objective

Launch static inference on one atomic seen task given the task number. Datasets are in lerobot format and located in `/coc/testnvme/xzhang3205/vla-adaptation/datasets/robocasa365/atomic-seen-splits/task_{i}_demo_50` where i is 1 to 18. Only use these datasets and ignore the rest. 18 runs should be launched separately so you should write code that launches one sbatch given one task number, each takes 4 A40 gpu. For sbatch configuration example see /coc/testnvme/xzhang3205/vla-adaptation/models/gr00t-n1.5/static_inference/prompts/template.sbatch.

Your objective is to write code to support static inference runs. For context see static-inference-context.md, cosine-similarity.md, vision-grad-norm.md, robocasa.md.

For launching and the main body of code, you should write inside `/coc/testnvme/xzhang3205/vla-adaptation/models/gr00t-n1.5/static_inference`. Since the goals require some change in gr00t architecture, you're allowed to modify ``/coc/testnvme/xzhang3205/vla-adaptation/models/gr00t-n1.5` under restrictions below:

## Restrictions

You should create separate methods for static inference that must NOT interfere with any original training or inference functionality in this codebase, just like forward and get_action / get_action_with_features are the existing, untouched paths. Keep zero interference by making new methods self-contained and never called by forward, get_action, or any method they invoke. Do not modify the bodies of any of these existing methods; do not add branches, flags, or arguments to them. Only add new methods; don't delete or alter old function blocks.

Adding new functions inside an existing class, or calling shared submodules in a new sequence are fine, but you must not change their configurations, parameter requires_grad flags, or module modes (.train()/.eval()), except within the static method's own scope where the original state must be restored before returning. 

When you recreate the forward pass



# Implementation

GR00T pads action chunks and applies action_mask to the MSE. When vision grad norm is calculated, mask everything to real action dims (so that it matches the training loss). Your implementation should match the original gr00t-n1.5 loss semantics exactly. **Do not invent a new loss.**

You should not change how the gr00t code originally generate noise for inference or attempt to generate another noise. For all downstream calculations use the same noise generated in the beginning.

Vision grad norm should be computed per sample (inference). You should follow gr00t style — sum(E·mask) / mask.sum() and pool every valid element into one average. 

# Storage

Cosine similarity and vision grad norm will be saved in the same run.

The content inside meta/ are decided by the flag `--save_meta=True` passed in to the static inference script

The latents that should be saved as files are (use these as file names followed by .npy as well):
- meta/u             (this should be the prediction target: v = groud_truth_actions_from_demo − noise)
- meta/v_{diffusion_step_idx}  [this should be the velocity prediction by the model]
- final_loss_{diffusion_step_idx} [this should be the loss value of the inference computation; the same loss computed in training]
- cosine_{diffusion_step_idx}
- gradnorm_vision_step_{diffusion_step_idx}  [this should be the local sensitivity score $||\nabla_{h_v}L^{(n)}||_2$, where n is the step idx]
Note that here diffusion_step_idx refers to the number of inference steps, set 4 from gr00t default.
stored u/v should be masked to real dims.

All values are calculated per frame. When a rollout episode finishes (corresponds to one demo trajectory used), they should be stacked together and saved as npy files. The stacking mechanism should be described in a documentation so other users upon viewing can know exactly what is what.

The content inside /meta should be decided by the flag `--save_meta=True` passed in to the static inference script

Results should be written to subfolders with timestamps inside `/coc/testnvme/xzhang3205/static/gr00t`

