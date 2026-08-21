# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def static_inference(
        self,
        inputs: dict,
        num_inference_steps: int | None = None,
        compute_gradnorm: bool = True,
    ) -> dict:
        """
        Static inference (analysis-only; additive method, not used by forward/get_action).

        Given one demonstration frame (batch size 1, inputs must contain the padded
        ground-truth `action` and `action_mask`, i.e. processed with the action
        modality included), runs the standard 4-step flow-matching denoising loop
        starting from a single noise tensor drawn exactly like get_action, and
        compares the per-step velocity predictions against the ground-truth
        target u = gt_action - noise.

        Returns a dict of CPU tensors / floats:
            - "u": [B, H, D] prediction target (gt_action - noise), float32 on CPU
            - "v": list of num_steps tensors [B, H, D], velocity prediction per step
            - "final_loss": list of num_steps floats, masked MSE(v_n, u) per step,
              same semantics as the n1.5 training loss
              (sum((v_n - u)^2 * mask) / mask.sum())
            - "cosine": list of num_steps floats, cosine between v_n and u over the
              valid (masked) elements only
            - "gradnorm_vision": list of num_steps floats, ||grad_{h_v} L_n||_2 where
              h_v is the vision embedding (image encoder + projector output, before
              the language model); NaN if compute_gradnorm=False
            - "action_mask": [B, H, D] mask on CPU
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        assert "action" in action_inputs, "static_inference requires ground-truth actions"
        assert "action_mask" in action_inputs, "static_inference requires action_mask"
        gt_action = action_inputs.action  # [B, max_action_horizon, max_action_dim]
        action_mask = action_inputs.action_mask
        embodiment_id = action_inputs.embodiment_id

        head = self.action_head
        num_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else head.num_inference_timesteps
        )
        dt = 1.0 / num_steps

        # ---- Pass 1: standard denoising loop under no_grad ----
        with torch.no_grad():
            backbone_outputs = self.backbone(backbone_inputs)
            # process_backbone_output mutates the dict in place; this dict is fresh
            # from the backbone call above, so this is safe.
            backbone_outputs = head.process_backbone_output(backbone_outputs)
            vl_embs = backbone_outputs.backbone_features
            state_features = head.state_encoder(action_inputs.state, embodiment_id)

            # Generate noise ONCE, exactly like get_action, and reuse the same
            # tensor as the t=0 latent for all steps and computations.
            batch_size = vl_embs.shape[0]
            noise = torch.randn(
                size=(batch_size, head.config.action_horizon, head.config.action_dim),
                dtype=vl_embs.dtype,
                device=vl_embs.device,
            )

            u = gt_action - noise
            latents = noise.clone()

            v_list = []
            latent_list = []
            final_loss_list = []
            cosine_list = []
            for n in range(num_steps):
                v_n = head.static_denoise_step(
                    backbone_outputs, state_features, latents, n, embodiment_id
                )
                latent_list.append(latents.clone())
                final_loss_list.append(float(self._static_masked_mse(v_n, u, action_mask)))
                # Cosine between v_n and u over valid (masked) elements only,
                # pooling all valid elements into one inner product. gr00t
                # predicts v = A* - eps and our stored u = A* - eps, so
                # cos(v_n, u) is +1 for a perfect prediction.
                v_masked = v_n * action_mask
                u_masked = u * action_mask
                cosine_n = (v_masked * u_masked).sum() / (
                    v_masked.norm() * u_masked.norm() + 1e-6
                )
                v_list.append(v_n.detach())
                cosine_list.append(float(cosine_n))
                latents = latents + dt * v_n

        # ---- Pass 2: vision grad norm per step (grad through h_v) ----
        if compute_gradnorm:
            # h_v computed under no_grad and detached; the gradient target is the
            # additive delta, which equals dL/dh_v at delta=0.
            with torch.no_grad():
                h_v = self.backbone.extract_vision_embeddings(backbone_inputs)
            h_v = h_v.detach()

            gradnorm_list = []
            for n in range(num_steps):
                delta = torch.zeros_like(h_v).requires_grad_(True)
                backbone_outputs_n = self.backbone.forward_with_vision_embeds(
                    backbone_inputs, h_v + delta
                )
                backbone_outputs_n = head.process_backbone_output(backbone_outputs_n)
                state_features_n = head.state_encoder(action_inputs.state, embodiment_id)
                v_n = head.static_denoise_step(
                    backbone_outputs_n, state_features_n, latent_list[n], n, embodiment_id
                )
                loss_n = self._static_masked_mse(v_n, u, action_mask)
                g = torch.autograd.grad(loss_n, delta)[0]
                gradnorm_list.append(float(g.float().norm(p=2)))
        else:
            gradnorm_list = [float("nan")] * num_steps

        return {
            "u": u.detach().float().cpu(),
            "v": [v.float().cpu() for v in v_list],
            "final_loss": final_loss_list,
            "cosine": cosine_list,
            "gradnorm_vision": gradnorm_list,
            "action_mask": action_mask.detach().cpu(),
        }

    @staticmethod
    def _static_masked_mse(v: torch.Tensor, u: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Masked MSE with the exact n1.5 training-loss semantics:
        sum((v - u)^2 * mask) / mask.sum() (FlowmatchingActionHead.forward).
        The +1e-6 epsilon used in the n1.6 static-inference code is dropped to
        match n1.5 exactly; a zero mask is asserted against instead (an empty
        mask would indicate a data/config bug, and NaN would silently poison
        every downstream statistic)."""
        mask_sum = action_mask.sum()
        assert mask_sum > 0, "action_mask has no valid elements"
        return ((v - u) ** 2 * action_mask).sum() / mask_sum

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
