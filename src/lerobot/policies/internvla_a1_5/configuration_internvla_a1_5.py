import logging
from dataclasses import dataclass, field, replace
from typing import Sequence

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    ExtractVideoFramesTransformFn,
    FASTInternVLAA15ActionTokenizerTransformFn,
    InternVLAA15ChatProcessorTransformFn,
    InternVLAA15VQAProcessorTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import HF_HOME, OBS_IMAGES


@DatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenize_state: bool = False
    max_prompt_length: int = 650
    mode: str = "train"
    chunk_size: int = 50
    use_fast_action_tokens: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"
    fast_tokenizer_path: str = "physical-intelligence/fast"
    norm_mode: str = "mean_std"
    state_dropout_prob: float = 0.0
    eval_output_tag: str = "<Subtask, Action>"

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=InternVLAA15DatasetConfig.height,
                    width=InternVLAA15DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                ExtractVideoFramesTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                LoadActionTextFromJsonlTransformFn(),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15DatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15DatasetConfig.max_action_dim,
                ),
                ReorderStateActionTransform(),
                UnifyInternVLAA15InputsTransformFn(
                    num_video_frames=InternVLAA15DatasetConfig.num_video_frames,
                    video_height=InternVLAA15DatasetConfig.video_height,
                    video_width=InternVLAA15DatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta" and not has_delta:
            logging.info("action_mode='delta' -> Adding DeltaActionTransformFn")
            inputs = [DeltaActionTransformFn(), *inputs]
        elif self.action_mode == "abs" and has_delta:
            logging.info("action_mode='abs' -> Removing DeltaActionTransformFn")
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]

        inputs = [
            t
            for t in inputs
            if not isinstance(
                t, (FASTInternVLAA15ActionTokenizerTransformFn, InternVLAA15ChatProcessorTransformFn)
            )
        ]

        if self.use_fast_action_tokens:
            fast_idx = next(
                (i for i, t in enumerate(inputs) if isinstance(t, LoadActionTextFromJsonlTransformFn)),
                next(
                    (i for i, t in enumerate(inputs) if isinstance(t, PadStateAndActionTransformFn)),
                    len(inputs),
                ),
            )
            inputs.insert(
                fast_idx,
                FASTInternVLAA15ActionTokenizerTransformFn(
                    action_tokenizer_name=self.fast_tokenizer_path,
                    qwen35_model_name=self.vlm_model_name_or_path,
                    chunk_size=self.chunk_size,
                    max_action_dim=self.max_action_dim,
                ),
            )

        num_views = 4 if self.enable_head_history else 3
        processor = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=self.vlm_model_name_or_path,
            tokenize_state=self.tokenize_state,
            max_state_dim=self.max_state_dim,
            max_length=self.max_prompt_length,
            num_views=num_views,
            use_fast_action_tokens=self.use_fast_action_tokens,
            mode=self.mode,
            norm_mode=self.norm_mode,
            state_dropout_prob=self.state_dropout_prob,
            vlash_condition_on_delay=(self.vlash_enabled and self.vlash_condition_on_delay),
            eval_output_tag=self.eval_output_tag,
        )
        insert_idx = next(
            (i for i, t in enumerate(inputs) if isinstance(t, PadStateAndActionTransformFn)),
            len(inputs),
        )
        inputs.insert(insert_idx, processor)

        for idx_t, t in enumerate(inputs):
            if isinstance(t, NormalizeTransformFn):
                inputs[idx_t] = replace(t, mode=self.norm_mode)
            elif isinstance(t, ExtractVideoFramesTransformFn):
                inputs[idx_t] = replace(
                    t, video_height=self.video_height, video_width=self.video_width
                )
            elif isinstance(t, ResizeImagesWithPadFn):
                inputs[idx_t] = replace(t, height=self.height, width=self.width)
            elif isinstance(t, PadStateAndActionTransformFn):
                inputs[idx_t] = replace(
                    t, max_state_dim=self.max_state_dim, max_action_dim=self.max_action_dim
                )
            elif isinstance(t, UnifyInternVLAA15InputsTransformFn):
                inputs[idx_t] = replace(
                    t,
                    num_video_frames=self.num_video_frames,
                    video_height=self.video_height,
                    video_width=self.video_width,
                    chunk_size=self.chunk_size,
                    max_action_dim=self.max_action_dim,
                )

        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@DataTransformFn.register_subclass("unify_internvla_a1_5_inputs")
@dataclass
class UnifyInternVLAA15InputsTransformFn(DataTransformFn):
    """Unify robot samples and always include video_frames for WAN.

    Always outputs observation.video_frames so that robot and VQA samples
    have identical keys and can be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    chunk_size: int = 50
    max_action_dim: int = 32

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_NONE
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_NONE, dtype=torch.long))

        video_key = "observation.video_frames"
        if video_key in data:
            video_frames = data[video_key]
        else:
            video_frames = torch.zeros(
                self.num_video_frames + 1, 3, self.video_height, self.video_width
            )

        action_tensor = data[ACTION]
        action_is_pad = data.get(
            "action_is_pad",
            torch.zeros(action_tensor.shape[0] if action_tensor.ndim >= 2 else self.chunk_size, dtype=torch.bool),
        )
        action_dim_mask = data.get(
            "action_dim_mask",
            torch.ones(action_tensor.shape[-1], dtype=torch.bool),
        )

        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: action_tensor,
            "action_is_pad": action_is_pad.to(dtype=torch.bool),
            "action_dim_mask": action_dim_mask.to(dtype=torch.bool),
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": data["vqa_type"],
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            video_key: video_frames,
            "vlash_delay_steps": data.get("vlash_delay_steps", torch.tensor(0, dtype=torch.long)),
        }


@DataTransformFn.register_subclass("unify_internvla_a1_5_vqa_inputs")
@dataclass
class UnifyInternVLAA15VQAInputsTransformFn(DataTransformFn):
    """VQA unify transform that includes a dummy video_frames tensor.

    Ensures VQA samples have the same keys as robot samples so they can
    be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    chunk_size: int = 50
    max_action_dim: int = 32

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_TEXT
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_TEXT, dtype=torch.long))

        video_frames = torch.zeros(
            self.num_video_frames + 1, 3, self.video_height, self.video_width
        )
        action_tensor = data[ACTION]
        action_is_pad = torch.zeros(
            action_tensor.shape[0] if action_tensor.ndim >= 2 else self.chunk_size,
            dtype=torch.bool,
        )
        action_dim_mask = torch.ones(action_tensor.shape[-1], dtype=torch.bool)

        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: action_tensor,
            "action_is_pad": action_is_pad,
            "action_dim_mask": action_dim_mask,
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": torch.tensor(1, dtype=torch.long),
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            "observation.video_frames": video_frames,
            "vlash_delay_steps": torch.tensor(0, dtype=torch.long),
        }


@VQADatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15VQADatasetConfig(VQADatasetConfig):
    """VQA dataset config with dummy video_frames for shared batching."""

    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # Note: If you resize the VQA images, the coordinates in the VQA Data should be scaled accordingly. 
                # Please pre-process the VQA data offline based on its format.
                ResizeVQAImagesWithPadFn(
                    height=InternVLAA15VQADatasetConfig.height,
                    width=InternVLAA15VQADatasetConfig.width,
                ),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15VQADatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15VQADatasetConfig.max_action_dim,
                ),
                InternVLAA15VQAProcessorTransformFn(),
                UnifyInternVLAA15VQAInputsTransformFn(
                    num_video_frames=InternVLAA15VQADatasetConfig.num_video_frames,
                    video_height=InternVLAA15VQADatasetConfig.video_height,
                    video_width=InternVLAA15VQADatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        inputs = list(self.data_transforms.inputs)
        processor_type = InternVLAA15VQAProcessorTransformFn
        inputs = [t for t in inputs if not isinstance(t, processor_type)]
        insert_idx = next(
            (
                i
                for i, t in enumerate(inputs)
                if isinstance(t, UnifyInternVLAA15VQAInputsTransformFn)
            ),
            len(inputs),
        )
        inputs.insert(insert_idx, InternVLAA15VQAProcessorTransformFn())
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15Config(PreTrainedConfig):
    # VLM model selection - supports Qwen3.5-2B/4B/8B
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"

    # Action expert customization
    action_expert_hidden_size: int | None = 1024
    action_expert_intermediate_size: int | None = 3072

    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # VQA configurations
    enable_vqa_loss: bool = True
    lambda_vqa: float = 1.0
    tokenize_state: bool = True

    # FAST action tokens
    action_token_min: int = 248077
    action_token_max: int = 250124

    # Action loss masking (VLASH / episode-tail padding & padded action dims)
    mask_action_pad: bool = False
    padded_action_dim_weight: float = 1.0

    # Knowledge insulation
    knowledge_insulation: bool = False
    block_action_attend_fast_tokens: bool = True

    inference_action_type: str = "fm"  # "fm" for flow matching, "fast" for fast-token supervision
    inference_backend: str = "standard"  # "standard" or "optimized"

    # Use SDPA (scaled_dot_product_attention) instead of eager attention in training
    use_sdpa: bool = False

    num_learnable_tokens: int = 50

    wan_checkpoint_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    wan_config_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    vae_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    video_precision: str = "bfloat16"

    freeze_wan_dit: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    video_loss_weight: float = 1.0
    video_loss_only: bool = False
    action_loss_only: bool = False
    freeze_learnable_tokens: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.lambda_vqa < 0:
            raise ValueError(f"lambda_vqa must be >= 0, got {self.lambda_vqa}")
        if self.action_token_min > self.action_token_max:
            raise ValueError(
                f"action_token_min ({self.action_token_min}) must be <= action_token_max ({self.action_token_max})"
            )
        if self.inference_backend not in {"standard", "optimized"}:
            raise ValueError(
                "inference_backend must be either 'standard' or 'optimized', "
                f"got {self.inference_backend!r}"
            )
        if self.inference_backend == "optimized" and not self.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None:
        n = self.num_video_frames + 1
        return [self.chunk_size * i // (n - 1) for i in range(n)]
