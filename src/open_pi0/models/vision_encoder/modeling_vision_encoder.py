import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel
from transformers.utils import logging

from .configuration_vision_encoder import VisionEncoderConfig

logger = logging.get_logger(__name__)


class MultiModalProjector(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_tower.hidden_size, config.projection_dim, bias=True
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(image_features)

        return hidden_states


class VisionEncoderPreTrainedModel(PreTrainedModel):
    config_class = VisionEncoderConfig
    base_model_prefix = "vision_tower"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True


class VisionEncoderModel(VisionEncoderPreTrainedModel):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__(config)

        self.vision_tower = AutoModel.from_config(config.vision_tower)

        self.projector = MultiModalProjector(config)

        self.scale = config.projection_dim ** -0.5

        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """

        image_features: torch.Tensor = self.vision_tower(pixel_values).last_hidden_state
        image_features = self.projector(image_features)
        return image_features * self.scale
