from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.models.siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PiZeroConfig(PretrainedConfig):
    model_type = "pi_zero"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "action_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        action_config=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )

        self.vision_config: SiglipVisionConfig

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )

        self.action_config = action_config
        if isinstance(self.action_config, dict):
            ...
        elif action_config is None:
            ...

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

        super().__init__(**kwargs)
