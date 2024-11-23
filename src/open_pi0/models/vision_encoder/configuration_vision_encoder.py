from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class VisionEncoderConfig(PretrainedConfig):
    model_type = "vision_encoder"
    sub_configs = {"vision_tower"}

    def __init__(
        self,
        vision_tower=None,
        projection_dim=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_tower = vision_tower
        self.projection_dim = projection_dim

        if isinstance(self.vision_tower, dict):
            vision_tower["model_type"] = (
                vision_tower["model_type"] if "model_type" in vision_tower else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_tower["model_type"]](**vision_tower)
        elif vision_tower is None:
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
