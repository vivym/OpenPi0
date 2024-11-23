from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class Pi0GemmaActionExpertConfig(PretrainedConfig):
    model_type = "pi0_gemma_action_expert"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        hidden_activation: str = "gelu_pytorch_tanh",
        action_horizon: int = 64,
        action_dim: int = 18,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        super().__init__(**kwargs)


class Pi0GemmaConfig(PretrainedConfig):
    model_type = "pi0_gemma"
    sub_configs = {"vision_config": AutoConfig, "action_config": Pi0GemmaActionExpertConfig}

    def __init__(
        self,
        vision_config=None,
        action_config=None,
        vocab_size=257152,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vision_config = vision_config
        self.action_config = action_config
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4304,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )

        if isinstance(self.action_config, dict):
            self.action_config = Pi0GemmaActionExpertConfig(**action_config)
        elif action_config is None:
            self.action_config = Pi0GemmaActionExpertConfig()

        self.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = hidden_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
