from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers.models.paligemma import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.models.siglip import SiglipVisionConfig

from open_pi0.models.pi0_gemma import (
    Pi0GemmaActionExpertConfig,
    Pi0GemmaConfig,
    Pi0GemmaForConditionalGeneration,
    Pi0GemmaProcessor,
)

def is_action_expert_param(name: str) -> bool:
    # TODO: This is a temporary solution to filter out the action expert parameters.
    return "action_" in name or "_action" in name or "state_proj." in name


def main():
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
    )

    noise_scheduler.save_pretrained("weights/pi0gemma-3b-mix-224-initial")

    paligemma_processor: PaliGemmaProcessor = \
        PaliGemmaProcessor.from_pretrained("google/paligemma-3b-mix-224")

    processor = Pi0GemmaProcessor(
        image_processor=paligemma_processor.image_processor,
        tokenizer=paligemma_processor.tokenizer,
    )
    processor.save_pretrained("weights/pi0gemma-3b-mix-224-initial")

    paligemma_model: PaliGemmaForConditionalGeneration = \
        PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")

    config = Pi0GemmaConfig(
        vision_config=SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=27,
            patch_size=14,
            vision_use_head=False,
        ),
        action_config=Pi0GemmaActionExpertConfig(
            hidden_size=1024,
            intermediate_size=4096,
            hidden_activation="gelu_pytorch_tanh",
            state_dim=15,
            action_horizon=64,
            action_dim=7,
        ),
        vocab_size=257216,
        hidden_size=2048,
        intermediate_size=16384,
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
    )
    model = Pi0GemmaForConditionalGeneration(config)

    model.vision_tower.load_state_dict(
        paligemma_model.vision_tower.state_dict()
    )

    model.multi_modal_projector.load_state_dict(
        paligemma_model.multi_modal_projector.state_dict()
    )

    model.lm_head.load_state_dict(
        paligemma_model.language_model.lm_head.state_dict()
    )

    missing_keys, unexpected_keys = model.model.load_state_dict(
        paligemma_model.language_model.model.state_dict(), strict=False
    )

    assert len(unexpected_keys) == 0

    for key in missing_keys:
        assert "action_" in key or "_action" in key, key

    model.save_pretrained("weights/pi0gemma-3b-mix-224-initial")

    print("Action expert parameters:")

    for name, param in model.named_parameters():
        if is_action_expert_param(name):
            print(name, param.shape)


if __name__ == "__main__":
    main()
