import argparse
import inspect
from typing import Optional

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from open_pi0.data.calvin import CalvinH5Dataset
from open_pi0.models.pi0_gemma import (
    Pi0GemmaProcessor,
    Pi0GemmaForConditionalGeneration,
    Pi0GemmaForConditionalGenerationOutputWithPast,
)


def retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int | None = None,
    device: Optional[str | torch.device] = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    preprocessor: Pi0GemmaProcessor = Pi0GemmaProcessor.from_pretrained(
        "weights/pi0gemma-3b-mix-224-initial",
    )

    train_dataset = CalvinH5Dataset(
        root_path="data/CALVIN/task_ABCD_D/training.h5",
        obs_seq_len=1,
        action_seq_len=64,
        use_relative_actions=True,
        image_size=224,
        repeat=1,
        training=False,
        preprocessor=preprocessor,
    )

    samples = [train_dataset[x] for x in range(2)]
    uncond_sample = [train_dataset.get_uncond_sample(x) for x in range(2)]

    samples = uncond_sample + samples

    batch = preprocessor.collate(samples)

    noise_scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "weights/pi0gemma-3b-mix-224-initial"
    )

    model: Pi0GemmaForConditionalGeneration = Pi0GemmaForConditionalGeneration.from_pretrained(args.ckpt)
    model.to(device=device)

    weight_dtype = next(model.parameters()).dtype

    input_ids: torch.Tensor = batch["input_ids"]
    attention_mask: torch.Tensor = batch["attention_mask"]
    pixel_values: torch.Tensor = batch["pixel_values"]
    propri_states: torch.Tensor = batch["propri_states"]
    actions: torch.Tensor = batch["actions"]

    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)
    pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
    propri_states = propri_states.to(device=device, dtype=weight_dtype)
    actions = actions.to(device=device)
    actions = actions[:actions.shape[0] // 2]

    num_inference_steps = 50

    timesteps, num_inference_steps = retrieve_timesteps(noise_scheduler, num_inference_steps, device, None)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * noise_scheduler.order, 0)

    pred_actions = torch.randn_like(actions)

    with torch.inference_mode():
        with tqdm(range(num_inference_steps), desc="Inference steps") as progress_bar:
            for i, t in enumerate(timesteps):
                t: torch.Tensor
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(actions.shape[0] * 2)

                outputs: Pi0GemmaForConditionalGenerationOutputWithPast = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    propri_states=propri_states,
                    timesteps=timestep,
                    noisy_actions=torch.cat([pred_actions, pred_actions], dim=0),
                    return_dict=True,
                )
                model_pred = outputs.model_pred

                model_pred_uncond, model_pred = model_pred.chunk(2, dim=0)
                model_pred = model_pred_uncond + 3.0 * (model_pred - model_pred_uncond)

                pred_actions = noise_scheduler.step(model_pred, t, pred_actions, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % noise_scheduler.order == 0):
                    progress_bar.update()

    diff = (pred_actions - actions).abs()
    print(diff.mean(), diff.std())
    print(diff.max(), diff.min())


if __name__ == "__main__":
    main()
