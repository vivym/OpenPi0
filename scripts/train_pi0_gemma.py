import argparse
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from open_pi0.data.calvin import CalvinH5Dataset
from open_pi0.models.pi0_gemma import (
    Pi0GemmaProcessor,
    Pi0GemmaForConditionalGeneration,
    Pi0GemmaForConditionalGenerationOutputWithPast,
)
from open_pi0.optimization import get_scheduler
from open_pi0.utils.ema import EMAModel

logger = get_logger(__name__)


def is_action_expert_param(name: str) -> bool:
    # TODO: This is a temporary solution to filter out the action expert parameters.
    return "action_" in name or "_action" in name or "state_proj." in name


@lru_cache()
def get_beta_distribution(batch_size: int, alpha: float, beta: float, device: str = "cpu"):
    """
    Get a beta distribution with the given alpha and beta parameters.
    """
    alpha_tensor = torch.full((batch_size,), alpha, device=device)
    beta_tensor = torch.full((batch_size,), beta, device=device)
    return torch.distributions.beta.Beta(alpha_tensor, beta_tensor)


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    beta_alpha: float = None,
    beta_beta: float = None,
    beta_cutoff: float = None,
):
    """
    Compute the density for sampling the timesteps when doing flow matching training.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "beta":
        beta_dist = get_beta_distribution(batch_size, beta_alpha, beta_beta)
        u = beta_dist.sample()
        u = beta_cutoff * u
        u = torch.clamp(u, 0, 1)
    elif weighting_scheme == "neg_beta":
        beta_dist = get_beta_distribution(batch_size, beta_alpha, beta_beta)
        u = beta_dist.sample()
        u = beta_cutoff * (1 - u)
        u = torch.clamp(u, 0, 1)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_flow_matching(
    weighting_scheme: str,
    sigmas=None,
):
    """
    Computes loss weighting scheme for flow matching training.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


@dataclass
class Arguments:
    # The name of the dataset to use (via the datasets library).
    dataset_name: str = "CALVIN"

    # Path to pretrained model or model identifier from huggingface.co/models.
    pretrained_model_name_or_path: str = "weights/pi0gemma-3b-mix-224-initial"

    # Revision of pretrained model identifier from huggingface.co/models.
    revision: str = None

    # Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16
    variant: str = None

    # Batch size (per device) for the training dataloader.
    per_device_train_batch_size: int = 8

    # Batch size (per device) for the evaluation dataloader.
    per_device_eval_batch_size: int = 8

    # Initial learning rate (after the potential warmup period) to use.
    learning_rate: float = 1e-4

    # Initial learning rate for the action-expert module.
    learning_rate_action: float | None = None

    # The optimizer type to use. Choose between ["AdamW", "prodigy"]
    optimizer: str = "AdamW"

    # Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW
    use_8bit_adam: bool = False

    # The beta1 parameter for the Adam and Prodigy optimizers.
    adam_beta1: float = 0.9

    # The beta2 parameter for the Adam and Prodigy optimizers.
    adam_beta2: float = 0.999

    # Coefficients for computing the Prodigy stepsize using running averages. If set to None,
    # uses the value of square root of beta2. Ignored if optimizer is adamW
    prodigy_beta3: float = None

    # Use AdamW style decoupled weight decay
    prodigy_decouple: bool = True

    # Epsilon value for the Adam optimizer and Prodigy optimizers.
    adam_epsilon: float = 1e-08

    # Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW
    prodigy_use_bias_correction: bool = True

    # Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default.
    # Ignored if optimizer is adamW
    prodigy_safeguard_warmup: bool = True

    # Max gradient norm.
    max_grad_norm: float = 1.0

    # Weight decay to use.
    weight_decay: float = 1e-4

    # Total number of training epochs to perform.
    num_train_epochs: int | None = 100

    # Total number of training steps to perform. If provided, overrides num_train_epochs.
    max_train_steps: int | None = None

    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1

    # The scheduler type to use. Choose between
    # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"

    # Number of steps for the warmup in the lr scheduler.
    lr_warmup_steps: int = 0

    # Number of warmup steps for the action-expert module.
    lr_warmup_steps_action: int = 0

    # Number of hard resets of the lr in cosine_with_restarts scheduler.
    lr_num_cycles: int = 1

    # Power factor of the polynomial scheduler.
    lr_power: float = 1.0

    # Where to store the final model.
    output_dir: str = "outputs/pi0-gemma-3b"

    # A seed for reproducible training.
    seed: int | None = None

    # Optional input sequence length after tokenization. The training dataset will be truncated in block of
    # this size for training. Default to the model max input length for single sentence inputs (take into
    # account special tokens).
    block_size: int | None = None

    # The number of processes to use for the preprocessing.
    preprocessing_num_workers: int | None = None

    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    dataloader_num_workers: int = 8

    # Overwrite the cached training and evaluation sets.
    overwrite_cache: bool = False

    # Whether or not to push the model to the Hub.
    push_to_hub: bool = False

    # The name of the repository to keep in sync with the local `output_dir`.
    hub_model_id: str | None = None

    # The token to use to push to the Model Hub.
    hub_token: str | None = None

    # Whether to trust the execution of code from datasets/models defined on the Hub.
    # This option should only be set to `True` for repositories you trust and in which you have read the
    # code, as it will execute code present on the Hub on your local machine.
    trust_remote_code: bool = False

    # Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.
    checkpointing_steps: str | None = None

    # Max number of checkpoints to preserve.
    checkpoints_total_limit: int | None = None

    # If the training should continue from a checkpoint folder.
    resume_from_checkpoint: str | None = None

    # Number of steps to wait before logging training metrics.
    logging_steps: int = 50

    # Whether to enable experiment trackers for logging.
    with_tracking: bool = False

    # The integration to report the results and logs to. Supported platforms are `"tensorboard"`,
    # `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.
    # Only applicable when `--with_tracking` is passed.
    report_to: str = "all"

    # It is an option to create the model as an empty shell, then only materialize its parameters
    # when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited.
    low_cpu_mem_usage: bool = False

    # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    gradient_checkpointing: bool = False

    # We default to the "none" weighting scheme for uniform sampling and uniform loss
    # Choose between ["sigma_sqrt", "logit_normal", "mode", "cosmap", "neg_beta", "none"]
    weighting_scheme: str = "none"

    # mean to use when using the `'logit_normal'` weighting scheme.
    logit_mean: float = 0.0

    # std to use when using the `'logit_normal'` weighting scheme.
    logit_std: float = 1.0

    # Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.
    mode_scale: float = 1.29

    # alpha parameter for the beta distribution. Only effective when using the `'neg_beta'` as the `weighting_scheme`.
    beta_alpha: float = 1.5

    # beta parameter for the beta distribution. Only effective when using the `'neg_beta'` as the `weighting_scheme`.
    beta_beta: float = 1.0

    # Cutoff value for the beta distribution. Only effective when using the `'neg_beta'` as the `weighting_scheme`.
    beta_cutoff: float = 0.999

    # [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
    # *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
    logging_dir: str = "logs"

    # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = False

    # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=
    # 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the
    # flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.
    mixed_precision: str | None = None

    # Whether or not to enable exponential moving average (EMA) for the model.
    use_ema: bool = False

    # Use faster foreach implementation of EMAModel.
    foreach_ema: bool = False

    # EMA decay rate.
    ema_decay: float = 0.9999

    # The number of steps to wait before starting to update the EMA weights.
    ema_update_after_steps: int = 0

    # Whether to warmup the EMA model.
    ema_warmup: bool = False

    # Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `ema_warmup` is True.
    ema_inv_gamma: float = 1.0

    # Exponential factor of EMA warmup. Default: 2/3. Only used if `ema_warmup` is True.
    ema_power: float = 2 / 3


def parse_args(input_args=None) -> Arguments:
    parser = argparse.ArgumentParser(description="Training script for pi0")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CALVIN",
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="weights/pi0gemma-3b-mix-224-initial",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--learning_rate_action",
        type=float,
        default=None,
        help="Initial learning rate for the action-expert module.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )

    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )

    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )

    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )

    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay to use.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--lr_warmup_steps_action",
        type=int,
        default=0,
        help="Number of warmup steps for the action-expert module.",
    )

    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )

    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pi0-gemma-3b",
        help="Where to store the final model.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets.",
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )

    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to preserve.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of steps to wait before logging training metrics.",
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters"
            " when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "beta", "neg_beta", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )

    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )

    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )

    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    parser.add_argument(
        "--beta_alpha",
        type=float,
        default=1.5,
        help="alpha parameter for the beta distribution. Only effective when using the `'neg_beta'` as the `weighting_scheme`.",
    )

    parser.add_argument(
        "--beta_beta",
        type=float,
        default=1.0,
        help="beta parameter for the beta distribution. Only effective when using the `'neg_beta'` as the `weighting_scheme`.",
    )

    parser.add_argument(
        "--beta_cutoff",
        type=float,
        default=0.999,
        help="Cutoff value for the beta distribution. Only effective when using the `'neg_beta'` as the `weighting_scheme`.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether or not to enable exponential moving average (EMA) for the model.",
    )

    parser.add_argument(
        "--foreach_ema",
        action="store_true",
        help="Use faster foreach implementation of EMAModel.",
    )

    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate.",
    )

    parser.add_argument(
        "--ema_update_after_steps",
        type=int,
        default=0,
        help="The number of steps to wait before starting to update the EMA weights.",
    )

    parser.add_argument(
        "--ema_warmup",
        action="store_true",
        help="Whether to warmup the EMA model.",
    )

    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `ema_warmup` is True.",
    )

    parser.add_argument(
        "--ema_power",
        type=float,
        default=2 / 3,
        help="Exponential factor of EMA warmup. Default: 2/3. Only used if `ema_warmup` is True.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args: Arguments):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        model_id = args.hub_model_id or Path(args.output_dir).name
        repo_id = None
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=model_id,
                exist_ok=True,
            ).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

    accelerator.wait_for_everyone()

    preprocessor: Pi0GemmaProcessor = Pi0GemmaProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )

    if args.dataset_name == "CALVIN":
        train_dataset = CalvinH5Dataset(
            root_path="data/CALVIN/task_ABCD_D/training.h5",
            obs_seq_len=1,
            action_seq_len=64,
            use_relative_actions=True,
            image_size=224,
            repeat=1,
            training=True,
            preprocessor=preprocessor,
        )

        eval_dataset = CalvinH5Dataset(
            root_path="data/CALVIN/task_ABCD_D/validation.h5",
            obs_seq_len=1,
            action_seq_len=64,
            use_relative_actions=True,
            image_size=224,
            repeat=1,
            training=False,
            preprocessor=preprocessor,
        )
    else:
        raise ValueError(f"Dataset {args.dataset_name} not found.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=preprocessor.collate,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=preprocessor.collate,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        pin_memory=True,
    )

    noise_scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )

    model: Pi0GemmaForConditionalGeneration = Pi0GemmaForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )

    model.requires_grad_(True)

    if args.use_ema:
        model_copy: Pi0GemmaForConditionalGeneration = Pi0GemmaForConditionalGeneration.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
        )
        ema_model = EMAModel(
            model_copy.parameters(),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_steps,
            use_ema_warmup=args.ema_warmup,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            foreach=args.foreach_ema,
            model_cls=Pi0GemmaForConditionalGeneration,
            model_config=model.config,
        )

    # TODO: save/load state hooks, handle ema loading logic

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.learning_rate_action is None:
        args.learning_rate_action = args.learning_rate

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not is_action_expert_param(n)
            ],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and not is_action_expert_param(n)
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and is_action_expert_param(n)
            ],
            "lr": args.learning_rate_action,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and is_action_expert_param(n)
            ],
            "lr": args.learning_rate_action,
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            optimizer_grouped_parameters,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            optimizer_grouped_parameters,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler: LRScheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=[
            args.lr_warmup_steps * accelerator.num_processes,
            args.lr_warmup_steps * accelerator.num_processes,
            args.lr_warmup_steps_action * accelerator.num_processes,
            args.lr_warmup_steps_action * accelerator.num_processes,
        ],
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, lr_scheduler
    )

    train_dataloader: DataLoader
    eval_dataloader: DataLoader
    model: Pi0GemmaForConditionalGeneration
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler

    if args.use_ema:
        ema_model.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("pi0_gemma", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    def unwrap_model(model) -> Pi0GemmaForConditionalGeneration:
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def get_sigmas(timesteps: torch.Tensor, n_dim=4, dtype=torch.float32) -> torch.Tensor:
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    num_train_timesteps: int = noise_scheduler.config.num_train_timesteps

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        if args.with_tracking:
            total_loss = 0.0
            loss_denominator = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for batch in active_dataloader:
            with accelerator.accumulate(model):
                input_ids: torch.Tensor = batch["input_ids"]
                attention_mask: torch.Tensor = batch["attention_mask"]
                pixel_values: torch.Tensor = batch["pixel_values"]
                propri_states: torch.Tensor = batch["propri_states"]
                actions: torch.Tensor = batch["actions"]

                pixel_values = pixel_values.to(dtype=weight_dtype)
                propri_states = propri_states.to(dtype=weight_dtype)

                # Sample noise that we'll add to the actions
                noise = torch.randn_like(actions)

                # Sample a random timestep for each sample in the batch
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=noise.shape[0],
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    beta_alpha=args.beta_alpha,
                    beta_beta=args.beta_beta,
                    beta_cutoff=args.beta_cutoff,
                )
                indices = (u * num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=actions.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=actions.ndim, dtype=actions.dtype)
                noisy_actions = (1.0 - sigmas) * actions + sigmas * noise
                noisy_actions = noisy_actions.to(dtype=weight_dtype)

                targets = noise - actions
                loss_weights = compute_loss_weighting_for_flow_matching(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                outputs: Pi0GemmaForConditionalGenerationOutputWithPast = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    propri_states=propri_states,
                    timesteps=timesteps,
                    noisy_actions=noisy_actions,
                    targets=targets,
                    loss_weights=loss_weights,
                    return_dict=True,
                )
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    loss_denominator += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = [p for p in model.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())

                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps:08d}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    if args.use_ema and accelerator.is_main_process:
                        ema_model.save_pretrained(os.path.join(output_dir, "ema"))
                    logger.info(f"Saved state to {output_dir}")

            if (
                args.with_tracking and
                completed_steps % args.logging_steps == 0 and
                accelerator.sync_gradients and
                torch.is_tensor(total_loss)
            ):
                total_losses = accelerator.gather(total_loss.repeat(args.per_device_train_batch_size))
                total_loss = torch.mean(total_losses) / loss_denominator

                accelerator.log(
                    {
                        "train_loss": total_loss,
                        "lr_lm": lr_scheduler.get_last_lr()[0],
                        "lr_action": lr_scheduler.get_last_lr()[2],
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

                total_loss = 0.0
                loss_denominator = 0

            if completed_steps >= args.max_train_steps:
                break

        if args.use_ema:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

        model.eval()
        losses = []
        for batch in eval_dataloader:
            with torch.no_grad():
                # TODO: refactor this to a function
                input_ids: torch.Tensor = batch["input_ids"]
                attention_mask: torch.Tensor = batch["attention_mask"]
                pixel_values: torch.Tensor = batch["pixel_values"]
                propri_states: torch.Tensor = batch["propri_states"]
                actions: torch.Tensor = batch["actions"]

                pixel_values = pixel_values.to(dtype=weight_dtype)
                propri_states = propri_states.to(dtype=weight_dtype)

                # Sample noise that we'll add to the actions
                noise = torch.randn_like(actions)

                # Sample a random timestep for each sample in the batch
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=noise.shape[0],
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    beta_alpha=args.beta_alpha,
                    beta_beta=args.beta_beta,
                    beta_cutoff=args.beta_cutoff,
                )
                indices = (u * num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=actions.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=actions.ndim, dtype=actions.dtype)
                noisy_actions = (1.0 - sigmas) * actions + sigmas * noise
                noisy_actions = noisy_actions.to(dtype=weight_dtype)

                targets = noise - actions
                loss_weights = compute_loss_weighting_for_flow_matching(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                outputs: Pi0GemmaForConditionalGenerationOutputWithPast = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    propri_states=propri_states,
                    timesteps=timesteps,
                    noisy_actions=noisy_actions,
                    targets=targets,
                    loss_weights=loss_weights,
                    return_dict=True,
                )

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        logger.info(f"epoch {epoch}, eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log({"eval_loss": eval_loss}, step=completed_steps)

        if args.use_ema:
            ema_model.restore(model.parameters())

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = unwrap_model(model)

            if args.use_ema:
                ema_model.store(unwrapped_model.parameters())
                ema_model.copy_to(unwrapped_model.parameters())

            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                preprocessor.save_pretrained(args.output_dir)
                noise_scheduler.save_pretrained(args.output_dir)
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message=f"Training in progress epoch {epoch}",
                    repo_type="model",
                )

            if args.use_ema:
                ema_model.restore(unwrapped_model.parameters())

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            if args.use_ema and accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema"))
            logger.info(f"Saved state to {output_dir}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = unwrap_model(model)

        if args.use_ema:
            ema_model.copy_to(unwrapped_model.parameters())

        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            noise_scheduler.save_pretrained(args.output_dir)
            preprocessor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message="End of training",
                    repo_type="model",
                )

    accelerator.wait_for_everyone()

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
