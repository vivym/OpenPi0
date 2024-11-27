from typing import Union, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.models.gemma import GemmaTokenizer
from transformers.models.siglip import SiglipImageProcessor
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.utils import logging

logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"
STATE_TOKEN = "<state>"
ACTION_TOKEN = "<action>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]


class Pi0GemmaTextKwargs(TextKwargs):
    suffix: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] | None


class Pi0GemmaImagesKwargs(ImagesKwargs):
    do_convert_rgb: bool | None


class Pi0GemmaActionKwargs(TypedDict, total=False):
    horizon: int | None


class Pi0GemmaProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Pi0GemmaTextKwargs
    images_kwargs: Pi0GemmaImagesKwargs
    action_kwargs: Pi0GemmaActionKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
        },
    }


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(
    prompt: str,
    bos_token: str,
    image_seq_len: int,
    image_token: str,
    num_images: int,
) -> str:
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    return f"{image_token * image_seq_len * num_images}{bos_token}{prompt}\n"


def make_batched_images(images) -> list[list[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if (
        isinstance(images, (list, tuple)) and \
        isinstance(images[0], (list, tuple)) and \
        is_valid_image(images[0][0])
    ):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")


# TODO: Add action_processor (normalize, denormalize, etc.)
# TODO: Add state_processor (normalize, denormalize, etc.)
class Pi0GemmaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor: SiglipImageProcessor | None = None,
        tokenizer: GemmaTokenizer | None = None,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        tokens_to_add = []

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add.append(image_token)

        if not hasattr(tokenizer, "state_token"):
            state_token = AddedToken(STATE_TOKEN, normalized=False, special=True)
            tokens_to_add.append(state_token)

        if not hasattr(tokenizer, "action_token"):
            action_token = AddedToken(ACTION_TOKEN, normalized=False, special=True)
            tokens_to_add.append(action_token)

        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

        if not hasattr(tokenizer, "image_token"):
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id

        if not hasattr(tokenizer, "state_token"):
            self.state_token_id = tokenizer.convert_tokens_to_ids(STATE_TOKEN)
        else:
            self.state_token_id = tokenizer.state_token_id

        if not hasattr(tokenizer, "action_token"):
            self.action_token_id = tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
        else:
            self.action_token_id = tokenizer.action_token_id

        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.image_processor: SiglipImageProcessor
        self.tokenizer: GemmaTokenizer

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] | None = None,
        **kwargs: Unpack[Pi0GemmaProcessorKwargs],
    ) -> BatchFeature:
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            Pi0GemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `Pi0GemmaProcessor` instance.")

        if text is None:
            text = ""

        if _is_str_or_image(text):
            text = [text]

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                logger.warning(
                    "You are passing both `text` and `images` to `Pi0GemmaProcessor`. The processor expects special "
                    "image tokens in the text, as many tokens as there are images per each text. It is recommended to "
                    "add `<image>` tokens in the very beginning of your text and `<bos>` token after that. For this call, "
                    "we will infer how many images each text has and add special tokens."
                )

                if isinstance(text, (list, tuple)) and isinstance(images, (list, tuple)):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. "
                            "Each prompt should be associated with an image or list of images."
                        )

                # make a nested list of lists to be able to iterate over the images and text below
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")

                if suffix is not None and _is_str_or_image(suffix):
                    suffix = [suffix]
                if suffix is not None:
                    suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    for prompt, image_list in zip(text, images)
                ]
                images = make_batched_images(images)
            else:
                text = [
                    sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    for sample in text
                ]
                input_strings = [f"{sample}\n" for sample in text]

        pixel_values = self.image_processor(
            images, **output_kwargs["images_kwargs"]
        )["pixel_values"]

        # max_length has to account for the image tokens
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs: dict[str, torch.Tensor] = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})

        return BatchFeature(data=return_data)

    def prepare_for_traning_sample(
        self,
        images: list[np.ndarray],
        instruction: str,
        propri_states: torch.Tensor,
        actions: torch.Tensor | None = None,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        pixel_values = self.image_processor(images)["pixel_values"]
        pixel_values = torch.stack(
            [torch.from_numpy(x) for x in pixel_values],
            dim=0,
        )

        prompt = build_string_from_input(
            prompt=instruction,
            bos_token=self.tokenizer.bos_token,
            image_seq_len=self.image_seq_length,
            image_token=IMAGE_TOKEN,
            num_images=len(images),
        )

        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": pixel_values,
            "propri_states": propri_states,
            "actions": actions,
        }

    def collate(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch: dict[str, list[torch.Tensor]] = {}

        for sample in samples:
            for key, value in sample.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        max_length = max(x.shape[0] for x in batch["input_ids"])
        for key, value in batch.items():
            if any(x is None for x in value):
                logger.warning(f"Skipping key: {key}")
                continue

            if key in ["input_ids", "attention_mask"]:
                if key == "input_ids":
                    pad_value = self.tokenizer.pad_token_id
                else:
                    pad_value = 0

                batch[key] = torch.stack(
                    [
                        F.pad(x, (0, max_length - x.shape[0]), mode="constant", value=pad_value)
                        for x in value
                    ],
                    dim=0,
                )
            elif key in ["pixel_values", "propri_states", "actions"]:
                batch[key] = torch.stack(value, dim=0)
            else:
                raise ValueError(f"Unknown key: {key}")

        return batch
