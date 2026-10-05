"""EarthDial-4B predictor: the zero-shot RS baseline for the eval table.

Why this model is in the repo at all: the load-bearing bet of the interior is
that a QLoRA-tuned Qwen2.5-VL beats the purpose-built RS VLMs, all of which sit
on 2023-24 bases. That bet needs a referee, and EarthDial-4B (CVPR'25, MIT code,
multi-sensor RGB/MS/SAR/temporal) is it -- which is the entire reason the eval
harness exists before the training run.

Scope, per the decision locked 2026-08-26: this predictor is a *benchmark*
participant, not a served model. Only the tuned Qwen answers users, so nothing
here needs a FastAPI endpoint, a warm cache or an ensemble path -- it needs to
produce a defensible row in the before/after table and then release its VRAM.
:meth:`close` is therefore load-bearing: the harness runs both predictors over
the same benchmark inside one T4 session.

The API differs from Qwen's in a way that matters. EarthDial is an InternVL
derivative (Phi-3-mini language tower), so it is loaded with ``AutoModel`` +
``trust_remote_code=True`` and queried through the repo's own ``model.chat``
method with pre-normalised ``pixel_values`` rather than a processor and a chat
template. Two consequences are baked in below: ``use_flash_attn=False`` must be
passed to the remote code (InternVL enables FlashAttention-2 by default, which
does not exist on a T4), and ``model.chat`` returns a bare string, so
``confidence`` is always ``None`` -- the evidence block says why rather than
substituting a number nobody measured.

# VERIFY: the repo id, the ``chat`` signature and the 448 px tile convention are
# recalled from the EarthDial release notes and InternVL's published usage, not
# from a real download on this machine. The first Colab run against
# ``akshaydudhane/EarthDial_4B_RGB`` confirms or corrects them; the failure mode
# is a clear SatQueryError from :meth:`EarthDialPredictor._generate`, not a
# wrong number.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import SatQueryError, Task
from ..log import get_logger
from .qwen_vl import (
    Generation,
    VLPredictorBase,
    _require_module,
    _VLM_HINT,
)

__all__ = [
    "PROMPTS",
    "SYSTEM_PROMPT",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_SIZE",
    "EarthDialPredictor",
    "build_question",
    "build_pixel_values",
]

LOG = get_logger(__name__)

#: InternVL's ViT normalisation. Wrong constants here do not crash, they just
#: quietly cost accuracy, which is the worst kind of bug in a baseline number.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: InternVL tiles are 448 px. One tile per image, deliberately: dynamic tiling
#: would raise effective resolution but multiplies vision tokens per image, and
#: with two images on a 4 GB budget the pair tasks stop fitting. RS tiles are
#: already capped at 768 px upstream, so the loss is modest and constant.
INPUT_SIZE = 448

SYSTEM_PROMPT = None  # EarthDial's remote code injects its own system prompt.

#: Prompts are separate from Qwen's on purpose: the models were instruction-tuned
#: differently, so a prompt tuned for one is not evidence about the other, and a
#: shared table would silently couple two independent tuning surfaces.
#: EarthDial follows the GeoChat-style convention of short, flat instructions and
#: emits boxes as ``<box>[[x0, y0, x1, y1]]</box>`` normalised to 0-1000, which is
#: what :func:`satquery.models.qwen_vl.parse_boxes` reads with ``scale=1000``.
PROMPTS: dict[str, str] = {
    "vqa": (
        "Answer the question about this satellite image using a single word or "
        "number.\n{question}"
    ),
    "caption": (
        "Describe this satellite image in detail, covering land cover and "
        "man-made structures."
    ),
    "ground": (
        "Please provide the bounding box coordinates of the region this "
        "sentence describes: {question}"
    ),
    "change_vqa": (
        "These are two satellite images of the same place, the first earlier and "
        "the second later. Answer the question about the change between them "
        "using a single word or number.\n{question}"
    ),
    "change_caption": (
        "These are two satellite images of the same place, the first earlier and "
        "the second later. Describe what changed between them."
    ),
}


def build_question(prompt: str, n_images: int) -> str:
    """Prefix the InternVL image placeholders onto the instruction.

    InternVL counts ``<image>`` tokens against ``num_patches_list``, so the
    number of placeholders is part of the contract, not decoration. The
    ``Image-1:`` / ``Image-2:`` labels are InternVL's documented multi-image
    format and are what lets a change question refer to "the second image" at
    all.
    """
    if n_images < 1:
        raise ValueError("at least one image is required")
    if n_images == 1:
        return f"<image>\n{prompt}"
    header = "".join(f"Image-{i + 1}: <image>\n" for i in range(n_images))
    return f"{header}{prompt}"


def _to_tile(image: Any, input_size: int = INPUT_SIZE) -> np.ndarray:
    """One PIL image -> ``(3, s, s)`` float32, ImageNet-normalised."""
    pil = _require_module(
        "PIL.Image", who="EarthDial image preparation", hint='pip install "satquery[data]"'
    )
    # Pillow moved the resampling constants under Image.Resampling in 9.1.
    resampling = getattr(pil, "Resampling", pil)
    resized = image.convert("RGB").resize((input_size, input_size), resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0  # (h, w, 3)
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))


def build_pixel_values(
    images: Sequence[Any],
    *,
    input_size: int = INPUT_SIZE,
    torch: Any = None,
    dtype: Any = None,
    device: Any = None,
) -> Any:
    """Stack images into InternVL's ``(n_tiles, 3, s, s)`` pixel tensor.

    numpy does the transform and torch only wraps the result, which keeps the
    normalisation testable on a machine with no torch and makes the two-image
    path (one tile per image, concatenated along dim 0, ``num_patches_list``
    ``[1, 1]``) inspectable without a GPU.
    """
    if not images:
        raise ValueError("at least one image is required")
    stacked = np.stack([_to_tile(image, input_size) for image in images])
    if torch is None:
        return stacked
    tensor = torch.from_numpy(stacked)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class EarthDialPredictor(VLPredictorBase):
    """EarthDial-4B RGB, zero-shot.

    Same public shape as :class:`~satquery.models.qwen_vl.QwenVLPredictor` so the
    harness can swap them by registry name, but the load and decode paths are
    InternVL's, not Qwen's.
    """

    name = "earthdial"
    supported: tuple[Task, ...] = ("vqa", "caption", "ground", "change_vqa", "change_caption")
    default_model_id = "akshaydudhane/EarthDial_4B_RGB"
    prompts: Mapping[str, str] = PROMPTS
    system_prompt = SYSTEM_PROMPT
    coord_scale = 1000.0

    def _load(self) -> tuple[Any, Any]:
        torch = _require_module("torch", who=f"{self.name} inference", hint=_VLM_HINT)
        transformers = _require_module(
            "transformers", who=f"{self.name} inference", hint=_VLM_HINT
        )
        self._torch = torch

        kwargs: dict[str, Any] = {
            "torch_dtype": self._resolve_dtype(torch),
            "low_cpu_mem_usage": True,
            # The weights carry their own modelling code; without this the load
            # fails outright rather than falling back to something generic.
            "trust_remote_code": True,
            # InternVL's remote code turns FlashAttention-2 on by default. It
            # needs SM80; a T4 is SM75. This one kwarg is the difference between
            # a working free-tier baseline and an import-time crash.
            "use_flash_attn": False,
        }
        quant = self._quantization_config(torch, transformers)
        if quant is not None:
            kwargs["quantization_config"] = quant
            kwargs["device_map"] = "auto"

        LOG.info(
            "loading %s (4bit=%s, dtype=%s, flash_attn=off)",
            self.model_id,
            quant is not None,
            self.dtype,
        )
        model = transformers.AutoModel.from_pretrained(self.model_id, **kwargs)
        if quant is None and self.device not in ("", "auto"):
            model = model.to(self.device)
        # use_fast=False: the InternVL tokenisers ship slow-only, and the fast
        # conversion drops the added image tokens.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        if self.adapter:
            model = self._attach_adapter(model)
        if hasattr(model, "eval"):
            model.eval()
        return tokenizer, model

    def _generate(self, prompt: str, images: Sequence[Any], *, task: Task) -> Generation:
        model, tokenizer = self._model, self._processor
        chat = getattr(model, "chat", None)
        if chat is None:
            raise SatQueryError(
                f"{self.model_id} did not expose the InternVL-style chat(tokenizer, "
                "pixel_values, question, generation_config) method this adapter "
                "drives. Either trust_remote_code was refused or the repo's API "
                "changed; check the model card before trusting any number from it."
            )
        question = build_question(prompt, len(images))
        torch = self._torch
        pixel_values = build_pixel_values(
            images,
            input_size=INPUT_SIZE,
            torch=torch,
            dtype=self._resolve_dtype(torch) if torch is not None else None,
            device=getattr(model, "device", None) if torch is not None else None,
        )
        num_patches = [1] * len(images)
        generation_config = {
            "max_new_tokens": self.max_new_tokens_for(task),
            "do_sample": False,  # greedy: benchmark numbers must be reproducible
        }
        kwargs: dict[str, Any] = {}
        if len(images) > 1:
            kwargs["num_patches_list"] = num_patches

        raw = chat(tokenizer, pixel_values, question, generation_config, **kwargs)
        # Some InternVL forks return (response, history) when asked for history;
        # tolerate the tuple rather than stringifying it into the report.
        if isinstance(raw, tuple) and raw:
            raw = raw[0]
        return Generation(
            text=str(raw).strip(),
            confidence=None,
            meta={
                "api": "internvl_chat",
                "question": question,
                "n_images": len(images),
                "num_patches_list": num_patches,
                "input_size": INPUT_SIZE,
                "max_new_tokens": generation_config["max_new_tokens"],
                "confidence_unavailable": (
                    "model.chat() returns text only, with no token scores to average"
                ),
            },
        )
