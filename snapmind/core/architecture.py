# ─── SECTION: Architecture Registry ──────────────────────
from __future__ import annotations

from dataclasses import dataclass

from snapmind.core.registry import Registry

ARCHITECTURE: Registry = Registry("architecture", expected_type=object)
"""Registry for :class:`SupportedArchitecture` records, keyed by model name."""


@dataclass
class SupportedArchitecture:
    """Declarative record for a supported model architecture.

    One instance per model family. Consolidates the model class, default config,
    HuggingFace weight source, and tokenizer source into a single registration.

    Register via::

        ARCHITECTURE.register("gpt2", SupportedArchitecture(name="gpt2", ...))
    """

    name: str
    """Short name used for CLI ``--model`` selection (e.g. ``"gpt2"``, ``"llama"``)."""

    model_cls: type
    """Python class that implements the model (a :class:`BaseModelABC` subclass)."""

    default_config: dict
    """Default :class:`ModelConfig` fields for this architecture.
    Passed as keyword args when constructing a :class:`ModelConfig`.
    """

    hf_repo: str | None = None
    """HuggingFace repo ID for weight downloads (e.g. ``"openai-community/gpt2"``).
    ``None`` means no pretrained weights are available (random weights only).
    """

    hf_filename: str = "model.safetensors"
    """Filename in the HF repo (``"model.safetensors"``, ``"consolidated.safetensors"``, …)."""

    tokenizer_hf_repo: str | None = None
    """HF repo ID for the tokenizer, if different from *hf_repo*.
    ``None`` falls back to *hf_repo*.
    """


# ─── ENDSECTION: Architecture Registry ──────────────────
