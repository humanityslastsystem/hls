"""L0 -- Model artifact + hardware envelope."""

from __future__ import annotations

import json
from typing import Any

from pydantic import computed_field, field_validator, model_validator

from hls.core.poly import Spec

#: A chat-template argument value: the JSON scalars a template switch takes.
TemplateArg = bool | int | float | str

#: Chat-template arguments a renderer supplies itself, so a pin may not name them. A
#: conversation's ``messages`` and ``tools`` are per-request, and ``tokenize`` and
#: ``add_generation_prompt`` decide what a renderer gets back at all.
RESERVED_TEMPLATE_KEYS = frozenset({"messages", "tools", "tokenize",
                                    "add_generation_prompt"})


class ModelSpec(Spec):
    """The model a task serves: its weights at one hub commit, its architecture family and how
    its prompts are rendered. ``fingerprint`` is its identity."""

    #: The hub repository of the weights, such as "Qwen/Qwen3.6-27B".
    base_checkpoint: str
    #: The architecture family ("TX", "MoE", "HYB", ...), part of the fingerprint.
    arch: str
    #: The chat format the brief names for the implementer ("chatml", "gemma", ...).
    chat_format: str = "chatml"
    #: The hub commit the weights are read at. Provisioning refuses anything but a commit id, so
    #: the default branch name is refused there.
    revision: str = "main"
    #: Keyword arguments every renderer passes to the model's chat template beside
    #: ``messages``, ``tools`` and the generation prompt: the reference engines, the
    #: calibration pool members and, through the brief, the artifact. A switch such
    #: as ``{"enable_thinking": True}`` changes the rendered prompt and so the tokens a
    #: faithful artifact emits, which makes it part of the model's identity. The keys a
    #: renderer supplies itself (``RESERVED_TEMPLATE_KEYS``) are rejected, so a pin can
    #: never displace the conversation it is rendered with.
    chat_template_kwargs: dict[str, TemplateArg] = {}

    @field_validator("chat_template_kwargs")
    @classmethod
    def _no_reserved_keys(cls, v: dict[str, TemplateArg]) -> dict[str, TemplateArg]:
        """Refuse a pin that names an argument the renderers set themselves.

        Every renderer merges this dict into its own call to ``apply_chat_template``.
        A pin naming one of those arguments would silently win or lose depending on
        merge order and stack, and its value here is a scalar, so a pinned ``tools``
        could only ever replace a turn's real tool definitions with a string."""
        clash = sorted(set(v) & RESERVED_TEMPLATE_KEYS)
        if clash:
            raise ValueError(
                f"chat_template_kwargs may not set "
                f"{', '.join(repr(k) for k in clash)}: the renderers supply "
                f"{sorted(RESERVED_TEMPLATE_KEYS)} themselves from the request and the "
                f"call site. Drop them from the pin.")
        return v

    @model_validator(mode="wrap")
    @classmethod
    def _fingerprint_as_written(cls, data: Any, handler: Any) -> "ModelSpec":
        """A dump read back holds the computed ``fingerprint``: it must be this model's, and is
        then dropped, so an edited fingerprint is refused rather than ignored."""
        written = None
        if isinstance(data, dict) and "fingerprint" in data:
            written = data["fingerprint"]
            data = {key: value for key, value in data.items() if key != "fingerprint"}
        model = handler(data)
        if written is not None and written != model.fingerprint:
            raise ValueError(f"fingerprint is {written!r}, but this model's is {model.fingerprint!r}")
        return model

    @computed_field
    @property
    def fingerprint(self) -> str:
        base = f"{self.base_checkpoint}@{self.revision}#{self.arch}"
        # The pin is spelled on every model, empty when none is set, so a pin and its
        # absence read alike: "?" then the arguments.
        return base + "?" + "&".join(f"{k}={json.dumps(v)}"
                                     for k, v in sorted(self.chat_template_kwargs.items()))

    def __hash__(self) -> int:
        """The fingerprint is the model's identity; a dict field costs a frozen
        pydantic model the hash it would otherwise generate from its field values."""
        return hash(self.fingerprint)

    def prompt_fields(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "base_checkpoint": self.base_checkpoint,
            "arch": self.arch,
            "chat_format": self.chat_format,
            "revision": self.revision,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
        }


class AcceleratorSpec(Spec):
    kind: str                        # "B200" | "H100" | "MPS" | "CPU" | ...
    count: int
    memory_gb_per_device: int = 0    # 0 = unspecified


class InterconnectSpec(Spec):
    kind: str                        # "NVLink5" | "PCIe" | ...


def gib_of_gb(gb: int) -> int:
    """*gb* decimal GB in whole GiB, rounded down."""
    return gb * 10**9 // 2**30


def gb_of_gib(gib: int) -> int:
    """*gib* GiB in whole decimal GB, rounded down, so a figure stated in GB never overstates the
    one in GiB."""
    return gib * 2**30 // 10**9


class Hardware(Spec):
    accelerator: AcceleratorSpec
    interconnect: InterconnectSpec | None = None
    cpu_count: int = 0
    memory_gb: int = 0
    storage_gb: int = 0
    # NOTE: runtime egress is NOT a hardware property -- it lives on ``TaskSpec.network``
    # (the deployed rollout container's policy) and is collapsed to harbor's launch gate
    # at the deploy boundary. Producer-build egress is ``AgentProducerSpec.network``.

    @property
    def memory_gib(self) -> int:
        """``memory_gb``, decimal GB, in whole GiB rounded down, so a figure stated in GiB never
        overstates the one declared."""
        return gib_of_gb(self.memory_gb)

    @property
    def storage_gib(self) -> int:
        """``storage_gb``, decimal GB, in whole GiB rounded down."""
        return gib_of_gb(self.storage_gb)

    def prompt_fields(self) -> dict:
        return {
            "accelerator_kind": self.accelerator.kind,
            "accelerator_count": self.accelerator.count,
            "memory_gb_per_device": self.accelerator.memory_gb_per_device,
            "interconnect": self.interconnect.kind if self.interconnect else None,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "storage_gb": self.storage_gb,
        }
