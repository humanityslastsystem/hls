import pytest
from pydantic import ValidationError

from hls.core.model import AcceleratorSpec, Hardware, ModelSpec


def test_modelspec_revision_defaults_main_and_is_in_fingerprint():
    m = ModelSpec(base_checkpoint="org/m", arch="TX", revision="abc123")
    assert m.revision == "abc123"
    assert "abc123" in m.fingerprint
    assert ModelSpec(base_checkpoint="org/m", arch="TX").revision == "main"


def test_a_model_spec_has_no_adapters():
    """A model is its checkpoint, revision, architecture and template pin, and its fingerprint
    spells those; a task that names adapters is refused."""
    m = ModelSpec(base_checkpoint="meta-llama/Llama-3.1-8B-Instruct", arch="TX",
                  chat_format="llama3-instruct")
    assert "adapters" not in ModelSpec.model_fields
    assert m.fingerprint == "meta-llama/Llama-3.1-8B-Instruct@main#TX?"
    with pytest.raises(ValidationError, match="adapters"):
        ModelSpec(base_checkpoint="foo", arch="TX", adapters=())


def test_hardware_is_physical_only():
    # Runtime egress is NOT a hardware property -- it lives on TaskSpec.network.
    hw = Hardware(accelerator=AcceleratorSpec(kind="B200", count=1, memory_gb_per_device=192))
    assert hw.accelerator.count == 1
    assert not hasattr(hw, "network")
    assert not hasattr(hw, "effective_domain_policy")


def test_hardware_states_decimal_gb_and_converts_to_whole_gib_rounded_down():
    """A task's hardware states memory and storage in decimal GB; a shape or a brief stated in GiB
    takes them rounded down, so neither overstates the other, and the brief's GB from a GiB
    figure the same way."""
    from hls.core.model import gb_of_gib

    hw = Hardware(accelerator=AcceleratorSpec(kind="B200", count=1), memory_gb=512, storage_gb=2048)
    assert (hw.memory_gib, hw.storage_gib) == (476, 1907)
    assert hw.memory_gib * 2**30 <= 512 * 10**9 < (hw.memory_gib + 1) * 2**30
    assert gb_of_gib(180) == 193 and gb_of_gib(180) * 10**9 <= 180 * 2**30 < (gb_of_gib(180) + 1) * 10**9
    # The figures are views of the declared ones, never fields of their own.
    assert "memory_gib" not in hw.model_dump() and "storage_gib" not in hw.model_dump()


def test_model_spec_prompt_fields():
    from hls.core.model import ModelSpec
    m = ModelSpec(base_checkpoint="meta-llama/Llama-3.1-8B-Instruct", arch="TX",
                  chat_format="llama3-instruct")
    f = m.prompt_fields()
    assert f["fingerprint"] == "meta-llama/Llama-3.1-8B-Instruct@main#TX?"
    assert f["base_checkpoint"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert f["arch"] == "TX"
    assert f["chat_format"] == "llama3-instruct"
    assert f["revision"] == "main"


def test_hardware_prompt_fields():
    from hls.core.model import AcceleratorSpec, Hardware, InterconnectSpec
    hw = Hardware(accelerator=AcceleratorSpec(kind="B200", count=2, memory_gb_per_device=183),
                  interconnect=InterconnectSpec(kind="NVLink5"),
                  cpu_count=64, memory_gb=512)
    f = hw.prompt_fields()
    assert f["accelerator_kind"] == "B200"
    assert f["accelerator_count"] == 2
    assert f["memory_gb_per_device"] == 183
    assert f["interconnect"] == "NVLink5"
    assert f["cpu_count"] == 64
    assert "allow_internet" not in f         # runtime egress is not a hardware field


def test_hardware_prompt_fields_no_interconnect():
    from hls.core.model import AcceleratorSpec, Hardware
    hw = Hardware(accelerator=AcceleratorSpec(kind="CPU", count=1))
    assert hw.prompt_fields()["interconnect"] is None


def test_chat_template_kwargs_default_empty_and_in_the_dump():
    """No pin is an empty one, and the dump names it like any other field, so the workload
    payload holds every field of the model."""
    from hls.core.poly import dump_spec
    m = ModelSpec(base_checkpoint="org/m", arch="TX")
    assert m.chat_template_kwargs == {}
    assert m.fingerprint == "org/m@main#TX?"  # the pin suffix is spelled even when empty
    assert dump_spec(m)["chat_template_kwargs"] == {}
    assert m.prompt_fields()["chat_template_kwargs"] == {}


def test_chat_template_kwargs_in_fingerprint_dump_and_round_trip():
    from hls.core.poly import dump_spec
    m = ModelSpec(base_checkpoint="org/m", arch="TX", revision="r1",
                  chat_template_kwargs={"enable_thinking": True, "effort": "high"})
    assert m.fingerprint == 'org/m@r1#TX?effort="high"&enable_thinking=true'
    dumped = dump_spec(m)
    assert dumped["chat_template_kwargs"] == {"enable_thinking": True, "effort": "high"}
    assert ModelSpec.model_validate(dumped) == m
    assert m.prompt_fields()["chat_template_kwargs"] == {"enable_thinking": True,
                                                         "effort": "high"}


def test_chat_template_kwargs_rejects_a_reserved_key():
    """Every renderer merges the pin into its own apply_chat_template call, so a pin
    naming one of the arguments the renderer supplies would win or lose by merge order.
    Rejecting it here makes the two merge paths agree by construction."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="tools"):
        ModelSpec(base_checkpoint="org/m", arch="TX", chat_template_kwargs={"tools": "x"})
    for key in ("messages", "tokenize", "add_generation_prompt"):
        with pytest.raises(ValidationError, match=key):
            ModelSpec(base_checkpoint="org/m", arch="TX", chat_template_kwargs={key: "x"})
    # A pin that names no reserved key is untouched.
    assert ModelSpec(base_checkpoint="org/m", arch="TX",
                     chat_template_kwargs={"enable_thinking": True}
                     ).chat_template_kwargs == {"enable_thinking": True}


def test_chat_template_kwargs_changes_workload_fingerprint():
    """A different chat-template pin renders a different prompt, so it must move the
    workload fingerprint the same way the removal list does (see
    test_the_removal_list_is_part_of_the_workload_identity in
    test_datasource_removals.py)."""
    from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.datasets import HFDataset
    from hls.spec.datasource import Count, DataSource, SplitSpec
    from hls.spec.taskspec import TaskSpec
    from hls.spec.wire import OpenAIChatProtocol
    from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
    from hls.verify import LetterMatchMetric, MetricVerifier
    from hls.verify.calibration import workload_fingerprint

    from hls.prompt import JinjaPromptTemplate

    def _dataset() -> HFDataset:
        return HFDataset(path="cais/mmlu", subset="abstract_algebra", revision="main",
                         source_split="test",
                         prompt=JinjaPromptTemplate(template="{{ question }}"),
                         target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))

    def task(chat_template_kwargs):
        return TaskSpec(
            id="T",
            model=ModelSpec(base_checkpoint="m", arch="TX",
                            chat_template_kwargs=chat_template_kwargs),
            hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
            contract=RequestResponseContract(
                path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                output_modalities=frozenset({"text"}), protocol=OpenAIChatProtocol()),
            workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                              output_length_cap=Uncapped(), slo=NoSLO()),
            data_source=DataSource(dataset=_dataset(),
                                   split=SplitSpec(dev=Count(value=1), test=Count(value=4))),
            verifiers=[MetricVerifier(metric=LetterMatchMetric())],
            scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))

    bare, pinned = (workload_fingerprint(task(k))
                    for k in ({}, {"enable_thinking": True}))
    assert bare != pinned


def test_model_spec_hashes_by_fingerprint():
    """A dict field costs a frozen pydantic model its generated hash, so ModelSpec
    supplies one over the fingerprint. Equal specs must land on the same value."""
    bare = ModelSpec(base_checkpoint="org/m", arch="TX")
    pinned = ModelSpec(base_checkpoint="org/m", arch="TX",
                       chat_template_kwargs={"enable_thinking": True})
    assert hash(bare) == hash(ModelSpec(base_checkpoint="org/m", arch="TX"))
    assert hash(pinned) == hash(ModelSpec(base_checkpoint="org/m", arch="TX",
                                          chat_template_kwargs={"enable_thinking": True}))
    assert hash(bare) != hash(pinned)
    assert len({bare, pinned}) == 2
