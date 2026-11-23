"""The side channel that carries Qwen3-TTS codec codes out of a vLLM-Omni worker.

vLLM-Omni returns audio and nothing else, so the multimodal_tts baseline reads its codes
off a spool that a stage-0 producer hook writes. Tested here, without the package or a
GPU: the hook's contract with the connector that calls it, and the adapter's contract
with the files the hook leaves behind.

``vllm_omni`` is faked into ``sys.modules`` for the duration of a test rather than
installed. That covers the wrapper's argument order, that it returns the packaged
function's value untouched, and that a failure in the spooling cannot reach the
connector; the packaged function itself runs only in the baseline image.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "multimodal_tts"))

import omni_codes_spool as spool           # noqa: E402

#: The sub-talker settings this task pins. Held here rather than imported from the
#: adapter so a test that asserts the pin is not asserting the pin against itself.
_PINNED_SUBTALKER = {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
#: Codebook 0's settings this task pins, held here for the same reason.
_PINNED_TALKER = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}


@pytest.fixture(autouse=True)
def _spool_dir(tmp_path, monkeypatch):
    """Point the module at a scratch spool and forget any decode it already recorded."""
    monkeypatch.setenv(spool.SPOOL_DIR_ENV, str(tmp_path / "codes"))
    monkeypatch.setattr(spool, "_decode_recorded", False)
    return tmp_path / "codes"


class _Params:
    """Stage-0 sampling params, as the engine hands them to the hook on a request."""

    def __init__(self, extra_args=None, **fields):
        self.extra_args = extra_args
        for name, value in fields.items():
            setattr(self, name, value)


class _Request:
    """The engine's request object, reduced to what the hook reads off it."""

    def __init__(self, *, external_req_id="eng-1", key="abc", finished=True, **fields):
        self.external_req_id = external_req_id
        self.sampling_params = _Params(
            extra_args={spool.REQUEST_KEY_FIELD: key} if key else None,
            temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            min_tokens=2, max_tokens=4096, seed=None, **fields)
        self._finished = finished

    def is_finished(self):
        return self._finished


class _Config:
    def __init__(self, subtalker):
        self.subtalker_sampling_params = subtalker


class _Manager:
    """The transfer manager, reduced to the two things the hook reads off it.

    ``code_prompt_token_ids`` is a defaultdict in the engine, which is why the hook must
    not subscript it; this mirrors that so the test would catch a subscript."""

    def __init__(self, rows=None, subtalker=None, attr="config"):
        from collections import defaultdict
        self.code_prompt_token_ids = defaultdict(list)
        if rows is not None:
            self.code_prompt_token_ids["eng-1"] = rows
        setattr(self, attr, _Config(subtalker if subtalker is not None
                                    else dict(_PINNED_SUBTALKER)))


def _frame(value=7):
    return [value] * spool.CODEC_GROUPS


def _install_fake_vllm_omni(monkeypatch, fn):
    """Put a fake packaged hook where the wrapper imports it from."""
    mod = types.ModuleType("vllm_omni.model_executor.stage_input_processors.qwen3_tts")
    mod.talker2code2wav_async_chunk = fn
    for name in ("vllm_omni", "vllm_omni.model_executor",
                 "vllm_omni.model_executor.stage_input_processors"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm_omni.model_executor.stage_input_processors.qwen3_tts", mod)


# ---- the hook's contract with the connector that calls it ------------------

def test_the_wrapper_signature_matches_the_packaged_hook():
    """The connector decides whether to pass ``new_token_ids`` by inspecting the hook,
    counting a ``**kwargs`` as consent. The packaged function does not take it, so a
    wrapper that consented would be handed an argument it could not forward."""
    params = inspect.signature(spool.spooling_hook).parameters
    assert list(params) == ["transfer_manager", "multimodal_output", "request",
                            "is_finished"]
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert "new_token_ids" not in params


def test_the_packaged_return_value_is_passed_through_untouched(monkeypatch):
    sentinel = object()
    seen = {}

    def fake(transfer_manager, multimodal_output, request, is_finished):
        seen.update(mm=multimodal_output, finished=is_finished)
        return sentinel

    _install_fake_vllm_omni(monkeypatch, fake)
    mgr = _Manager(rows=[_frame()])
    out = spool.spooling_hook(mgr, {"codes": {}}, _Request(), True)

    assert out is sentinel
    assert seen == {"mm": {"codes": {}}, "finished": True}


def test_a_spooling_failure_cannot_break_the_request(monkeypatch, capsys):
    """The connector catches a hook exception and proceeds with no payload, which for
    this hook means the codec stage never receives its frames. Losing the codes is the
    price of a broken side channel; losing the audio is not."""
    def fake(transfer_manager, multimodal_output, request, is_finished):
        return "payload"

    _install_fake_vllm_omni(monkeypatch, fake)
    monkeypatch.setattr(spool, "_spool", lambda *a, **k: 1 / 0)

    assert spool.spooling_hook(_Manager(), {}, _Request(), True) == "payload"
    assert "failed to spool codes" in capsys.readouterr().out


def test_the_packaged_hook_runs_before_the_codes_are_read(monkeypatch):
    """It appends the final frame on its way through, so reading first would spool a
    matrix one frame short on every request."""
    mgr = _Manager()

    def fake(transfer_manager, multimodal_output, request, is_finished):
        transfer_manager.code_prompt_token_ids["eng-1"].append(_frame(3))
        return "payload"

    _install_fake_vllm_omni(monkeypatch, fake)
    spool.spooling_hook(mgr, {}, _Request(key="k1"), True)

    assert json.loads((spool.spool_dir() / "k1.json").read_text()) == [_frame(3)]


# ---- what lands in the spool ----------------------------------------------

def test_a_finished_request_spools_its_codes_under_the_caller_s_key():
    rows = [_frame(1), _frame(2)]
    spool._spool(_Manager(rows=rows), _Request(key="k2"), True)
    assert json.loads((spool.spool_dir() / "k2.json").read_text()) == rows


def test_an_unfinished_request_spools_nothing_but_still_reports_the_decode():
    spool._spool(_Manager(rows=[_frame()]), _Request(key="k3"), False)
    assert not (spool.spool_dir() / "k3.json").exists()
    assert (spool.spool_dir() / spool.DECODE_FILE).exists()


def test_a_request_without_our_key_is_left_alone():
    """Anything else driving this server is not ours to correlate, and a file named for
    a key we did not mint could be read as some other request's answer."""
    spool._spool(_Manager(rows=[_frame()]), _Request(key=None), True)
    assert list(spool.spool_dir().glob("*.json")) == [
        spool.spool_dir() / spool.DECODE_FILE]


@pytest.mark.parametrize("rows", [
    pytest.param([], id="no frames"),
    pytest.param([[1, 2, 3]], id="wrong width"),
    pytest.param([[0] * 16, [2048] * 16], id="code past the codebook"),
    pytest.param([[0.5] * 16], id="not integers"),
])
def test_a_malformed_matrix_is_not_spooled(rows):
    """The harness checks the same shape on the wire and is what grades it; this only
    keeps a malformed matrix from being served as if it were the model's answer."""
    spool._spool(_Manager(rows=rows), _Request(key="k4"), True)
    assert not (spool.spool_dir() / "k4.json").exists()


def test_reading_the_codes_does_not_leave_an_entry_behind():
    """The engine holds these in a defaultdict, so a subscript would create an empty
    entry for a request that never had one and leak it for the life of the server."""
    mgr = _Manager()
    spool._spool(mgr, _Request(external_req_id="never-seen", key="k5"), True)
    assert "never-seen" not in mgr.code_prompt_token_ids


def test_the_spool_is_purged_of_what_a_previous_server_left():
    d = spool.spool_dir()
    d.mkdir(parents=True)
    (d / "stale.json").write_text("[]")
    spool.purge()
    assert list(d.iterdir()) == []


# ---- the decode read-back --------------------------------------------------

def test_the_decode_is_reported_from_the_engine_s_own_config():
    spool._spool(_Manager(rows=[_frame()]), _Request(key="k6"), True)
    observed = json.loads((spool.spool_dir() / spool.DECODE_FILE).read_text())
    assert observed["subtalker_sampling_params"] == _PINNED_SUBTALKER
    assert observed["talker_sampling_params"]["temperature"] == 1.0
    assert observed["talker_sampling_params"]["repetition_penalty"] == 1.0
    assert observed["talker_sampling_params"]["min_tokens"] == 2
    assert observed["config_attr_path"] == "config"


def test_an_off_pin_sub_talker_is_reported_as_it_is_found():
    """The value is written through rather than judged here. Whoever reads it decides
    whether it is the pin, and a hook that corrected it would hide the defect."""
    spool._spool(_Manager(rows=[_frame()], subtalker={"do_sample": False}),
                 _Request(key="k7"), True)
    observed = json.loads((spool.spool_dir() / spool.DECODE_FILE).read_text())
    assert observed["subtalker_sampling_params"]["do_sample"] is False


def test_a_config_the_hook_cannot_find_reports_an_empty_path():
    """Then the reader knows the pin is unconfirmed rather than confirmed."""
    class _Bare:
        code_prompt_token_ids: dict = {}

    spool._record_decode(_Bare(), _Request(key="k8"))
    observed = json.loads((spool.spool_dir() / spool.DECODE_FILE).read_text())
    assert observed["config_attr_path"] == ""
    assert observed["subtalker_sampling_params"] is None


def test_the_decode_is_recorded_once_per_server():
    mgr = _Manager(rows=[_frame()])
    spool._spool(mgr, _Request(key="k9"), True)
    path = spool.spool_dir() / spool.DECODE_FILE
    first = path.read_text()
    mgr.config.subtalker_sampling_params = {"do_sample": True}
    spool._spool(mgr, _Request(key="k10"), True)
    assert path.read_text() == first


def test_a_reader_never_sees_half_a_matrix(tmp_path):
    """The reader is another process polling for the file to appear, so a partial write
    would be read as malformed codes."""
    target = tmp_path / "sub" / "codes.json"
    spool._write_atomic(target, [_frame()])
    assert json.loads(target.read_text()) == [_frame()]
    assert list(target.parent.glob("*.part")) == []


# ---- the adapter's side: what it does with what the hook left --------------

import server_vllm_omni as adapter       # noqa: E402  (needs the path insert above)


def _record(**over):
    """An engine decode report of the shape the hook writes, pinned unless overridden."""
    talker = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
              "min_tokens": 2, "max_tokens": 4096, "seed": None}
    talker.update(over.pop("talker", {}))
    payload = {"config_attr_path": "config",
               "subtalker_sampling_params": dict(_PINNED_SUBTALKER),
               "talker_sampling_params": talker,
               "transfer_manager_type": "ChunkTransferAdapter"}
    payload.update(over)
    spool._write_atomic(spool.spool_dir() / spool.DECODE_FILE, payload)


def test_the_pinned_decode_is_accepted():
    _record()
    assert adapter._check_pin() == []


def test_a_missing_report_is_a_complaint():
    """No file means the hook never ran, so neither the decode nor the codes side
    channel is confirmed. Silence is not agreement."""
    assert "did not run" in adapter._check_pin()[0]


@pytest.mark.parametrize("field,value", [
    pytest.param("do_sample", False, id="argmax instead of sampling"),
    pytest.param("temperature", 0.7, id="another temperature"),
    pytest.param("top_k", 20, id="another top-k"),
    pytest.param("top_p", 0.8, id="a truncation the pin does not carry"),
])
def test_a_sub_talker_off_the_pin_is_refused(field, value):
    """vLLM-Omni validates none of these keys, so any one of them is the same silent
    class of defect. A greedy sub-talker in particular is what runs the talker into
    non-terminating repetition."""
    sub = dict(_PINNED_SUBTALKER)
    sub[field] = value
    _record(subtalker_sampling_params=sub)
    assert any(f"sub-talker {field}" in c for c in adapter._check_pin())


def test_a_sub_talker_the_hook_could_not_find_is_refused():
    _record(subtalker_sampling_params=None, config_attr_path="")
    assert any("unconfirmed" in c for c in adapter._check_pin())


# -- the engine may spell a setting differently than it was pinned --

@pytest.mark.parametrize("got,pinned,same", [
    pytest.param(0, -1, True, id="zero against minus one"),
    pytest.param(-1, 0, True, id="minus one against zero"),
    pytest.param(-1, -1, True, id="unchanged"),
    pytest.param(50, -1, False, id="a real truncation where none was pinned"),
    pytest.param(0, 50, False, id="no truncation where one was pinned"),
])
def test_only_the_disabled_spellings_of_top_k_are_interchangeable(got, pinned, same):
    """vLLM takes top_k -1 and hands back 0, both meaning no truncation, so the pin and
    its echo may spell "off" differently.

    The equivalence is between two ways of saying off, not between any two values: a
    group pinned to top-k 50 and serving with it disabled is a different decode, and
    reading -1 and 0 as wildcards would swallow it."""
    assert adapter._same_setting("top_k", got, pinned) is same


@pytest.mark.parametrize("got,pinned,same", [
    pytest.param(1, 1.0, True, id="int against float"),
    pytest.param(1.05, 1.05, True, id="float unchanged"),
    pytest.param(0.0, 0.0, True, id="zero temperature"),
    pytest.param(0.9, 0.0, False, id="a real temperature"),
    pytest.param(None, 1.0, False, id="the engine reported nothing"),
])
def test_a_number_is_compared_by_value_not_by_type(got, pinned, same):
    """The engine rebuilds these into its own objects, so an int may come back a float.
    A key it does not report at all is still a complaint: unreported is unconfirmed."""
    assert adapter._same_setting("temperature", got, pinned) is same


@pytest.mark.parametrize("got,pinned,same", [
    pytest.param(True, True, True, id="unchanged"),
    pytest.param(1, True, True, id="one against true"),
    pytest.param(False, True, False, id="the switch is off"),
    pytest.param(0, True, False, id="zero against true"),
    pytest.param(2, True, False, id="a truthy value that is not the pin"),
    pytest.param("true", True, False, id="a string that is not the pin"),
])
def test_a_switch_is_compared_by_value_and_not_by_truthiness(got, pinned, same):
    """do_sample decides whether fifteen code groups sample at all, and the numeric rule
    already reads True, 1 and 1.0 as one setting.

    The last two cases are why there is no truthiness branch. Under one, a do_sample of
    2 or of "true" would pass as the pin, and a value nobody can account for is the
    thing this check exists to surface rather than to wave through."""
    assert adapter._same_setting("do_sample", got, pinned) is same


def test_a_real_deviation_still_fails_after_the_normalization_rule():
    """The point of the rule is to stop false alarms, not to stop alarms. A talker at
    temperature 0 is taking the argmax, which is the decode this checkpoint does not
    terminate under and the one the gate exists to catch."""
    _record(talker={"temperature": 0.0})
    assert any("talker temperature" in c for c in adapter._check_pin())


def test_a_trip_reports_the_whole_record_the_engine_gave():
    """A trip stops every later request, so the log line that names one complaint is
    also the last chance to see everything else the engine said."""
    _record(talker={"temperature": 0.9})
    observed = adapter._observed_decode()
    assert "talker_sampling_params" in observed
    assert "subtalker_sampling_params" in observed


def test_an_unreadable_record_reports_why_rather_than_raising():
    assert "unreadable" in adapter._observed_decode()


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.0), ("top_k", 20), ("top_p", 0.8), ("repetition_penalty", 1.05),
])
def test_a_talker_off_the_pin_is_refused(field, value):
    _record(talker={field: value})
    assert any(field in c for c in adapter._check_pin())


def test_a_different_end_of_speech_block_is_refused():
    """The reference blocks end-of-speech for exactly two opening frames; a server that
    blocked it for another number has its opening scored against a distribution the
    reference never saw."""
    _record(talker={"min_tokens": 1})
    assert any("min_tokens" in c for c in adapter._check_pin())


def test_a_seeded_talker_is_refused():
    """A seed makes vLLM-Omni run the sub-talker one row at a time, which costs
    throughput; the adapter writes none, so one that comes back is not its config."""
    _record(talker={"seed": 42})
    assert any("seed" in c for c in adapter._check_pin())


def test_the_adapter_reads_the_codes_and_deletes_the_file():
    """Under a sealed container the spool shares a small tmpfs with everything else, so
    what a drive leaves behind must be bounded by requests in flight, not by requests
    served."""
    import asyncio
    rows = [_frame(5)]
    spool.spool_dir().mkdir(parents=True, exist_ok=True)
    spool._write_atomic(spool.spool_dir() / "live.json", rows)

    assert asyncio.run(adapter._read_codes("live")) == rows
    assert not (spool.spool_dir() / "live.json").exists()


def test_codes_that_never_arrive_time_out_rather_than_hang(monkeypatch):
    """A reply then goes out with no codes frame, which the harness records against it.
    Blocking instead would turn a missing side channel into a hung drive."""
    import asyncio
    monkeypatch.setattr(adapter, "_CODES_WAIT_SEC", 0.05)
    spool.spool_dir().mkdir(parents=True, exist_ok=True)
    assert asyncio.run(adapter._read_codes("absent")) is None


# ---- the deploy config the engine is actually started on -------------------
# This rewrite IS the pin: everything else only reports on it.

#: The packaged config's shape, with the checkpoint's own decode on the talker stage
#: (0.9, top-k 50, penalty 1.05), which the rewrite must replace.
_PACKAGED_STAGES = [
    {"stage_id": 0,
     "default_sampling_params": {"temperature": 0.9, "top_k": 50, "max_tokens": 4096,
                                 "min_tokens": 2, "repetition_penalty": 1.05},
     "subtalker_sampling_params": {"do_sample": True, "temperature": 0.9,
                                   "top_k": 50, "top_p": 1.0}},
    {"stage_id": 1,
     "default_sampling_params": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                                 "max_tokens": 65536, "repetition_penalty": 1.0}},
]


@pytest.fixture
def packaged_yaml(tmp_path, monkeypatch):
    """A fake installed vllm_omni with a packaged deploy config of the shipped shape."""
    import yaml
    pkg = tmp_path / "vllm_omni"
    (pkg / "deploy").mkdir(parents=True)
    (pkg / "deploy" / "qwen3_tts.yaml").write_text(
        yaml.safe_dump({"async_chunk": True, "connectors": {"c": {"name": "shm"}},
                        "stages": [dict(s) for s in _PACKAGED_STAGES]}))
    mod = types.ModuleType("vllm_omni")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "vllm_omni", mod)
    return tmp_path


def _written(monkeypatch, tmp_path):
    """Run the rewrite, with only its output path redirected into the test's directory.

    Everything else goes through the real ``Path``, so what is exercised is the function
    the server runs and not a stand-in for it."""
    import yaml
    out = tmp_path / "pinned.yaml"
    real_path = adapter.Path
    monkeypatch.setattr(adapter, "Path",
                        lambda p=".", _r=real_path: out if str(p).endswith(
                            "qwen3_tts_pinned.yaml") else _r(p))
    assert adapter._deploy_config() == str(out)
    return yaml.safe_load(out.read_text())["stages"]


def test_codebook_zero_is_pinned_to_the_contract_s_decode(packaged_yaml, monkeypatch):
    """The plain softmax at temperature 1 replaces the packaged 0.9, top-k 50 and
    penalty 1.05."""
    stage0 = _written(monkeypatch, packaged_yaml)[0]
    assert {k: stage0["default_sampling_params"][k] for k in _PINNED_TALKER} == _PINNED_TALKER


def test_the_sub_talker_is_pinned_to_the_contract_s_decode(packaged_yaml, monkeypatch):
    """Sampled, and every setting stated rather than inherited: vLLM-Omni validates none
    of these keys, so one left to a default is one nothing would notice changing."""
    stage0 = _written(monkeypatch, packaged_yaml)[0]
    assert stage0["subtalker_sampling_params"] == _PINNED_SUBTALKER


def test_both_sampling_tables_are_the_decode_the_task_declares():
    """The adapter runs in the baseline image, which holds no hls source, so it keeps its own
    copy of the contract's decode. The values here are read from the reference the task's gates
    are declared with: the plain softmax keeps the whole vocabulary, which vLLM spells top-p 1.0."""
    from tasks.multimodal_tts import task

    declared = task.VERIFIERS[0].reference
    assert all(v.reference is declared for v in task.VERIFIERS)
    assert adapter._TALKER_SAMPLING == {"temperature": declared.temperature, "top_k": declared.top_k,
                                        "top_p": 1.0, "repetition_penalty": declared.repetition_penalty}
    assert adapter._SUBTALKER_SAMPLING == {"do_sample": declared.subtalker_sample,
                                           "temperature": declared.subtalker_temperature,
                                           "top_k": declared.subtalker_top_k, "top_p": 1.0}
    # The literals the other tests of this file hold are those same values.
    assert (adapter._TALKER_SAMPLING, adapter._SUBTALKER_SAMPLING) == (_PINNED_TALKER, _PINNED_SUBTALKER)


def test_the_adapter_restates_the_tasks_request_values():
    """Its own copies of the speaker, the language, the checkpoint and the codec's shape."""
    from tasks.multimodal_tts import task

    declared = task.VERIFIERS[0].reference
    assert (adapter._SPEAKER, adapter._LANGUAGE) == (declared.speaker.lower(), declared.language)
    assert adapter.MODEL_ID == task.MODEL.base_checkpoint
    assert (spool.CODEC_GROUPS, spool.CODEC_CODEBOOK_SIZE) == (task.CONTRACT.protocol.codec_groups,
                                                               task.CONTRACT.protocol.codec_codebook_size)


def test_the_end_of_speech_block_is_left_as_packaged(packaged_yaml, monkeypatch):
    """The reference reproduces exactly the packaged two frames when it teacher-forces,
    so this is a value to preserve rather than one to choose."""
    stage0 = _written(monkeypatch, packaged_yaml)[0]
    assert stage0["default_sampling_params"]["min_tokens"] == 2
    assert stage0["default_sampling_params"]["max_tokens"] == 4096


def test_no_seed_is_written(packaged_yaml, monkeypatch):
    """A seed makes vLLM-Omni build per-row generators and, on a graph-wrapped talker,
    run the sub-talker one row at a time, which costs throughput on the measurement this
    server exists to take. The sampled groups are not held to a value by any gate, so
    the reproducibility it would buy is reproducibility nothing reads."""
    for stage in _written(monkeypatch, packaged_yaml):
        assert "seed" not in stage["default_sampling_params"]


def test_the_codes_hook_is_installed_on_the_talker_stage_only(packaged_yaml,
                                                              monkeypatch):
    stages = _written(monkeypatch, packaged_yaml)
    assert stages[0]["custom_process_next_stage_input_func"] == adapter._SPOOL_HOOK
    assert "custom_process_next_stage_input_func" not in stages[1]


def test_the_fallback_installs_no_hook(packaged_yaml, monkeypatch):
    """HLS_TTS_CODES=0 keeps the decode pin and drops the side channel, which is the
    configuration for a latency measurement that must not be staked on it."""
    monkeypatch.setattr(adapter, "_CODES_ENABLED", False)
    stages = _written(monkeypatch, packaged_yaml)
    assert "custom_process_next_stage_input_func" not in stages[0]
    assert stages[0]["subtalker_sampling_params"] == _PINNED_SUBTALKER


def test_a_config_with_no_sub_talker_stage_is_refused(packaged_yaml, monkeypatch):
    """That key is the only handle on fifteen of the sixteen groups. Serving them under
    settings nothing states is worse than not serving."""
    import yaml
    pkg = packaged_yaml / "vllm_omni" / "deploy" / "qwen3_tts.yaml"
    cfg = yaml.safe_load(pkg.read_text())
    del cfg["stages"][0]["subtalker_sampling_params"]
    pkg.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="subtalker_sampling_params"):
        _written(monkeypatch, packaged_yaml)
