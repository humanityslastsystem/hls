"""What the brief has to say for a contestant to build the right server.

The brief is part of the task's definition: a sentence missing from it is a difference between
contestants that the score carries. Each number it states is read off the constant or field the
harness enforces, and these tests hold the brief to them.
"""

from __future__ import annotations

import difflib
import re

import pytest

from hls.pipeline import stage
from hls.pipeline.pipelines import producer_network
from hls.prompt import blocked_packages_of, uploads_brief_section
from hls.reference import service_harness as rs
from hls.util import remove_tree
from hls.verify.report import TOP_LOGPROBS_REQUESTED
from tests.agent import test_stage
from tests.support import node
from tests.unit.brief_task import render_brief

from tasks.agentic_coding import task as task_mod


def _brief():
    """The brief and its provenance, with the producer's deny list, uploads and reference
    service flag, as the agent_build stage gives them."""
    producer = task_mod.ARTIFACT.producer
    return render_brief(task_mod.TASK, task_mod.ARTIFACT.brief,
                        blocked_packages=blocked_packages_of(producer.network),
                        uploads_section=uploads_brief_section(producer.upload_brief, producer.upload_dest),
                        reference_service=producer.reference_service)


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """The brief the agent_build stage writes for this task with the producer's reference
    service flag set as given. One stage per test, its network the producer's, on the test
    node of the harness's own suite (``tests.conftest.hermetic_node``), whose fixtures this
    directory does not load."""
    monkeypatch.setattr(stage, "hard_limit", lambda: node.HARD)
    monkeypatch.setattr(stage, "node_limits", lambda: dict(node.UCOUNTS))
    monkeypatch.setattr(stage, "uid_tasks", lambda uid: node.UID_TASKS)
    monkeypatch.setattr(stage, "node_tasks", lambda: node.NODE_TASKS)
    monkeypatch.setattr(stage, "node_pid_max", lambda: node.PID_MAX)
    built: list = []

    def brief(reference_service: bool) -> str:
        producer = task_mod.ARTIFACT.producer.model_copy(update={"reference_service": reference_service})
        artifact = task_mod.ARTIFACT.model_copy(update={"producer": producer})
        if not built:
            params = test_stage._params(network=producer_network(producer))
            built.append(test_stage.briefing(test_stage._ctx(tmp_path, params)))
        return built[0].brief(task_mod.TASK, artifact)

    yield brief
    while test_stage._CONTEXTS:
        test_stage._CONTEXTS.pop().close()
    while test_stage._SCRATCHES:
        remove_tree(test_stage._SCRATCHES.pop(), ignore_errors=True)


def test_the_brief_states_the_window_the_framework_and_the_reference_run_at():
    # One constant: the loop cuts a trajectory at it and the reference is built at it.
    brief, _ = _brief()
    assert f"- Context window: {task_mod.MAX_CONTEXT_LEN:,} tokens; serve the model at this window." in brief
    assert task_mod.AGENT.framework.max_context_len == task_mod.MAX_CONTEXT_LEN
    assert task_mod.VERIFIERS[0].reference.max_model_len == task_mod.MAX_CONTEXT_LEN


def test_the_brief_states_every_device_of_the_node():
    count = task_mod.TASK.hardware.accelerator.count
    assert count == 8
    assert f"The accelerators are `cuda:0` through `cuda:{count - 1}`" in _brief()[0]


def test_the_brief_states_the_turn_bound_the_driver_holds_a_session_to():
    # Read off the agent the trajectory driver runs, so the bound stated is the one enforced.
    brief, prov = _brief()
    bound = task_mod.TASK.agent.max_turns
    assert bound == task_mod.MAX_TURNS
    workload = {s.block: s for s in prov.sections}["workload"].rendered
    assert f"A session's turns are generated as it runs, up to {bound}, and each turn" in _flat(workload)
    assert f"- Turns: up to {bound} per session\n" in workload


def test_the_brief_says_what_the_reference_is():
    # Unnamed, "the reference model" reads as a transformers forward pass, the one stack
    # this checkpoint has no kernel in.
    brief, prov = _brief()
    assert ("The reference is this checkpoint served by vLLM at commit `a8d1aa9c` on the "
            "FP8 weights as shipped, decoding under the rules above. A `transformers` "
            "forward pass in bf16 is not the reference.") in _flat(brief)
    decoding = {s.block: s for s in prov.sections}["decoding"]
    assert decoding.source == "task"
    # Appended to the inherited paragraphs, not in place of them.
    assert "**Verification.**" in decoding.rendered


def test_the_brief_leaves_precision_to_the_contestant():
    assert ("Numerical precision and approximations are yours to choose; the "
            "verification above is the only faithfulness constraint.") in _flat(_brief()[0])


def test_the_brief_says_the_gate_teacher_forces_through_the_prefill_path():
    assert ("The reference distribution at a served position is computed by "
            "teacher-forcing your served tokens through the reference's prefill path; "
            "your reported distribution is whatever path produced your sample.") in \
        _flat(_brief()[0])


def test_the_brief_says_the_ceiling_absorbs_the_prefill_decode_difference():
    assert ("Prefill and decode use different kernels in most stacks, including the "
            "reference; the threshold was calibrated on a serving stack's sampled tokens "
            "scored this way, so it absorbs that difference.") in _flat(_brief()[0])


def test_the_commit_named_as_the_reference_is_read_off_the_pin():
    brief, _ = _brief()
    assert f"vLLM at commit `{task_mod.VLLM_COMMIT[:8]}`" in _flat(brief)
    assert len(task_mod.VLLM_COMMIT) == 40


def test_the_brief_names_the_architecture_source_on_disk():
    flat = _flat(_brief()[0])
    assert ("The installed `transformers` package contains an implementation of this "
            "architecture at `transformers/models/glm5_next/modeling_glm5_next.py`.") in flat


def test_the_brief_states_the_measured_transformers_distance():
    flat = _flat(_brief()[0])
    assert ("Run in bf16 on this checkpoint, that implementation is 6.0e-3 nats per token "
            "from the reference at generated positions, measured over 36 prompts of up to "
            "8,192 tokens.") in flat


def test_the_brief_does_not_vouch_for_the_transformers_implementation():
    flat = _flat(_brief()[0])
    assert ("It is not the reference your distributions are measured against; that "
            "reference is named under ## Decoding.") in flat
    for vouching in ("architecture reference", "the reference implementation",
                     "This checkpoint's architecture definition"):
        assert vouching not in flat


def test_the_uploads_section_does_not_claim_to_be_the_only_reference():
    # It is not: the transformers source it names is on disk too.
    uploads = _flat({s.block: s for s in _brief()[1].sections}["uploads"].rendered)
    assert uploads.startswith("## Toolchain Documentation")
    assert "There is no other reference available to you" not in uploads
    assert "relying on recall" not in uploads


def test_no_part_of_the_brief_claims_to_be_the_only_reference():
    # The uploads name the transformers source, the reference logprobs and the scoring
    # service, so no sentence anywhere may say that nothing else can serve as one.
    flat = _flat(_brief()[0])
    assert "no other reference" not in flat.lower() and "relying on recall" not in flat
    assert ("Installed, with offline documentation at the paths given (relative to your working "
            "directory):") in flat


def test_the_toolchain_is_the_one_the_tasks_sandbox_holds():
    # The brief states the toolchain of the task's own sandbox image.
    flat = _flat(_brief()[0])
    assert ("`TORCH_CUDA_ARCH_LIST` is set for this GPU; torch's `torch/include` headers and vendored "
            "pybind11 are present, so `torch.utils.cpp_extension.load()` and setuptools "
            "`BuildExtension` build offline.") in flat
    assert "- Triton, with runnable annotated tutorials -- `references/triton/` - CUTLASS" in flat
    assert "PyTorch C++/CUDA extensions build here" not in flat and "27 runnable" not in flat


def test_the_brief_states_the_readiness_probe_the_harness_sends():
    # hls.drive.readiness.wait_inference_ready: after /health, the contract's own probe
    # until a status below 400 or exactly 400.
    brief = _brief()[0]
    assert ("the moment it sees 200 and immediately sends inference traffic.\n\nAfter /health returns 200, "
            "the harness sends one probe request to the inference\nendpoint below and repeats it until the "
            "response status is 2xx, 3xx or exactly 400.\nAny other status (404, 405, 422, other 4xx, 5xx) or "
            "no response keeps the server not\nready, and the startup timeout fails the run.\n\n### POST "
            "/v1/chat/completions\n") in brief


def test_the_accelerators_line_does_not_give_the_agent_every_gpu():
    # The scoring service shares them.
    brief = _brief()[0]
    assert "`$HLS_WEIGHTS_DIR`. The accelerators are `cuda:0` through `cuda:7`.\n- Data:" in brief
    assert "all 8 are yours" not in brief


def test_the_response_sample_does_not_contradict_the_prose():
    # The prose demands 21 alternatives at every position and the sample shows two, so
    # the line above the sample says which is the rule.
    flat = _flat(_brief()[0])
    assert f"the top {TOP_LOGPROBS_REQUESTED} alternatives at that position" in flat
    assert (f"each `top_logprobs` array is abbreviated to 2 entries below; a conforming "
            f"reply carries {TOP_LOGPROBS_REQUESTED} at every position") in flat


def test_the_scored_sequence_length_is_the_reference_bundles():
    flat = _flat(_brief()[0])
    assert task_mod.REFERENCE_SERVICE_SCORE_32K_TOKENS == 32_709
    assert (f"A {task_mod.REFERENCE_SERVICE_SCORE_32K_TOKENS:,}-token sequence scores in "
            f"{task_mod.REFERENCE_SERVICE_SCORE_32K_SEC} s awake.") in flat


def test_the_brief_documents_the_reference_logprobs():
    flat = _flat(_brief()[0])
    assert "`references/glm53-reference/` holds reference output for self-checking." in flat
    assert ("`prompts.jsonl` carries one record per prompt, with its token ids and its "
            "text, at lengths from a few thousand to several hundred thousand tokens.") in flat
    assert ("`reference.jsonl` carries, per prompt, the reference's sampled continuation "
            "token ids and, at every position, two distributions: `decode`, streamed by "
            "the reference while it sampled that continuation, and `forced`, the "
            "reference's prefill path teacher-forced on the same continuation, which is "
            "the path the grader uses.") in flat
    assert ("Each gives the served token's log-probability and the "
            f"{TOP_LOGPROBS_REQUESTED} most probable alternatives with their token ids "
            "and log-probabilities, at temperature 1.") in flat
    assert ("`selfcheck.py` reads a file of that shape from your own server and computes "
            f"the KL statistic and the top-{TOP_LOGPROBS_REQUESTED} coverage the grader "
            "computes, against `forced` by default, reporting prefill and generated "
            "positions separately.") in flat


def test_the_brief_documents_the_sampled_prompt_positions():
    flat = _flat(_brief()[0])
    assert ("Each prompt also carries `prompt_forced`, the reference's prefill-path "
            f"top-{TOP_LOGPROBS_REQUESTED} at a recorded sample of prompt positions -- "
            "every position in the last 1,024 tokens of the prompt and every 64th "
            "position before that -- so a prefill kernel can be checked against the "
            "reference directly.") in flat


def test_the_brief_says_how_the_two_reference_distributions_relate():
    flat = _flat(_brief()[0])
    assert "The two differ at the level of kernel noise." in flat
    assert "The reference itself cannot be run in your sandbox." in flat
    # The upload root the files are stated relative to.
    assert (f"These files are available at `{task_mod.ARTIFACT.producer.upload_dest}` inside "
            "your sandbox.") in flat


def test_the_brief_documents_the_reference_scoring_service():
    # The service's interface and behaviour, with every path and constant read off the
    # harness module the build stage and the service share.
    brief, _ = _brief()
    flat = _flat(brief)
    assert "## Reference Scoring Service" in brief
    assert ("During the build, and only then, a service scores token sequences with the "
            "grader's reference: it teacher-forces `prompt_token_ids` followed by "
            "`continuation_token_ids` through the reference named under ## Decoding, on "
            "its prefill path with prefix caching off, which is how the grader scores your "
            "served tokens.") in flat
    assert (f"It serves HTTP/1.0 on the unix socket at `${rs.SOCKET_ENV_VAR}` "
            f"(`{rs.SOCKET_PATH}`), one request per connection, one request at a time. "
            "It has no TCP port.") in flat
    for route in ("GET /status", "POST /score", "POST /sleep, POST /wake"):
        assert route in flat
    assert f'"idle_sleep_sec": {rs.IDLE_SLEEP_SEC}' in flat
    assert f'"top{TOP_LOGPROBS_REQUESTED}": [[int, float], ...]' in flat
    assert (f"more than {task_mod.MAX_CONTEXT_LEN - 1:,} tokens in total, or nothing to "
            "score returns HTTP 400.") in flat
    assert (f"Awake it holds {task_mod.REFERENCE_SERVICE_AWAKE_GIB} GiB on each GPU; asleep, "
            f"{task_mod.REFERENCE_SERVICE_ASLEEP_GIB} GiB.") in flat
    assert f"`/score` and `/wake` wake it in {task_mod.REFERENCE_SERVICE_WAKE_SEC} s." in flat
    assert (f"It goes back to sleep {rs.IDLE_SLEEP_SEC} s after its last `/score` or "
            f"`/wake`, or on `/sleep` in {task_mod.REFERENCE_SERVICE_SLEEP_SEC} s.") in flat
    assert (f"A wake needs `wake_need_bytes` ({task_mod.REFERENCE_SERVICE_WAKE_NEED_GIB} GiB) free on "
            "every GPU; when any GPU has less, `/score` and `/wake` return HTTP 507 with "
            "`gpu_free_bytes` and `wake_need_bytes` and score nothing: stop or shrink "
            "your server and retry.") in flat


def test_the_brief_states_the_scoring_services_startup_and_its_states():
    flat = _flat(_brief()[0])
    assert (f"The service starts with your session and takes about "
            f"{task_mod.REFERENCE_SERVICE_STARTUP_SEC} s to load the reference (one that has not "
            f"loaded within {rs.READY_TIMEOUT_SEC:,.0f} s is stopped), holding up to "
            f"{task_mod.REFERENCE_SERVICE_AWAKE_GIB} GiB on each GPU while it does.") in flat
    assert (f"Until it is asleep, `/status` reports `{rs.STARTING_STATE}`, and `/score`, "
            "`/wake` and `/sleep` return HTTP 503 with `state`.") in flat
    assert (f"Leave the GPUs unused until `/status` reports `{rs.READY_STATE}`: a service "
            "that fails to load is stopped, and its socket then refuses connections for "
            "the rest of the build.") in flat
    # The /status line lists every state the harness module names, and no other.
    line = flat.split("GET /status")[1].split("gpu_free_bytes")[0]
    listed = set(re.findall(r'"(\w+)"', line)) - {"state"}
    assert listed == {rs.STARTING_STATE, rs.FAILED_STATE, rs.UNLOADED_STATE} | rs.READY_STATES


def test_the_brief_states_that_the_service_shares_the_sessions_memory_and_answers_a_shortage():
    # The service's memory is oversubscribed on the sandbox's: the brief says how much it holds
    # asleep and what happens when the two pass it, and that a shortage is an HTTP 507 the agent
    # answers by releasing memory, with the names the service's replies carry.
    flat = _flat(_brief()[0])
    assert ("Your session and the service share the memory stated for your sandbox: asleep the "
            f"service keeps a copy of the weights there, about {task_mod.REFERENCE_SERVICE_ASLEEP_HOST_GIB} GiB, "
            "and when the two together pass that memory, the kernel ends your session's largest "
            "process.") in flat
    assert ("A load, a sleep or a wake that memory is short for changes nothing and returns HTTP 507 "
            "with `memory`, `gpu` or `host`; a load short of GPU memory also returns `gpu_free_bytes` "
            "and `gpu_need_bytes`, and `/status` carries the last shortage as `memory_error`: release "
            "that memory and retry.") in flat
    assert (f"A load short of memory leaves `/status` at `{rs.UNLOADED_STATE}` until `POST /wake` loads "
            "it again.") in flat
    assert '"memory_error": {...} | null' in flat


def test_the_brief_states_the_references_run_to_run_spread():
    flat = _flat(_brief()[0])
    assert ("Scores are bit-reproducible for sequences under "
            f"{task_mod.REFERENCE_SERVICE_EXACT_BELOW_TOKENS:,} tokens. Beyond that the "
            "reference itself varies run to run: at prompt positions by a median of "
            f"{task_mod.REFERENCE_SERVICE_REPEAT_PROMPT_MEDIAN_NATS} nats per token "
            f"({task_mod.REFERENCE_SERVICE_REPEAT_PROMPT_OVER_1_NAT_PCT} percent of positions "
            "by more than 1 nat), at generated positions by a median of 0 and at most "
            f"{task_mod.REFERENCE_SERVICE_REPEAT_CONT_MAX_NATS}; the grader's reference "
            "carries the same variation.") in flat


def test_the_graded_build_runs_the_scoring_service():
    assert task_mod.ARTIFACT.producer.reference_service is True


def test_the_scoring_service_is_stated_only_when_the_build_runs_it(staged):
    # A build without the service has no socket to offer, so its brief must not mention
    # one; the two briefs the stage writes differ by the service's section and nothing else.
    on, off = staged(True), staged(False)
    assert "## Reference Scoring Service" in on
    for absent in ("Reference Scoring Service", rs.SOCKET_ENV_VAR, rs.SOCKET_DIR,
                   "/score", "a service scores"):
        assert absent not in off, absent
    assert "The reference itself cannot be run in your sandbox." in _flat(off)
    assert "The reference itself cannot be run in your sandbox." in _flat(on)
    off_lines, on_lines = off.splitlines(), on.splitlines()
    changes = [op for op in difflib.SequenceMatcher(a=off_lines, b=on_lines,
                                                    autojunk=False).get_opcodes()
               if op[0] != "equal"]
    assert len(changes) == 1 and changes[0][0] == "insert"
    inserted = on_lines[changes[0][3]:changes[0][4]]
    assert inserted[0] == "## Reference Scoring Service"
    assert inserted[-1].endswith("until `POST /wake` loads it again.")


def test_the_brief_does_not_call_the_service_a_sidecar(staged):
    # "Sidecar" names the egress gateway in this harness; the scorer has its own name.
    assert "sidecar" not in staged(True).lower()


def test_the_files_the_brief_names_are_in_the_uploaded_tree():
    """The brief states these files are under ``references/glm53-reference/``, and a contestant
    that looks for them and finds nothing is reading a false spec. Read from the merged view of
    the task's upload folders (``hls.agent.uploads.merged_view``) that this checkout holds: the
    toolchain library is not committed, and the task's own folder, which holds these, is."""
    import pathlib

    from hls.agent.uploads import merged_view

    producer = task_mod.ARTIFACT.producer
    assert producer.upload_dest == "/workspace/references"
    base = pathlib.Path(task_mod.__file__).parent
    folders = [(base / value).resolve() for value in producer.upload_dir]
    view = merged_view([folder for folder in folders if folder.is_dir()])
    named = re.findall(r"`([\w.]+\.(?:md|jsonl|py))`", producer.upload_brief)
    assert set(named) == {"README.md", "prompts.jsonl", "reference.jsonl", "selfcheck.py"}
    for name in sorted(named):
        assert f"glm53-reference/{name}" in view, f"the brief names references/glm53-reference/{name}"


def test_the_workload_fingerprint_is_unchanged():
    # Nothing here is traffic: the brief, the upload text and the producer's package
    # policy sit outside the calibration fingerprint, so the committed calibration and
    # baseline still apply. A change to this constant means a re-calibration, not a
    # test edit.
    from hls.verify.calibration import workload_fingerprint

    assert workload_fingerprint(task_mod.TASK) == (
        "28d71b8571014c05505e2b31beb43fb3d7ea6ec7b24b8e74d90dcf9f0bc17522")
