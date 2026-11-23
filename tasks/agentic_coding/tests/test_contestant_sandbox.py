"""The contestant's sandbox: what it ships, what it may install, what it may read.

The brief states three things about the environment a contestant builds in: a
toolchain that is installed, a list of serving frameworks that are blocked, and a tree
of offline documentation. These tests hold the task's declarations to what the brief
says.
"""

from __future__ import annotations

from hls.prompt import blocked_packages_of
from hls.recipes.catalog import image_set, task_images
from tests.unit.brief_task import render_brief

from tasks.sample_task import task as sample
from tasks.agentic_coding import task as task_mod


def test_the_sandbox_is_the_submission_base_and_harness_roles_run_on_the_reference_image():
    # The agent builds in the agent cap on the task's own layer, a submission is packaged on
    # that sandbox, and the harness roles run on the task's reference image.
    declared = task_mod.IMAGES
    assert image_set("agentic_coding") == declared
    assert declared.layer == "agentic-coding-tf-5-17-0" and declared.package_on_sandbox
    assert declared.harness_on_reference
    images = task_images("agentic_coding")
    assert images.sandbox == images.submission_base == "agentic-coding-tf-5-17-0-agent"
    assert images.harness == images.reference == images.measured["reference"]
    assert set(images.measured) == {"baseline", "reference", "member-sglang"}


def test_the_blocked_frameworks_are_denied_by_the_producer_policy():
    producer = task_mod.ARTIFACT.producer
    packages = producer.network.packages
    assert packages is not None and packages.mode == "deny"
    assert list(packages.entries) == list(task_mod._DENIED_PACKAGES)
    # Every name the shared serving-framework list denies, this task denies too.
    for name in sample._DENIED_SERVING_FRAMEWORKS:
        assert name in packages.entries, name
    # The baseline's own stack, and the frameworks the brief names, are on it.
    for name in ("vllm", "sglang", "flash-attn", "xformers", "tensorrt-llm"):
        assert name in packages.entries
    assert producer.network.domains.mode == "allow" and not producer.network.domains.entries


def test_the_deny_list_is_the_global_176_name_list():
    # Every task carries one list, src/hls/egress/denied_packages.py, 176 names. A
    # name from each of its ten groups stands for the group; the first and last names
    # pin the order the brief renders.
    names = task_mod._DENIED_PACKAGES
    assert len(names) == 176
    assert names[0] == "vllm" and names[-1] == "mamba-ssm"
    for name in ("ai-dynamo", "tokenspeed-mla", "vllm-omni", "faster-whisper",
                 "torch-tensorrt", "hf-kernels", "flash-attn-4", "marlin",
                 "triton-kernels", "unsloth"):
        assert name in names, name


def test_the_names_a_framework_is_reachable_through_are_denied():
    # Each name reaches a framework by another route: an engine the checkpoint's
    # README advertises, its companion wheels, a framework's kernels published under
    # another name, and the reference kernels for this architecture's linear-attention
    # layers.
    packages = task_mod.ARTIFACT.producer.network.packages
    for name in ("tokenspeed", "tokenspeed-kernel", "tokenspeed-scheduler",
                 "tokenspeed-triton", "tokenspeed-deepgemm", "tokenspeed-flashmla",
                 "tokenspeed-nightly", "tokenspeed-proton", "ktransformers",
                 "sgl-kernel", "sglang-kernel", "fla-core", "flash-linear-attention",
                 "lightllm", "aphrodite-engine", "mlc-llm", "mlc-ai-nightly",
                 "exllamav3", "nano-vllm", "vllm-flash-attn", "flashinfer-cubin",
                 "flashinfer-jit-cache", "kernels", "kernels-community"):
        assert not packages.permits(name, normalize=True), name


def test_the_deny_list_carries_no_name_twice():
    # The policy matches PEP 503-normalized names, so `tensorrt_llm` and `tensorrt-llm`
    # are one entry; a duplicate would render twice in the brief's Blocked line.
    from hls.egress.policy import pep503_normalize

    normalized = [pep503_normalize(n) for n in task_mod._DENIED_PACKAGES]
    assert len(normalized) == len(set(normalized))


def test_a_spelling_the_deny_list_does_not_use_is_still_refused():
    # PEP 503 folds case and `-`/`_`/`.`, so the list needs one spelling per project.
    packages = task_mod.ARTIFACT.producer.network.packages
    for name in ("TokenSpeed", "TokenSpeed.Kernel", "sgl_kernel", "FLA_Core",
                 "flash_linear_attention", "Kernels"):
        assert not packages.permits(name, normalize=True), name


def test_this_architectures_own_layer_kernels_are_denied():
    # Each of these packages is one layer of this checkpoint's hot path, written and
    # tuned: the conv1d and state-space scan, the MLA attention kernel, the FP8 grouped
    # GEMM the MoE layers need, and the fused layer kernels. Installing one is being
    # handed the work the score is for.
    packages = task_mod.ARTIFACT.producer.network.packages
    for name in ("causal-conv1d", "mamba-ssm", "flash-mla", "deep-gemm", "liger-kernel",
                 "Causal_Conv1D", "Deep_GEMM", "Flash_MLA"):
        assert not packages.permits(name, normalize=True), name


def test_a_package_outside_the_rule_is_still_installable():
    # The policy is a deny list, not a whitelist: the build depends on ordinary
    # packages and a broadened list must not start refusing them.
    packages = task_mod.ARTIFACT.producer.network.packages
    # `xgrammar` is here deliberately: constrained decoding is not a layer of this model,
    # so the rule does not reach it.
    for name in ("torch", "transformers", "triton", "numpy", "fastapi", "uvicorn",
                 "pydantic", "safetensors", "nvidia-cutlass-dsl", "xgrammar"):
        assert packages.permits(name, normalize=True), name


def test_the_brief_names_exactly_what_the_policy_denies():
    # The Blocked line is rendered from the producer's deny list, which the agent_build stage
    # threads in, so the brief names every package the proxy refuses.
    entries = blocked_packages_of(task_mod.ARTIFACT.producer.network)
    assert entries == list(task_mod._DENIED_PACKAGES)
    brief, _ = render_brief(task_mod.TASK, task_mod.ARTIFACT.brief, blocked_packages=entries)
    flat = " ".join(brief.split())
    blocked = flat[flat.index("Blocked ("):flat.index("Your server and its kernels")]
    for name in entries:
        assert f"`{name}`" in blocked, name
    # Nothing beyond the list is claimed to be blocked.
    assert "other high-level" not in flat


def test_the_brief_states_the_rule_the_blocked_list_enforces():
    # The list is the names a proxy can refuse; the rule is what a submission is held to.
    # Without the rule, a framework published under an unlisted name reads as permitted.
    brief, prov = render_brief(task_mod.TASK, task_mod.ARTIFACT.brief,
                               blocked_packages=blocked_packages_of(task_mod.ARTIFACT.producer.network))
    flat = " ".join(brief.split())
    assert ("Your server and its kernels are your own work on the installed toolchain: "
            "no third-party LLM inference engine, serving framework, or attention, GEMM, "
            "MoE or linear-attention kernel library may be installed, vendored or copied, "
            "whether or not `pip install` refuses it.") in flat
    assert ("Submissions are audited against this rule; the Blocked list names the ones "
            "already known and does not bound it.") in flat
    # It renders inside Build Instructions, after the list it qualifies.
    build = {s.block: s for s in prov.sections}["build_instructions"]
    assert build.rendered.index("Blocked (") < build.rendered.index("your own work")
    assert build.source == "task"


def test_a_brief_without_a_deny_list_says_nothing_is_blocked():
    brief, _ = render_brief(task_mod.TASK, task_mod.ARTIFACT.brief, blocked_packages=None)
    flat = " ".join(brief.split())
    assert "any package is installable; nothing is blocked" in flat
    assert "Blocked (" not in flat
    # The rule holds whether or not a list was threaded in.
    assert "no third-party LLM inference engine" in flat


def test_the_documentation_the_brief_points_at_is_uploaded():
    producer = task_mod.ARTIFACT.producer
    assert producer.upload_dir == ("../assets/references", "references")
    assert producer.upload_dest == "/workspace/references"
    # The shared toolchain paragraph without its last sentence, which says the uploads are
    # the only reference on disk, plus this checkpoint's own on-disk references.
    assert sample._TOOLCHAIN_UPLOAD_BRIEF.startswith(task_mod._TOOLCHAIN_PARAGRAPH)
    assert "There is no other reference" in sample._TOOLCHAIN_UPLOAD_BRIEF.removeprefix(
        task_mod._TOOLCHAIN_PARAGRAPH)
    assert producer.upload_brief.startswith(task_mod._TOOLCHAIN_PARAGRAPH + "\n\n")
    assert producer.upload_brief == task_mod._UPLOAD_BRIEF
