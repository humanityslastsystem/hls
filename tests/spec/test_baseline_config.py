from hls.spec.baseline import Baseline


def test_baseline_defaults_serve_command_empty():
    b = Baseline(image="docker:vllm/vllm-openai:latest")
    assert b.image == "docker:vllm/vllm-openai:latest" and b.serve_command == []


def test_taskspec_baseline_defaults_none_and_roundtrips(sample_taskspec_with_baseline):
    ts = sample_taskspec_with_baseline
    assert ts.baseline is not None and ts.baseline.image
    # baseline is pure config: it must appear in the provenance dump
    assert "baseline" in ts.model_dump()
