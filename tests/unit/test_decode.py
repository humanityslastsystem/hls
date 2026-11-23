import pydantic
import pytest

from hls.core.decode import DecodeConfig, Greedy, Sample
from hls.core.poly import dump_spec


def test_greedy_fingerprint_stable():
    assert Greedy().fingerprint() == "greedy"


def test_sample_fingerprint_is_the_temperature():
    assert Sample(temperature=0.7).fingerprint() == "sample-t0.7"
    assert Sample(temperature=0.7).fingerprint() != Sample(temperature=0.8).fingerprint()


def test_sample_defaults_to_temperature_one():
    """The sampling contract the report verifier grades: draw from the model's own
    distribution, with nothing to reproduce on the reference side."""
    assert Sample().temperature == 1.0
    assert Sample().fingerprint() == "sample-t1.0"


def test_a_sampled_decode_carries_no_draw_seed():
    """The draw is left to each stack's own generator. Nothing replays it -- the report
    gate reads the distribution the server reported rather than a reference draw -- and
    sglang only acts on a per-request seed when the engine is built for deterministic
    inference, which swaps its sampling and attention backends. A seeded field would
    therefore fingerprint two decodes apart that no check distinguishes, and tempt a
    pool member onto a different kernel set than its neighbours."""
    assert "seed" not in Sample.model_fields
    # The dumped spec is what reaches the workload fingerprint, so a reintroduced
    # field would move every shipped task's calibration off its measurement.
    assert "seed" not in dump_spec(Sample(temperature=1.0))


def test_zero_temperature_is_refused():
    """Temperature 0 is the argmax, which is ``Greedy``; a Sample that means greedy
    would fingerprint as a sampled decode and be graded by the wrong verifier."""
    with pytest.raises(pydantic.ValidationError):
        Sample(temperature=0.0)


def test_decode_configs_are_decode_config():
    assert isinstance(Greedy(), DecodeConfig)
    assert isinstance(Sample(temperature=0.1), DecodeConfig)


def test_sample_dumps_with_kind():
    rec = dump_spec(Sample(temperature=0.7))
    assert rec == {"temperature": 0.7, "kind": "Sample"}


def test_greedy_dumps_with_kind():
    rec = dump_spec(Greedy())
    assert rec == {"frequency_penalty": 0.0, "kind": "Greedy"}


def test_greedy_defaults_to_no_penalty():
    """Plain greedy is the default, so an existing task's decode is unchanged."""
    assert Greedy().frequency_penalty == 0.0


def test_penalized_is_false_only_at_zero():
    assert Greedy().penalized is False
    assert Greedy(frequency_penalty=0.1).penalized is True


def test_penalty_is_folded_into_the_fingerprint():
    """The penalty changes which token the reference chooses, so it must scope the
    cache -- and a 0.0 penalty must keep the plain-greedy fingerprint."""
    assert Greedy(frequency_penalty=0.0).fingerprint() == "greedy"
    assert Greedy(frequency_penalty=0.1).fingerprint() == "greedy-fp0.1"


def test_negative_penalty_is_refused():
    """A negative frequency penalty rewards repetition; no engine's semantics are
    pinned for it, so the config refuses it."""
    with pytest.raises(pydantic.ValidationError):
        Greedy(frequency_penalty=-0.1)
