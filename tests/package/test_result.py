"""``package.json``: what the package stage writes (``PackageResult``) and what a grading reads of it
(``PackageOutcome``), so that a record whose unread fields are missing or unknown still grades."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hls.package.result import PackageOutcome, PackageResult
from hls.util import validation_errors

BUILT = PackageResult(task="sample", status="built", bundle_sha256="e" * 64, commit="f" * 40, image="k/submission.sif",
                      sif_sha256="a" * 64, wheels=["numpy==2.1.0"], served_ucounts=None)
INVALID = PackageResult(task="sample", status="invalid_artifact", bundle_sha256=None, commit=None,
                        reason="pyproject.toml is missing", returncode=1, log_tail="error: no project",
                        served_ucounts=None)


def _without(record: PackageResult, *names: str) -> bytes:
    """*record*'s ``package.json`` without the fields *names*, as a package stage that did not
    write them would have."""
    return json.dumps({k: v for k, v in record.model_dump(mode="json").items() if k not in names}).encode()


@pytest.mark.parametrize("record", [BUILT, INVALID], ids=["built", "invalid"])
def test_a_grading_reads_the_fields_it_grades_of_a_record_another_package_stage_wrote(record):
    """The fields a grading does not read may be missing or unknown: the record grades all the
    same."""
    older = PackageOutcome.model_validate_json(_without(record, "task", "bundle_sha256", "commit", "wheels",
                                                        "served_ucounts"))
    assert older == PackageOutcome.model_validate(record.model_dump())
    assert (older.status, older.image, older.sif_sha256, older.reason, older.returncode, older.log_tail) == (
        record.status, record.image, record.sif_sha256, record.reason, record.returncode, record.log_tail)
    newer = json.loads(record.model_dump_json()) | {"added_since": 1}
    assert PackageOutcome.model_validate(newer) == older


def test_a_record_without_its_status_is_refused_naming_it():
    with pytest.raises(ValidationError) as e:
        PackageOutcome.model_validate_json(_without(BUILT, "status"))
    assert validation_errors(e.value) == "missing status (required by this version of hls)"


@pytest.mark.parametrize("changes, error", [
    ({"image": None}, "image_set_exactly_when_built"),
    ({"reason": "it built and failed"}, "reason_set_exactly_when_invalid"),
], ids=["built-without-its-image", "built-with-a-reason"])
def test_a_record_whose_status_and_fields_disagree_is_refused_by_reader_and_writer(changes, error):
    """The one consistency check the reader and the writer share: a built package names its image
    and digest and gives no reason, an invalid one the reverse."""
    data = BUILT.model_dump(mode="json") | changes
    for model in (PackageOutcome, PackageResult):
        with pytest.raises(ValidationError) as e:
            model.model_validate(data)
        assert validation_errors(e.value) == f"{error} at the top level"
    invalid = INVALID.model_dump(mode="json") | {"image": "k/submission.sif", "sif_sha256": "a" * 64}
    with pytest.raises(ValidationError, match="image_set_exactly_when_built"):
        PackageOutcome.model_validate(invalid)


def test_the_writer_refuses_a_field_it_does_not_know_and_a_built_package_without_its_bundle():
    with pytest.raises(ValidationError) as e:
        PackageResult.model_validate(BUILT.model_dump(mode="json") | {"added_since": 1})
    assert validation_errors(e.value) == "extra_forbidden added_since"
    with pytest.raises(ValidationError) as e:
        PackageResult.model_validate(BUILT.model_dump(mode="json") | {"bundle_sha256": None})
    assert validation_errors(e.value) == "built_without_its_bundle at the top level"
