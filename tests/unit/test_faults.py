from hls.core.faults import (ArtifactFault, ArtifactTransportError,
                             ArtifactProtocolError, HarnessFault, DatasetError)


def test_artifact_faults_share_base_and_have_categories():
    assert issubclass(ArtifactTransportError, ArtifactFault)
    assert issubclass(ArtifactProtocolError, ArtifactFault)
    assert ArtifactFault.category == "artifact"
    assert ArtifactTransportError().category == "transport"
    assert ArtifactProtocolError().category == "protocol"


def test_harness_faults_disjoint_from_artifact_faults():
    assert issubclass(DatasetError, HarnessFault)
    assert not issubclass(HarnessFault, ArtifactFault)
    assert not issubclass(ArtifactFault, HarnessFault)


def test_artifact_fault_catchable_as_base_with_message():
    try:
        raise ArtifactProtocolError("bad body")
    except ArtifactFault as f:
        assert f.category == "protocol"
        assert "bad body" in str(f)
