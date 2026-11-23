"""A package warmer instance as ``hls package`` publishes it: the source_scan stage's record of the
bundle, then a built ``package.json``, the ``submission.sif`` it names and the package's record of
what the install added, under ``warmers/<task>/<id>``."""

from __future__ import annotations

from hls.package.result import PackageResult
from hls.substrate.secret_forms import FORMS
from hls.pipeline.pipelines import package_shape
from hls.spec.resources import GrantedResources
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref, publish_manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan
from tests.evaluate.fixture import TASK_ID

#: The warmer's id.
WARMER = "20261002T000000Z-abcdef"
#: The writable disk cap of the build whose bundle the warmer packages, in GiB: that of most
#: tasks' builds, which sizes the package stage's declared shape (``package_shape``).
BUILD_DISK_CAP_GIB = 180


def publish_stage(store: PosixStore, layout: InstanceLayout, stage: str, outputs: dict[str, str], *,
                  inputs: dict[str, str] | None = None, images: dict[str, str] | None = None) -> str:
    """Publish *stage*'s manifest over *outputs* (name to key) and return the manifest's digest."""
    shape = package_shape(BUILD_DISK_CAP_GIB)
    return publish_manifest(store, layout, Manifest(
        instance=layout.root, stage=stage, attempt=1,
        outputs={name: output_ref(store, key, sha256_bytes(store.read_bytes(key))) for name, key in outputs.items()},
        inputs=inputs or {},
        provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                              image_digests=images or {}, config_sha256="d" * 64),
        resources=Resources(declared=shape, granted=GrantedResources(gpus=0, cpus=8, memory_gib=36.0,
                                                                     disk_free_gib=300.0)),
        times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=3.0, ended_at=4.0)))


def publish_package(store: PosixStore, image: bytes, *, task: str = TASK_ID, warmer: str = WARMER,
                    layout: InstanceLayout | None = None) -> str:
    """Publish the source_scan and package stages that built *image*, each scan finding nothing, in
    the instance *layout* names, warmer *warmer* of *task* by default, and return the instance."""
    layout = layout if layout is not None else InstanceLayout.warmer(task, warmer)
    scanned = layout.attempt_file("source_scan", 1, f"outputs/{PACKAGE_SCAN_FILE}")
    store.write_bytes(scanned, PackageScan(secrets=[], findings=[], read=["bundle", "tree"],
                                                 forms=list(FORMS)).model_dump_json().encode())
    digest = publish_stage(store, layout, "source_scan", {"package_scan": scanned})
    installed = layout.attempt_file("package", 1, f"outputs/{PACKAGE_SCAN_FILE}")
    store.write_bytes(installed, PackageScan(secrets=[], findings=[], read=["install"],
                                                  forms=list(FORMS)).model_dump_json().encode())
    sif = layout.attempt_file("package", 1, "outputs/submission.sif")
    store.write_bytes(sif, image)
    result = PackageResult(task=task, status="built", bundle_sha256="e" * 64, commit="f" * 40, image=sif,
                           sif_sha256=sha256_bytes(image), served_ucounts=None)
    record = layout.attempt_file("package", 1, "outputs/package.json")
    store.write_bytes(record, result.model_dump_json(indent=2).encode())
    publish_stage(store, layout, "package", {"package": record, "submission": sif, "package_scan": installed},
                  inputs={layout.manifest("source_scan"): digest}, images={"submission": result.sif_sha256})
    return layout.root
