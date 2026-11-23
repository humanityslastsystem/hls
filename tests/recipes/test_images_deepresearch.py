"""deepresearch's images warmer plans every image of the task, each built from its recipe. What
each recipe holds is tested beside the task, in ``tasks/deepresearch/tests/test_images.py``."""

from __future__ import annotations

from hls.pipeline import commands
from hls.recipes import catalog
from hls.stages.images import image_keys
from hls.substrate.apptainer.images import ApptainerImages
from tests.provision.warmer_config import write_config

#: The task's images in the order the stage fills them, each after the image it builds on.
IMAGES = ["cuda", "deepresearch-sandbox", "deepresearch-sandbox-agent", "deepresearch-baseline",
          "deepresearch-reference-setuptools", "deepresearch-reference", "deepresearch-member-sglang",
          "deepresearch-member-vllm"]


def test_the_warmer_plans_every_image_of_the_task_in_build_order(tmp_path, login):
    plan = commands.plan_images_build(config=write_config(tmp_path), overrides={}, task="deepresearch",
                                      job_name="hls-images", echo=[].append)
    stage = plan.stage("images")
    assert stage.params["catalog_task"] == "deepresearch"
    declared = stage.params["images"]
    assert [d["name"] for d in declared] == IMAGES
    assert {d["kind"] for d in declared} == {"base", "layer", "cap"}
    # A cold cache holds none, so the stage is shaped to fill them all.
    keys = image_keys(catalog.task_images("deepresearch").decls, ApptainerImages.recipe_key)
    assert list(keys) == IMAGES and stage.params["found"] == []
    assert stage.shape.gpus == 0 and stage.shape.disk_gib > 0
