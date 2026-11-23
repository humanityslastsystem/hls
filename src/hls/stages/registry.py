"""Importing this module registers every stage implementation with
``hls.pipeline.stage.STAGES``: what the runner imports to find a stage by its kind, and the
pipelines to check their stages when a plan is made."""

from __future__ import annotations

from hls.stages import agent_build as agent_build
from hls.stages import baseline_warm as baseline_warm
from hls.stages import calibration as calibration
from hls.stages import credit as credit
from hls.stages import deploy_drive as deploy_drive
from hls.stages import export as export
from hls.stages import finalize as finalize
from hls.stages import images as images
from hls.stages import noop as noop
from hls.stages import oracle_fill as oracle_fill
from hls.stages import package as package
from hls.stages import provision as provision
from hls.stages import sandbox_images as sandbox_images
from hls.stages import score as score
from hls.stages import source_scan as source_scan
from hls.stages import verify as verify
