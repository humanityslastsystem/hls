"""Stage implementations. Each registers itself with ``hls.pipeline.stage.STAGES`` when its
module is imported, and ``hls.stages.registry`` imports every one, as the runner and the
pipelines do. This package imports none of them itself, so a module imported for one stage's
constants loads that stage alone. No role program imports a module of this package: what a role
shares with its stage, its records and their file names among them, lives below the stages."""
