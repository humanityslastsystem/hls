"""What a task provisions into the content-addressed cache, keyed how, and the role program
that fetches it.

The provision stage (``hls.stages.provision``) runs ``hls.provision.role`` as host-side
harness role steps: one to resolve what the task needs, then one per missing cache entry
(weights, dataset rows, tool bundles). The runner process itself downloads nothing."""
