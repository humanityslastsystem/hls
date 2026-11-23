"""The images of the pipelines: the bases, the task layers and the agent cap on top of them, and
the images each task module declares for its own servers. ``model`` declares them, ``render``
turns a declaration into an Apptainer ``.def`` and a cache key, ``catalog`` and ``pins`` hold the
shared images and read each task's, and ``inventory`` reads what a built image holds."""
