"""The package stage's library: a contestant's source bundle into the sealed submission image a
grading serves.

- ``checkout``: the bundle's ``main`` as a plain tree on node-local disk, by a hardened git;
- ``wheels``: the two sealed pip steps that run the contestant's build hooks, and the pins
  between them;
- ``offline``: the offline build of the image from the submission base and the wheels;
- ``integrity``: the check that the built image's system paths are the base's;
- ``result``: ``package.json``, what the stage publishes."""
