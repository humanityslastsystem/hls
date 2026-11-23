"""The per-turn output cap the task declares, in a module that imports nothing.

``task.py`` builds the workload from it, and the self-test the build sandbox carries
(``selftest.py``), which runs without the harness, stamps the same cap on the requests it sends."""

#: Per-turn output cap in tokens. A sampled decode has no greedy runaways to bound: on
#: this model 0.2 percent of sampled replies reach the cap, so it binds only the genuine
#: long tail, and a turn that needs to think at length can.
MAX_NEW_TOKENS = 8192
