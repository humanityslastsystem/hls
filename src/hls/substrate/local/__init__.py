"""The local substrate's scheduler: allocations on one host with no scheduler.

``LocalScheduler`` (``scheduler``) implements ``hls.substrate.base.Scheduler`` with a record per
job in a state directory (``records``) and a supervising process per job (``supervisor``), which
waits for the job's dependency and for free resources (``schedule``), holds the job's GPUs by
UUID, whole cores and memory in the reservation file (``placement``), runs the job's batch
script, and stops it at the job's end time. ``inventory`` reads what the host holds."""
