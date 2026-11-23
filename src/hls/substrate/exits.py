"""The exit statuses the harness's own layers around a role's command give the role's step, each a
number of its own.

A role's status passes up through the step wrapper (``hls.substrate.role_exec``) and, for a
sealed container, through the network seal (``transparent_seal.sh``) before the launcher reads
it. Each layer exits with a status of its own when it refuses or fails before it hands over to
what it wraps, and otherwise passes on the status of what it ran. With no two layers sharing a
number, and no harness program beneath them using one of theirs, a status names the layer that
ended the role on its own, whatever record (the wrapper's grant report, the seal's started file)
that layer did or did not write. The seal takes its numbers from here: its builder sets them as
variables in front of the script (``hls.substrate.apptainer.egress.seal_script``). Numbers 96 to
99 are none a shell gives (126, 127, 128 and up for a signal) nor apptainer's own (255), and no
named exit of a harness program uses one (a test reads every ``*_EXIT`` of the package)."""

#: The step wrapper's, when the uid already holds as many tasks on the node as the step's process
#: ceiling (``--nproc``): it refuses before it writes any report, since the step could fork nothing.
PROCESS_CEILING_EXIT = 96
#: The step wrapper's, when the step's grant differs from its declaration: it wrote its report,
#: whose mismatches say how, and ran nothing.
GRANT_MISMATCH_EXIT = 97
#: The seal's, when it refuses to run its payload: a limit on the uid's per-namespace counts it was
#: given (``HLS_SEAL_UCOUNTS``) is malformed, or could not be written and read back in its user
#: namespace. Every attempt would end the same way, so the seal does not try again.
SEAL_REFUSED_EXIT = 98
#: The seal's, when its interceptor never became ready: the payload never started, so the seal
#: tries again, a few times.
SEAL_UNBOUND_EXIT = 99
