"""The probe's sealed checks as a test's sandbox runs them: it has no seal to reach a route, an
unlisted host or the package index through, nor one that drops its ICMP or hides the ports its
commands bind, so each check runs a command that answers as a working seal, or a broken one,
would, and the inbound check (``probe.inbound_check``) answers as a working seal's does."""

from __future__ import annotations

import dataclasses

from hls.agent import probe

#: What the ICMP check prints through a working seal, and through one that let an echo out.
ICMP_HELD = """echo '{"icmp": {"passed": true, "sec": 0.0, "detail": "no echo left: stand-in"}}'"""
ICMP_LEFT = """echo '{"icmp": {"passed": false, "sec": 0.0, "detail": "a ping echo left the seal"}}'"""
#: The in-sandbox checks of what a nested namespace provides, which report not ok in a test's
#: sandbox, run without one. ``cap-eff`` is not here: an unprivileged host already has no
#: effective capabilities, so it passes without a namespace too.
NAMESPACE_CHECKS = ("ps-isolation", "dev-shm", "control-covered", "control-socket", "limits-not-inherited",
                    "ucount-ceilings", "pid1-recovery")
#: The host-side checks that read the sandbox step's own processes, which report not ok outside a
#: real step.
STEP_CHECKS = ("spawner-oom-score", "nspid-depth", "reaper-no-socket", "spawner-limits")
#: What each sealed check answers through a working seal.
SEAL_ANSWERS = {"route-reachable": "echo ok TLSv1.3", "unlisted-refused": "echo refused", "icmp-held": ICMP_HELD,
                "index-reachable": "true", "nproc-accounting": "echo forks=0"}


async def held_inbound(exec_, inputs, **_) -> probe.ProbeCheck:
    """The inbound check as through a working seal: every knock refused, nothing reached."""
    return probe.ProbeCheck(name=probe.INBOUND_CHECK, ok=True, observed="stand-in: every knock refused",
                            expected="no port a sandbox command binds is reached from any address of the node")


def answering(answers: dict[str, str], monkeypatch) -> None:
    """Make each of the probe's checks named in *answers* run that one command instead, the
    accounting check the seal probe runs (``probe.ACCOUNTING_CHECK``) among them: a test's sandbox
    runs on this machine's kernel, which need not count as the test's site says."""
    monkeypatch.setattr(probe, "SANDBOX_CHECKS", tuple(
        dataclasses.replace(c, commands=lambda inputs, a=answers[c.name]: (a,)) if c.name in answers else c
        for c in probe.SANDBOX_CHECKS))
    accounting = probe.ACCOUNTING_CHECK
    if accounting.name in answers:
        monkeypatch.setattr(probe, "ACCOUNTING_CHECK", dataclasses.replace(
            accounting, commands=lambda inputs, a=answers[accounting.name]: (a,)))
