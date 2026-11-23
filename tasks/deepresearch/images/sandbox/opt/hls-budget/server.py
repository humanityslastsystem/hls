#!/usr/bin/env python3
"""stdio MCP server exposing the agent's build budget. Pure stdlib; baked into the
sealed agent image at /opt/hls-budget/server.py and launched by Claude Code over stdio.
Config arrives as base64 JSON in env HLS_BUDGET_B64. The agent sees only limit / spent /
remaining per dimension."""
from __future__ import annotations
import base64
import json
import os
import sys

# Dual import: as a baked script the sibling is `meter.py`; as an installed package
# (host unit tests importing `hls.budget.server`) it is `hls.budget.meter`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from meter import BudgetMeter, Prices, TranscriptSource        # baked sibling (sandbox)
except ImportError:  # pragma: no cover
    from hls.budget.meter import BudgetMeter, Prices, TranscriptSource  # installed package (host)

_PROTOCOL = "2024-11-05"
_UNITS = {"cost_usd": ("cost", "usd"), "turns": ("turns", "turns"), "wallclock_sec": ("wallclock", "seconds")}


def _load_cfg() -> dict:
    raw = os.environ.get("HLS_BUDGET_B64")
    if not raw:
        return {}
    try:
        return json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception:
        return {}


def build_response(cfg: dict, spent_cost, spent_turns, spent_wall) -> dict:
    budget = cfg.get("budget", {}) or {}
    spent_by_dim = {"cost_usd": spent_cost, "turns": spent_turns, "wallclock_sec": spent_wall}
    dims: dict = {}
    for field, (name, unit) in _UNITS.items():
        limit = budget.get(field)
        if limit is None:
            continue
        spent = spent_by_dim[field]
        if field == "cost_usd" and (cfg.get("prices") is None or spent is None):
            dims[name] = {"unit": unit, "limit": limit, "spent": None, "remaining": None, "unavailable": True}
        else:
            dims[name] = {"unit": unit, "limit": limit, "spent": spent,
                          "remaining": limit - spent, "over_budget": spent >= limit}
    return {"dimensions": dims, "note": cfg.get("note", "")}


def _get_budget(cfg: dict) -> dict:
    prices = cfg.get("prices")
    meter = BudgetMeter(
        start_epoch=float(cfg.get("start_epoch", 0.0)),
        prices=Prices(**prices) if prices else None,
        source=TranscriptSource(cfg.get("transcript_dir", "")),
    )
    try:
        s = meter.spent()
        return build_response(cfg, s.cost_usd, s.turns, s.wallclock_sec)
    except Exception as exc:  # a dead budget tool must never fail the build
        return {"dimensions": {}, "note": cfg.get("note", ""), "error": str(exc)[:200]}


def _handle(msg: dict, cfg: dict) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    is_notification = "id" not in msg
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": _PROTOCOL, "capabilities": {"tools": {}},
            "serverInfo": {"name": "budget", "version": "1.0.0"}}}
    if method == "notifications/initialized":
        return None
    if is_notification:
        # JSON-RPC 2.0: a message with no "id" is a notification -- the server
        # MUST NOT reply, even if the method is unrecognized (e.g. Claude Code's
        # notifications/cancelled, notifications/roots/list_changed).
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
            "name": "get_budget",
            "description": "Return your build budget: limit, spent, and remaining for cost, "
                           "turns, and wall-clock time. Consult it to pace your work.",
            "inputSchema": {"type": "object", "properties": {}}}]}}
    if method == "tools/call" and (msg.get("params") or {}).get("name") == "get_budget":
        try:
            text = json.dumps(_get_budget(cfg), indent=2)
        except Exception as exc:  # a dead budget tool must never fail the build
            text = json.dumps({"dimensions": {}, "note": cfg.get("note", ""),
                               "error": str(exc)[:200]}, indent=2)
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}]}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}}


def main() -> None:
    cfg = _load_cfg()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(msg, cfg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
