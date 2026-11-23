// score.breakdown.slo reader — mirrors Scorer.score (src/hls/scoring.py) and the
// SLO.apply diagnostics it stores (src/hls/spec/workload.py).
//
// Scorer.score ALWAYS nests breakdown["slo"] per tenant: {tenant_id: diagnostics},
// single- and multi-tenant alike (a single-tenant run is just {"t0": {...}}). The
// diagnostics shape depends on which SLO produced it:
//   - StreamingLatencySLO: {p99_ttft_ms, p99_tpot_ms, ttft_ms, tpot_ms, n_met,
//     n_total, met_frac, has_timing} — no "p99_ms"/"target_ms" keys exist.
//   - NoSLO: {n} — no latency objective declared, nothing to headline.
// Older records (scored before per-tenant nesting landed, back when the now-
// removed P99Latency class wrote a flat {p99_ms, target_ms, n} dict directly
// under breakdown.slo) are detected by the presence of "p99_ms"/"target_ms" and
// treated as the one implicit run.

export interface SloSummary {
  /** Tenant this summary was read from; null for the legacy flat shape. */
  tenantId: string | null;
  /** Which diagnostic is headlined. */
  metric: "ttft" | "legacy" | "none";
  /** Display label for the headline p99 stat. */
  label: string;
  /** Headline p99 latency in ms (TTFT for streaming, raw value for legacy). */
  p99Ms: number | null;
  /** Target/budget in ms corresponding to p99Ms. */
  targetMs: number | null;
  /** p99 mean time-per-output-token in ms, streaming only. */
  tpotP99Ms: number | null;
  /** TPOT budget in ms, streaming only. */
  tpotTargetMs: number | null;
  /** Request count (NoSLO's n, streaming's n_total, or legacy's n). */
  n: number | null;
}

function num(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}

function sloSection(score: unknown): Record<string, unknown> | null {
  if (!score || typeof score !== "object") return null;
  const breakdown = (score as Record<string, unknown>).breakdown;
  if (!breakdown || typeof breakdown !== "object") return null;
  const slo = (breakdown as Record<string, unknown>).slo;
  if (!slo || typeof slo !== "object") return null;
  return slo as Record<string, unknown>;
}

function summarizeTenantDiag(tenantId: string, d: Record<string, unknown>): SloSummary {
  const ttftP99 = num(d.p99_ttft_ms);
  if (ttftP99 != null) {
    return {
      tenantId,
      metric: "ttft",
      label: "p99 TTFT",
      p99Ms: ttftP99,
      targetMs: num(d.ttft_ms),
      tpotP99Ms: num(d.p99_tpot_ms),
      tpotTargetMs: num(d.tpot_ms),
      n: num(d.n_total),
    };
  }
  // NoSLO (or an unrecognized diagnostics shape): nothing to headline.
  return {
    tenantId,
    metric: "none",
    label: "SLO",
    p99Ms: null,
    targetMs: null,
    tpotP99Ms: null,
    tpotTargetMs: null,
    n: num(d.n),
  };
}

/**
 * Read a normalized SLO summary off an eval's `score` object (the value of
 * `EvalEpisode.score`). Returns null when there's no SLO section at all.
 *
 * For the current per-tenant shape, picks the worst-case tenant (max TTFT p99)
 * to headline single-value displays (Overview stat tile, Deploy panel, Compare
 * column) — for the common single-tenant case this is simply that one entry.
 */
export function summarizeSlo(score: unknown): SloSummary | null {
  const section = sloSection(score);
  if (!section) return null;

  if ("p99_ms" in section || "target_ms" in section) {
    return {
      tenantId: null,
      metric: "legacy",
      label: "p99 latency",
      p99Ms: num(section.p99_ms),
      targetMs: num(section.target_ms),
      tpotP99Ms: null,
      tpotTargetMs: null,
      n: num(section.n),
    };
  }

  let worst: SloSummary | null = null;
  for (const [tenantId, raw] of Object.entries(section)) {
    const d = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
    const summary = summarizeTenantDiag(tenantId, d);
    if (!worst || (summary.p99Ms ?? -Infinity) > (worst.p99Ms ?? -Infinity)) {
      worst = summary;
    }
  }
  return worst;
}
