// Expanded content for one ATIF trajectory step: message, reasoning, tool
// calls (name + args), observations, and a metrics footer — each shown only
// when present. Ink styling: square corners, hairlines, mono for data.
//
// Harbor's ATIF conversion appends marker-led chunks to a tool observation's
// text: the model-facing result, then [stdout], [stderr], [exit_code],
// [interrupted], [is_image], [metadata] {json}, [error]. stdout/stderr/metadata
// can be large, so they collapse behind a disclosure; the short, important
// signals (exit_code, interrupted, is_image, error) stay inline.

import type { ReactNode } from "react";

import type { BuildFact } from "~/lib/build";
import type { ObservationContent, Step } from "~/lib/types";
import { cn } from "~/lib/utils";
import { ContentRenderer } from "./content-renderer";

type ObsSeg = { kind: string; text: string };

function parseObservation(content: ObservationContent): {
  body: ObservationContent;
  segs: ObsSeg[];
} {
  if (typeof content !== "string") return { body: content, segs: [] };
  const re =
    /(^|\n)\[(stdout|stderr|exit_code|interrupted|is_image|metadata|error)\]/g;
  const marks: { idx: number; start: number; kind: string }[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) !== null) {
    marks.push({ idx: m.index, start: m.index + m[1].length, kind: m[2] });
  }
  if (marks.length === 0) return { body: content, segs: [] };
  const body = content.slice(0, marks[0].idx).replace(/\n+$/, "");
  const segs: ObsSeg[] = marks.map((mk, i) => {
    const end = i + 1 < marks.length ? marks[i + 1].idx : content.length;
    const raw = content.slice(mk.start + `[${mk.kind}]`.length, end);
    return { kind: mk.kind, text: raw.replace(/^[ \n]/, "").replace(/\n+$/, "") };
  });
  return { body, segs };
}

function prettyJson(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s), null, 2);
  } catch {
    return s;
  }
}

function Pre({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <pre
      className={cn(
        "max-h-80 overflow-auto whitespace-pre-wrap border border-border bg-muted/40 px-3 py-2 font-mono text-[11px] leading-5 text-foreground",
        className,
      )}
    >
      {children}
    </pre>
  );
}

function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="mt-1 border border-dashed border-border bg-muted/20">
      <summary className="cursor-pointer select-none px-2 py-1 font-mono text-[11px] font-medium text-muted-foreground">
        {label}
      </summary>
      <div className="p-2 pt-0">{children}</div>
    </details>
  );
}

function ObservationSegment({ seg }: { seg: ObsSeg }) {
  if (seg.kind === "metadata") {
    return (
      <Disclosure label="tool result metadata">
        <Pre>{prettyJson(seg.text)}</Pre>
      </Disclosure>
    );
  }
  if (seg.kind === "stdout" || seg.kind === "stderr") {
    return (
      <Disclosure label={seg.kind}>
        <Pre>{seg.text}</Pre>
      </Disclosure>
    );
  }
  const isError =
    seg.kind === "error" || (seg.kind === "exit_code" && seg.text.trim() !== "0");
  const label = seg.kind === "exit_code" ? "exit code" : seg.kind;
  return (
    <div
      className={cn(
        "mt-1 font-mono text-[11px]",
        isError ? "text-destructive" : "text-muted-foreground",
      )}
    >
      {label}
      {seg.text ? `: ${seg.text}` : ""}
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
      {children}
    </div>
  );
}

export function StepContent({
  step,
  imageBase,
}: {
  step: Step;
  imageBase: BuildFact<string>;
}) {
  const hasMessage =
    step.message != null &&
    (typeof step.message !== "string" || step.message.trim() !== "");

  return (
    <div className="space-y-3 px-3 py-3">
      {hasMessage && (
        <ContentRenderer content={step.message} imageBase={imageBase} />
      )}

      {step.reasoning_content && (
        <div>
          <SectionLabel>Reasoning</SectionLabel>
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap border-l-2 border-border bg-muted/30 px-3 py-2 font-mono text-[11px] leading-5 italic text-muted-foreground">
            {step.reasoning_content}
          </pre>
        </div>
      )}

      {step.tool_calls && step.tool_calls.length > 0 && (
        <div>
          <SectionLabel>Tool calls</SectionLabel>
          <div className="space-y-2">
            {step.tool_calls.map((tc) => (
              <div key={tc.tool_call_id} className="border border-border bg-card">
                <div className="border-b border-border px-2 py-1 font-mono text-[11px] font-semibold text-foreground">
                  {tc.function_name}
                </div>
                <Pre className="max-h-64 border-0">
                  {JSON.stringify(tc.arguments, null, 2)}
                </Pre>
              </div>
            ))}
          </div>
        </div>
      )}

      {step.observation && step.observation.results.length > 0 && (
        <div>
          <SectionLabel>Observations</SectionLabel>
          {step.observation.results.map((result, idx) => {
            const { body, segs } = parseObservation(result.content);
            const showBody = typeof body !== "string" || body.trim() !== "";
            return (
              <div key={idx} className="mb-2">
                {showBody && (
                  <ContentRenderer content={body} imageBase={imageBase} />
                )}
                {segs.map((seg, j) => (
                  <ObservationSegment key={j} seg={seg} />
                ))}
              </div>
            );
          })}
        </div>
      )}

      {step.metrics && (
        <div className="border-t border-border pt-2 font-mono text-[11px] tabular-nums text-muted-foreground">
          {(step.metrics.prompt_tokens ?? 0).toLocaleString()} prompt /{" "}
          {(step.metrics.completion_tokens ?? 0).toLocaleString()} completion
          {step.metrics.cached_tokens != null &&
            ` / ${step.metrics.cached_tokens.toLocaleString()} cached`}
          {step.metrics.cost_usd != null &&
            ` / $${step.metrics.cost_usd.toFixed(4)}`}
        </div>
      )}
    </div>
  );
}
