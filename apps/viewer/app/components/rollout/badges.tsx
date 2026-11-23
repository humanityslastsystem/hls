// Reusable Ink-styled badge pills for rollout status, gate, and backend.
// Task 8+ imports these directly; do not add color tokens outside CSS vars.

import { Check, Cloud, Container, Minus, Server, X } from "lucide-react";

import { cn } from "~/lib/utils";

const pill =
  "inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-mono font-medium border-[1.5px] border-current shadow-[var(--shadow-xs)]";

export function StatusBadge({ status }: { status: string | null | undefined }) {
  if (!status)
    return (
      <span className="text-muted-foreground font-mono text-xs">—</span>
    );

  if (status === "ok") {
    return (
      <span className={cn(pill, "bg-ok/15 text-ok")}>ok</span>
    );
  }
  if (status === "error") {
    return (
      <span className={cn(pill, "bg-destructive/15 text-destructive")}>
        error
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className={cn(pill, "bg-primary/10 text-primary")}>
        <span className="h-1.5 w-1.5 bg-current animate-pulse" />
        running
      </span>
    );
  }
  if (status === "invalid_artifact") {
    return (
      <span className={cn(pill, "bg-muted text-muted-foreground")}>
        invalid
      </span>
    );
  }
  if (status === "stalled") {
    return (
      <span
        className={cn(pill, "border-amber-500/70 text-amber-600 dark:text-amber-400")}
      >
        stalled
      </span>
    );
  }
  return (
    <span className={cn(pill, "bg-muted text-muted-foreground")}>
      {status}
    </span>
  );
}

export function GateBadge({
  passed,
}: {
  passed: boolean | null | undefined;
}) {
  if (passed == null) {
    return (
      <span className={cn(pill, "text-muted-foreground")}>
        <Minus className="h-3 w-3" /> n/a
      </span>
    );
  }
  if (passed) {
    return (
      <span className={cn(pill, "bg-ok/15 text-ok")}>
        <Check className="h-3 w-3" /> pass
      </span>
    );
  }
  return (
    <span className={cn(pill, "bg-destructive/15 text-destructive")}>
      <X className="h-3 w-3" /> fail
    </span>
  );
}

export function BackendBadge({
  backend,
}: {
  backend: string | null | undefined;
}) {
  if (!backend) return <span className="text-muted-foreground">—</span>;
  const Icon = backend === "modal" ? Cloud : backend === "apptainer" ? Server : Container;
  return (
    <span className={cn(pill, "bg-primary/10 text-primary")}>
      <Icon className="h-3 w-3" /> {backend}
    </span>
  );
}

export function LiveBadge() {
  return (
    <span className={cn(pill, "bg-primary/10 text-primary")}>
      <span className="h-1.5 w-1.5 bg-current animate-pulse" />
      live
    </span>
  );
}
