// One build fact as a view shows it: its value formatted, or "absent" with the reason the record
// holds none, inline or as the marker's tooltip where the view is compact.

import type { ReactNode } from "react";

import type { BuildFact } from "~/lib/build";

export function Fact<T>({
  fact,
  format,
  inline = false,
}: {
  fact: BuildFact<T>;
  format: (value: T) => ReactNode;
  inline?: boolean;
}) {
  if ("value" in fact) return <>{format(fact.value)}</>;
  return (
    <span className="italic text-muted-foreground" title={fact.absent}>
      {inline ? `absent: ${fact.absent}` : "absent"}
    </span>
  );
}
