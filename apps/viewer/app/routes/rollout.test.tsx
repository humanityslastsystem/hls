// The rollout page's header on a store instance's record. The instance holds no rollout.json at its
// root: its record is finalize's output in a stage attempt (or failed/), which the scanner reads
// and serves, so the header links to the record the page shows, not to a path of the layout the
// stage pipeline replaced.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import fixture from "~/lib/fixtures/stage-rollout.json";
import type { RolloutRecord } from "~/lib/types";
import Rollout from "~/routes/rollout";

const rec = { ...fixture, evals: [], instance: `runs/${fixture.task_id}/${fixture.rollout_id}` } as RolloutRecord;

/** The rollout page of *r*, as the browser has it once the record is fetched. */
function page(r: RolloutRecord): string {
  const client = new QueryClient();
  client.setQueryData(["rollout", r.task_id, r.rollout_id], r);
  return renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <NuqsTestingAdapter>
        <MemoryRouter initialEntries={[`/rollouts/${r.task_id}/${r.rollout_id}`]}>
          <Routes>
            <Route path="rollouts/:task/:id" element={<Rollout />} />
          </Routes>
        </MemoryRouter>
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );
}

describe("the rollout page's header", () => {
  const html = page(rec);

  it("links to the record the scanner serves, which every layout has", () => {
    expect(html).toContain(`href="/api/rollouts/${fixture.task_id}/${fixture.rollout_id}"`);
  });

  it("links to no rollout.json at the instance's root, where only a published rollout has one", () => {
    expect(html).not.toContain("files/rollout.json");
  });
});
