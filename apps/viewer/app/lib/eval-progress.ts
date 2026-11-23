// How far a running instance's eval has got, read from the events its deploy_drive stage logs
// (hls.stages.deploy_drive), the stage that serves the artifact and drives it. The eval's other
// stages, verify and score, log no event that marks their progress. tests/viewer/test_store_layout.py
// holds each name below to the stage's emit calls.

/** What the name of each event the stage logs of its own begins with: the first, its process
 * ceiling, is logged before any role starts. */
export const DRIVE_EVENT_PREFIX = "drive.";

/** The event the stage logs once the served artifact answers. */
export const ARTIFACT_READY = "artifact_ready";

/** The event the stage logs when the drive is over, with its outcome. */
export const DRIVE_RESULT = "drive_result";

/** The field of that event which names the outcome (served, not_ready, ...). */
export const DRIVE_OUTCOME_FIELD = "outcome";

/** How far a running eval has got. */
export interface EvalProgress {
  /** "none" before the stage logs an event of its own; "starting" from its first, while the
   * artifact's sandbox is set up and the artifact started; "driving" once the artifact is ready;
   * "driven" once the drive's result is logged, when verification and scoring follow. */
  phase: "none" | "starting" | "driving" | "driven";
  /** The drive's outcome, from its result event; null when it names none or there is no result. */
  outcome: string | null;
}

const OWN = (type: string) =>
  type.startsWith(DRIVE_EVENT_PREFIX) || type === ARTIFACT_READY || type === DRIVE_RESULT;

/** The progress of the eval whose events are *events*, the page's events of the instance's logs:
 * those of the newest attempt of the drive stage (each event names its log as `source`), since a
 * retried attempt serves and drives again. */
export function evalProgress(events: readonly { type: string; data: Record<string, unknown> }[]): EvalProgress {
  const own = events.filter((ev) => OWN(ev.type));
  const newest = own.length > 0 ? own[own.length - 1].data.source : undefined;
  let phase: EvalProgress["phase"] = "none";
  let outcome: string | null = null;
  for (const ev of own) {
    if (ev.data.source !== newest) continue;
    if (ev.type === DRIVE_RESULT) {
      const named = ev.data[DRIVE_OUTCOME_FIELD];
      phase = "driven";
      outcome = typeof named === "string" ? named : null;
    } else if (ev.type === ARTIFACT_READY) {
      if (phase !== "driven") phase = "driving";
    } else if (phase === "none") {
      phase = "starting";
    }
  }
  return { phase, outcome };
}
