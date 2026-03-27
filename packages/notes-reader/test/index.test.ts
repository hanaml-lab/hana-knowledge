import { strict as assert } from "node:assert";
import { test } from "node:test";

import { believedOn, currentAt, parseNotes } from "../src/index.ts";

const NOTES = [
  "# graph-memory notes v1",
  "valid_from\tvalid_to\tsubject\trelation\tobject\trecorded_at\tretracted_at",
  "2025-06-01\t2026-02-01\tu42\tlives_in\tManchester\t2025-06-03\t",
  "2026-03-01\t\tu42\tlives_in\tLeeds\t2026-04-02\t2026-04-09",
  "2026-02-01\t\tu42\tlives_in\tLeeds\t2026-04-09\t",
  "",
].join("\n");

test("parses rows with open cells", () => {
  const notes = parseNotes(NOTES);
  assert.equal(notes.length, 3);
  assert.equal(notes[2].validTo, null);
  assert.equal(notes[1].retractedAt, "2026-04-09");
});

test("valid time and recorded time answer different questions", () => {
  const notes = parseNotes(NOTES);
  assert.deepEqual(currentAt(notes, "2026-02-15").map((n) => n.object), ["Leeds"]);
  assert.deepEqual(currentAt(notes, "2025-12-01").map((n) => n.object), ["Manchester"]);
  const april5 = believedOn(notes, "2026-04-05").filter((n) => n.object === "Leeds");
  assert.equal(april5[0].validFrom, "2026-03-01");
});

test("rejects other files", () => {
  assert.throws(() => parseNotes("a,b,c"), /not a graph-memory notes/);
  assert.throws(() => parseNotes(NOTES.replace("2025-06-03", "June")), /not an ISO date/);
});
