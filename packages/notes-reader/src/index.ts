/**
 * Reader for the graph-memory dated-notes export (`graph_memory.notes`).
 *
 * The format is tab-separated with a fixed header; empty date cells mean "open" or "unknown".
 */

export const HEADER = "# graph-memory notes v1";
export const COLUMNS = [
  "valid_from",
  "valid_to",
  "subject",
  "relation",
  "object",
  "recorded_at",
  "retracted_at",
] as const;

export interface Note {
  validFrom: string | null;
  validTo: string | null;
  subject: string;
  relation: string;
  object: string;
  recordedAt: string;
  retractedAt: string | null;
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

function dateCell(value: string, column: string, line: number): string | null {
  if (value === "") return null;
  if (!ISO_DATE.test(value)) {
    throw new Error(`line ${line}: ${column} is not an ISO date: ${value}`);
  }
  return value;
}

export function parseNotes(text: string): Note[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines[0] !== HEADER) throw new Error("not a graph-memory notes v1 file");
  if (lines[1] !== COLUMNS.join("\t")) throw new Error("unexpected columns");
  return lines.slice(2).map((raw, i) => {
    const line = i + 3;
    const cells = raw.split("\t");
    if (cells.length !== COLUMNS.length) {
      throw new Error(`line ${line}: expected ${COLUMNS.length} cells, found ${cells.length}`);
    }
    const [vf, vt, subject, relation, object, rec, ret] = cells;
    const recordedAt = dateCell(rec, "recorded_at", line);
    if (recordedAt === null) throw new Error(`line ${line}: recorded_at is required`);
    return {
      validFrom: dateCell(vf, "valid_from", line),
      validTo: dateCell(vt, "valid_to", line),
      subject,
      relation,
      object,
      recordedAt,
      retractedAt: dateCell(ret, "retracted_at", line),
    };
  });
}

/** Notes that held on `day` (valid time) and that the agent still believes. */
export function currentAt(notes: Note[], day: string): Note[] {
  return notes.filter(
    (n) =>
      n.retractedAt === null &&
      (n.validFrom === null || n.validFrom <= day) &&
      (n.validTo === null || day < n.validTo),
  );
}

/** What the agent believed on `day` (recorded time), whatever it has learned since. */
export function believedOn(notes: Note[], day: string): Note[] {
  return notes.filter((n) => n.recordedAt <= day && (n.retractedAt === null || day < n.retractedAt));
}
