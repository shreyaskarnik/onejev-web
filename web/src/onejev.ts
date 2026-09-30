/**
 * OneJev (OmniJev) System One decisions with Transformers.js.
 *
 * A port of qev's prompt.py / answers.py: each question is its own chat turn
 *   system: SYSTEM_PROMPT
 *   user:   <state>\n{state JSON, images inline where "<image:N>" stands}\n</state>\n\n
 *           Question: … Options: A. name: description … Answer with one letter: A, B, C.
 * and the answer is the softmax of the next-token logits of the option letters.
 */

export type Criteria = Record<string, string | null>;
export type Question =
  | { type: "choice"; instructions: string; criteria: Criteria }
  | { type: "score"; instructions: string; criteria: string[] }
  | { type: "noul"; instructions: string; criteria?: { true?: string; false?: string } };

export type Answer =
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: "score"; score: number; probabilities: Record<string, number>; confidence: number; legend: string[] }
  | { type: "noul"; noul: number };

export const SYSTEM_PROMPT =
  "Apply the question to the state. Choose exactly one of the listed options. " +
  "Respond with only its uppercase letter, with no explanation or reasoning.";
export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes";
const NOUL_DEFAULT_FALSE = "the statement is false / the answer is no";

/** Python's json.dumps(value, ensure_ascii=False, indent=2). */
export function pyJson(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? "null";
}

const entry = (value: unknown) => (value == null ? "" : typeof value === "string" ? value.trim() : pyJson(value));

export function renderQuestion(q: Question): { suffix: string; labels: string[] } {
  let kind: string, instructions: string, labels: string[], names: string[], descs: unknown[];
  if (q.type === "noul") {
    kind = "noul";
    instructions = entry(q.instructions) || "Is the statement true, or is the answer to the question yes?";
    labels = names = ["yes", "no"];
    descs = [q.criteria?.true ?? NOUL_DEFAULT_TRUE, q.criteria?.false ?? NOUL_DEFAULT_FALSE];
  } else if (q.type === "choice") {
    kind = "choice";
    instructions = entry(q.instructions) || "Which option applies to the state?";
    labels = names = Object.keys(q.criteria);
    descs = labels.map((k) => q.criteria[k]);
  } else {
    kind = "score";
    instructions = entry(q.instructions) || "Which level describes the state?";
    labels = q.criteria.map((_, i) => String(i));
    names = q.criteria.map((_, i) => `level ${i}`);
    descs = q.criteria;
  }
  if (labels.length > LETTERS.length) throw new Error("This demo supports up to 26 options per question.");
  const letters = LETTERS.slice(0, labels.length).split("");
  const options = letters
    .map((l, i) => {
      const text = entry(descs[i]);
      return text ? `${l}. ${names[i]}: ${text}` : `${l}. ${names[i]}`;
    })
    .join("\n");
  const header = kind === "score" ? "Rate the state:" : "Question:";
  const suffix = `${header} ${instructions}\n\nOptions:\n${options}\n\nAnswer with one letter: ${letters.join(", ")}.`;
  return { suffix, labels };
}

export function stateBlock(state: unknown): string {
  return `<state>\n${typeof state === "string" ? state : pyJson(state)}\n</state>\n\n`;
}

type Part = { type: "text"; text: string } | { type: "image" };

/** The user turn split at "<image:N>" placeholders into chat content parts (qev media.content_parts). */
export function userContent(text: string, images: number): string | Part[] {
  if (!images) return text;
  const parts: Part[] = [];
  const re = /<image:(\d+)>/g;
  let pos = 0;
  const seen: number[] = [];
  for (const m of text.matchAll(re)) {
    if (m.index! > pos) parts.push({ type: "text", text: text.slice(pos, m.index) });
    parts.push({ type: "image" });
    seen.push(Number(m[1]));
    pos = m.index! + m[0].length;
  }
  if (pos < text.length) parts.push({ type: "text", text: text.slice(pos) });
  if (seen.join() !== Array.from({ length: images }, (_, i) => i + 1).join())
    throw new Error("Refer to each image once, in order, as <image:1>, <image:2>, … in the state.");
  return parts;
}

export function messages(state: unknown, suffix: string, images: number) {
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userContent(stateBlock(state) + suffix, images) },
  ];
}

export function softmax(logits: number[], temperature = 1): number[] {
  const z = logits.map((x) => x / temperature);
  const m = Math.max(...z);
  const w = z.map((x) => Math.exp(x - m));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((x) => x / s);
}

export function makeAnswer(q: Question, labels: string[], probs: number[]): Answer {
  if (q.type === "noul") return { type: "noul", noul: probs[0] };
  const best = probs.indexOf(Math.max(...probs));
  const k = probs.length;
  if (q.type === "choice") {
    const confidence = k === 1 ? 1 : (probs[best] - 1 / k) / (1 - 1 / k);
    return { type: "choice", choice: labels[best], probabilities: Object.fromEntries(labels.map((l, i) => [l, probs[i]])), confidence };
  }
  const score = probs.reduce((s, p, i) => s + i * p, 0);
  const center = (k - 1) / 2;
  const uniformMad = probs.reduce((s, _, i) => s + Math.abs(i - center), 0) / k;
  const distance = probs.reduce((s, p, i) => s + p * Math.abs(i - best), 0);
  return {
    type: "score",
    score,
    probabilities: Object.fromEntries(labels.map((l, i) => [l, probs[i]])),
    confidence: k === 1 ? 1 : Math.max(0, 1 - distance / uniformMad),
    legend: q.criteria as string[],
  };
}
