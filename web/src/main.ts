import type { Answer, Question } from "./onejev";
import type { Request, Response } from "./worker";

const MAX_SIDE = 896;
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: unknown) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const params = new URLSearchParams(location.search);
// onnx-community/OneJev-0.8B-ONNX by default; ?models=local serves ./models instead.
const SOURCE = params.get("models") === "local" ? "/models/" : "hub";

type Kind = Question["type"];
type QDraft = { kind: Kind; text: string; options: string };
type Img = { data: Uint8ClampedArray; width: number; height: number; url: string };

let image: Img | null = null;
let questions: QDraft[] = [];
let device: "webgpu" | "wasm" = "wasm";
let loadedPrecision: string | null = null;
let runId = 0;

const PRESETS: Record<string, { context: string; questions: QDraft[] }> = {
  review: {
    context: "product review screenshot",
    questions: [
      { kind: "noul", text: "This review needs a reply from support today.", options: "" },
      { kind: "choice", text: "What is the reviewer's sentiment?", options: "positive\nmixed\nnegative\nfurious: angry and threatening action" },
      { kind: "noul", text: "The review reports a safety hazard.", options: "" },
      { kind: "score", text: "How damaging is this review to the brand?", options: "harmless\nminor\nserious\nsevere" },
    ],
  },
  payment: {
    context: "Task: pay the open invoice from ACME",
    questions: [
      { kind: "noul", text: "The invoice has been paid.", options: "" },
      { kind: "choice", text: "What should the agent do next?", options: "click: click an element\ntype: type text\nscroll: scroll the page\nstop: stop, the task is finished" },
      { kind: "score", text: "How far along is the task?", options: "not started\nhalfway\nalmost done\ndone" },
    ],
  },
  text: {
    context: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    questions: [
      { kind: "choice", text: "Which team should handle this?", options: "billing: charges, invoices, refunds\ntechnical: bugs, outages\nsales: pricing, new plans" },
      { kind: "score", text: "How urgent is this?", options: "not urgent\nsoon\ntoday\nright now" },
      { kind: "noul", text: "The customer threatens to cancel.", options: "" },
    ],
  },
};

// ------------------------------------------------------------------ worker
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let onMessage: (m: Response) => void = () => {};
worker.onmessage = (e: MessageEvent<Response>) => onMessage(e.data);

function status(text: string, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "muted";
}
function progress(fraction: number | null) {
  $("progress").hidden = fraction === null;
  ($("progress").firstElementChild as HTMLElement).style.width = `${Math.round((fraction ?? 0) * 100)}%`;
}

function ensureModel(): Promise<void> {
  const precision = ($("precision") as HTMLSelectElement).value as "q4f16" | "fp16";
  if (loadedPrecision === precision) return Promise.resolve();
  return new Promise((resolve, reject) => {
    status(`Downloading OneJev-0.8B (${precision === "fp16" ? "2.2" : "0.8"} GB, cached after the first time)…`);
    progress(0);
    onMessage = (m) => {
      if (m.type === "progress") progress(m.total ? m.loaded / m.total : 0);
      else if (m.type === "loaded") {
        loadedPrecision = precision;
        progress(null);
        resolve();
      } else if (m.type === "error") reject(new Error(m.message));
    };
    worker.postMessage({ type: "load", source: SOURCE, device, precision } satisfies Request);
  });
}

// ------------------------------------------------------------------ image
async function setImage(blob: Blob | null) {
  if (!blob) {
    image = null;
    ($("preview") as HTMLImageElement).hidden = true;
    $("drop-text").hidden = false;
    $("image-info").textContent = "no image";
    return;
  }
  const bmp = await createImageBitmap(blob);
  const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const url = URL.createObjectURL(blob);
  image = { data, width: w, height: h, url };
  const img = $("preview") as HTMLImageElement;
  img.src = url;
  img.hidden = false;
  $("drop-text").hidden = true;
  $("image-info").textContent = `${w} × ${h}${scale < 1 ? ` (scaled from ${bmp.width} × ${bmp.height})` : ""}`;
}

// ------------------------------------------------------------------ questions
function renderQuestions() {
  $("questions").innerHTML = questions
    .map(
      (q, i) => `<div class="q" data-i="${i}">
        <div class="q-row">
          <select data-f="kind">
            <option value="noul"${q.kind === "noul" ? " selected" : ""}>Yes / no</option>
            <option value="choice"${q.kind === "choice" ? " selected" : ""}>Choice</option>
            <option value="score"${q.kind === "score" ? " selected" : ""}>Score</option>
          </select>
          <input type="text" data-f="text" value="${esc(q.text)}" placeholder="${q.kind === "noul" ? "A statement to check" : "The question"}" />
          <button class="icon" data-remove title="Remove">✕</button>
        </div>
        ${q.kind === "noul" ? "" : `<textarea data-f="options" placeholder="${q.kind === "choice" ? "one option per line, optionally name: description" : "one level per line, lowest first"}">${esc(q.options)}</textarea>
        <span class="hint">${q.kind === "choice" ? "One option per line. Add a short description after a colon." : "Levels from lowest to highest, one per line."}</span>`}
      </div>`,
    )
    .join("");
}

function toQuestion(q: QDraft): Question {
  const lines = q.options.split("\n").map((l) => l.trim()).filter(Boolean);
  if (q.kind === "noul") return { type: "noul", instructions: q.text };
  if (q.kind === "score") {
    if (lines.length < 2) throw new Error(`"${q.text}" needs at least two levels.`);
    return { type: "score", instructions: q.text, criteria: lines };
  }
  if (lines.length < 2) throw new Error(`"${q.text}" needs at least two options.`);
  const criteria: Record<string, string | null> = {};
  for (const line of lines) {
    const at = line.indexOf(":");
    const name = (at > 0 ? line.slice(0, at) : line).trim();
    criteria[name] = at > 0 ? line.slice(at + 1).trim() || null : null;
  }
  return { type: "choice", instructions: q.text, criteria };
}

// ------------------------------------------------------------------ run
function state(): unknown {
  const context = ($("context") as HTMLInputElement).value.trim();
  if (!image) return context || "(empty)";
  return context ? { context, image: "<image:1>" } : { image: "<image:1>" };
}

function answerCard(title: string, a: Answer, ms: number): string {
  const bars = (entries: [string, number][]) => {
    const top = Math.max(...entries.map(([, p]) => p));
    return `<div class="bars">${entries
      .map(([label, p]) => `<div class="bar${p === top ? " best" : ""}"><span class="label" title="${esc(label)}">${esc(label)}</span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(1)}%"></span></span><span class="pct">${(p * 100).toFixed(0)}%</span></div>`)
      .join("")}</div>`;
  };
  let body: string;
  if (a.type === "noul") {
    body = `<div class="big">${a.noul >= 0.5 ? "Yes" : "No"} <span class="muted" style="font-size:15px;font-weight:500">p(yes) = ${a.noul.toFixed(2)}</span></div>${bars([["yes", a.noul], ["no", 1 - a.noul]])}`;
  } else if (a.type === "choice") {
    body = `<div class="big">${esc(a.choice)} <span class="muted" style="font-size:15px;font-weight:500">confidence ${a.confidence.toFixed(2)}</span></div>${bars(Object.entries(a.probabilities))}`;
  } else {
    const n = a.legend.length - 1;
    body = `<div class="big">${a.score.toFixed(2)} <span class="muted" style="font-size:15px;font-weight:500">of ${n} · nearest: ${esc(a.legend[Math.round(a.score)])}</span></div>${bars(a.legend.map((l, i) => [l, a.probabilities[String(i)]] as [string, number]))}`;
  }
  const kind = a.type === "noul" ? "yes / no" : a.type;
  return `<div class="a"><span class="kind">${kind}</span><h3>${esc(title)}</h3>${body}<div class="ms">${Math.round(ms)} ms</div></div>`;
}

async function run() {
  const button = $("run") as HTMLButtonElement;
  button.disabled = true;
  try {
    const qs = questions.filter((q) => q.text.trim());
    if (!qs.length) throw new Error("Add at least one question.");
    const record: Record<string, Question> = {};
    qs.forEach((q, i) => (record[`q${i + 1}`] = toQuestion(q)));
    await ensureModel();
    const id = ++runId;
    const st = state();
    $("answers-card").hidden = false;
    $("answers").innerHTML = "";
    $("timing").textContent = "";
    $("request").textContent = JSON.stringify(
      { model: "onejev", state: st, ...(image ? { media: [{ type: "image", data: "data:image/png;base64,…" }] } : {}), questions: record },
      null,
      2,
    );
    status(`Asking ${qs.length} question${qs.length > 1 ? "s" : ""} on ${device === "webgpu" ? "WebGPU" : "WASM"}…`);
    await new Promise<void>((resolve, reject) => {
      onMessage = (m) => {
        if (m.type === "answer" && m.id === id) {
          const i = Number(m.qid.slice(1)) - 1;
          $("answers").insertAdjacentHTML("beforeend", answerCard(qs[i].text, m.answer, m.ms));
        } else if (m.type === "done" && m.id === id) {
          $("timing").textContent = `${qs.length} questions in ${(m.ms / 1000).toFixed(2)} s · one forward pass each`;
          resolve();
        } else if (m.type === "error") reject(new Error(m.message));
      };
      const images = image ? [{ data: image.data, width: image.width, height: image.height }] : [];
      worker.postMessage({ type: "decide", id, state: st, images, questions: record } satisfies Request);
    });
    status("");
  } catch (err) {
    progress(null);
    status((err as Error).message.split("\n")[0], true);
  } finally {
    button.disabled = false;
  }
}

async function loadSample(name: string) {
  const preset = PRESETS[name];
  questions = preset.questions.map((q) => ({ ...q }));
  ($("context") as HTMLInputElement).value = preset.context;
  renderQuestions();
  if (name === "text") await setImage(null);
  else await setImage(await (await fetch(new URL(`samples/${name}.png`, location.href))).blob());
}

// ------------------------------------------------------------------ wiring
async function init() {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (gpu && (await gpu.requestAdapter())) device = "webgpu";
  } catch {
    device = "wasm";
  }
  $("device").textContent = device === "webgpu" ? "WebGPU" : "WASM (no WebGPU: slow)";
  $("device").classList.toggle("ok", device === "webgpu");

  const file = $("file") as HTMLInputElement;
  file.addEventListener("change", () => file.files?.[0] && setImage(file.files[0]));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => (e.preventDefault(), drop.classList.add("over")));
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer?.files?.[0];
    if (f?.type.startsWith("image/")) setImage(f);
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
    if (item) setImage(item.getAsFile());
  });
  document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((b) => b.addEventListener("click", () => loadSample(b.dataset.sample!)));

  const qs = $("questions");
  qs.addEventListener("input", (e) => {
    const el = e.target as HTMLInputElement;
    const i = Number(el.closest<HTMLElement>(".q")!.dataset.i);
    const f = el.dataset.f as keyof QDraft | undefined;
    if (f === "text" || f === "options") questions[i][f] = el.value;
  });
  qs.addEventListener("change", (e) => {
    const el = e.target as HTMLSelectElement;
    if (el.dataset.f !== "kind") return;
    const i = Number(el.closest<HTMLElement>(".q")!.dataset.i);
    questions[i].kind = el.value as Kind;
    renderQuestions();
  });
  qs.addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest("[data-remove]");
    if (!btn) return;
    questions.splice(Number(btn.closest<HTMLElement>(".q")!.dataset.i), 1);
    renderQuestions();
  });
  $("add").addEventListener("click", () => {
    questions.push({ kind: "noul", text: "", options: "" });
    renderQuestions();
  });
  $("run").addEventListener("click", run);
  await loadSample(params.get("sample") && PRESETS[params.get("sample")!] ? params.get("sample")! : "review");
}

init();
