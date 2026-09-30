/// <reference lib="webworker" />
// OneJev-0.8B in a worker: one full forward per question, answer = softmax over the option letters.
import { AutoModelForImageTextToText, AutoProcessor, RawImage, env } from "@huggingface/transformers";
import { LETTERS, makeAnswer, messages, renderQuestion, softmax, type Answer, type Question } from "./onejev";

export type Request =
  | { type: "load"; source: string; device: "webgpu" | "wasm"; precision?: "q4f16" | "fp16" }
  | { type: "decide"; id: number; state: unknown; images: { data: Uint8ClampedArray; width: number; height: number }[]; questions: Record<string, Question> };
export type Response =
  | { type: "progress"; loaded: number; total: number }
  | { type: "loaded"; ms: number; device: string }
  | { type: "answer"; id: number; qid: string; answer: Answer; ms: number; tokens: number }
  | { type: "done"; id: number; ms: number }
  | { type: "error"; id?: number; message: string };

const HUB_ID = "onnx-community/OneJev-0.8B-ONNX";
const post = (m: Response) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m);

let processor: any = null;
let model: any = null;
let slots: number[] = [];

async function load(source: string, device: "webgpu" | "wasm", precision: "q4f16" | "fp16" = "q4f16") {
  const started = performance.now();
  let id = HUB_ID;
  if (source !== "hub") {
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = source;
    id = "OneJev-0.8B-ONNX";
  }
  const files: Record<string, { loaded: number; total: number }> = {};
  const progress_callback = (info: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (info.status !== "progress" || !info.file) return;
    files[info.file] = { loaded: info.loaded ?? 0, total: info.total ?? 0 };
    const sum = (k: "loaded" | "total") => Object.values(files).reduce((s, f) => s + f[k], 0);
    post({ type: "progress", loaded: sum("loaded"), total: sum("total") });
  };
  processor = await AutoProcessor.from_pretrained(id, { progress_callback } as never);
  model = await AutoModelForImageTextToText.from_pretrained(id, {
    device,
    dtype: { embed_tokens: precision, vision_encoder: device === "webgpu" ? "fp16" : "fp32", decoder_model_merged: precision },
    progress_callback,
  } as never);
  slots = LETTERS.split("").map((l) => processor.tokenizer.encode(l, { add_special_tokens: false })[0]);
  post({ type: "loaded", ms: performance.now() - started, device });
}

async function decide(req: Extract<Request, { type: "decide" }>) {
  const started = performance.now();
  const images = req.images.map((im) => new RawImage(im.data, im.width, im.height, 4).rgb());
  for (const [qid, q] of Object.entries(req.questions)) {
    const t0 = performance.now();
    const { suffix, labels } = renderQuestion(q);
    const text = processor.apply_chat_template(messages(req.state, suffix, images.length), {
      add_generation_prompt: true,
      tokenize: false,
      enable_thinking: false,
    });
    const inputs = images.length ? await processor(text, images) : await processor(text);
    const out = await model(inputs);
    const [, L, V] = out.logits.dims;
    const data = out.logits.data as Float32Array | Uint16Array;
    const row = (L - 1) * V;
    // fp16 logits come back as raw 16-bit words where Float16Array is unavailable.
    const read = (i: number) => (data instanceof Uint16Array ? f16(data[row + i]) : (data[row + i] as number));
    const probs = softmax(labels.map((_, i) => read(slots[i])));
    post({ type: "answer", id: req.id, qid, answer: makeAnswer(q, labels, probs), ms: performance.now() - t0, tokens: inputs.input_ids.dims[1] });
  }
  post({ type: "done", id: req.id, ms: performance.now() - started });
}

/** IEEE half to float, for fp16 logits returned as raw 16-bit words. */
function f16(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") await load(msg.source, msg.device, msg.precision);
    else await decide(msg);
  } catch (err) {
    post({ type: "error", id: msg.type === "decide" ? msg.id : undefined, message: (err as Error)?.stack ?? String(err) });
  }
};
