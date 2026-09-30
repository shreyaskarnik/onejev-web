// Parity: Transformers.js + our ONNX vs OneJev's own engine (convert/fixtures/reference.json).
import { AutoModelForImageTextToText, AutoProcessor, RawImage, env } from "@huggingface/transformers";
import { readFileSync } from "node:fs";
import { LETTERS, makeAnswer, messages, renderQuestion, softmax, type Question } from "./src/onejev";

const dtypeArg = process.argv[2] ?? "fp32";
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL("../models/", import.meta.url).pathname;
const FIX = new URL("../convert/fixtures/", import.meta.url).pathname;
const cases = JSON.parse(readFileSync(FIX + "reference.json", "utf8"));

const processor = await AutoProcessor.from_pretrained("OneJev-0.8B-ONNX");
const dtype = dtypeArg === "fp32" ? { embed_tokens: "fp32", vision_encoder: "fp32", decoder_model_merged: "fp32" }
  : { embed_tokens: "q4f16", vision_encoder: "fp16", decoder_model_merged: "q4f16" };
const model = await AutoModelForImageTextToText.from_pretrained("OneJev-0.8B-ONNX", { dtype, device: "cpu" } as never);
const tok = (processor as any).tokenizer;
const slot = LETTERS.split("").map((l) => { const ids = tok.encode(l, { add_special_tokens: false }); if (ids.length !== 1) throw new Error("letter " + l); return ids[0]; });

let worst = 0, flips = 0;
for (const c of cases) {
  const images = await Promise.all(c.media.map((m: any) => RawImage.read(FIX + m.file)));
  for (const [qid, q] of Object.entries<Question>(c.questions)) {
    const { suffix, labels } = renderQuestion(q);
    const text = (processor as any).apply_chat_template(messages(c.state, suffix, images.length), { add_generation_prompt: true, tokenize: false, enable_thinking: false });
    const inputs = images.length ? await (processor as any)(text, images) : await (processor as any)(text);
    const t0 = performance.now();
    const out = await (model as any)(inputs);
    const ms = performance.now() - t0;
    const [, L, V] = out.logits.dims;
    const data = out.logits.data as Float32Array;
    const last = labels.map((_, i) => data[(L - 1) * V + slot[i]]);
    const probs = softmax(last);
    const got = makeAnswer(q, labels, probs) as any;
    const a = q.type === "noul" ? [got.noul, 1 - got.noul] : Object.values(got.probabilities as Record<string, number>);
    const b: number[] = c.hf[qid];
    const d = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
    worst = Math.max(worst, d);
    const top = (xs: number[]) => xs.indexOf(Math.max(...xs));
    if (top(a) !== top(b)) flips++;
    console.log(`${c.name}/${qid} tokens ${L} ${Math.round(ms)} ms | ours ${a.map((v) => v.toFixed(3)).join(" ")} | ref ${b.map((v) => v.toFixed(3)).join(" ")} | max diff ${d.toFixed(4)}`);
  }
}
console.log(`${dtypeArg}: max |dp| ${worst.toFixed(4)}, top answer differs on ${flips} questions`);
