// Browser parity: the demo's worker (q4f16 decoder + fp16 vision on WebGPU) vs Hugging Face's fp32 forward.
import type { Request, Response } from "./worker";
const log = document.getElementById("log")!;
const out = (s: string) => { log.textContent += s + "\n"; console.log(s); };
const params = new URLSearchParams(location.search);
const device = (params.get("device") ?? "webgpu") as "webgpu" | "wasm";
const precision = (params.get("precision") ?? "q4f16") as "q4f16" | "fp16";

async function pixels(url: string) {
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height);
  return { data: d.data, width: d.width, height: d.height };
}

(async () => {
  try {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const call = (req: Request, until: (m: Response) => boolean, each?: (m: Response) => void) =>
      new Promise<void>((resolve, reject) => {
        worker.onmessage = (e: MessageEvent<Response>) => {
          const m = e.data;
          if (m.type === "error") return reject(new Error(m.message));
          each?.(m);
          if (until(m)) resolve();
        };
        worker.postMessage(req);
      });
    const t0 = performance.now();
    await call({ type: "load", source: "/models/", device, precision }, (m) => m.type === "loaded");
    out(`loaded in ${Math.round(performance.now() - t0)} ms on ${device}`);
    const cases = await (await fetch("/fixtures/reference.json")).json();
    let worst = 0, flips = 0;
    const rows: unknown[] = [];
    for (const [i, c] of cases.entries()) {
      const images = await Promise.all(c.media.map((m: { file: string }) => pixels(`/fixtures/${m.file}`)));
      await call({ type: "decide", id: i, state: c.state, images, questions: c.questions }, (m) => m.type === "done", (m) => {
        if (m.type !== "answer") return;
        const a: any = m.answer;
        const got: number[] = a.type === "noul" ? [a.noul, 1 - a.noul] : Object.values(a.probabilities);
        const ref: number[] = c.hf[m.qid];
        const d = Math.max(...got.map((v, k) => Math.abs(v - ref[k])));
        const top = (x: number[]) => x.indexOf(Math.max(...x));
        worst = Math.max(worst, d);
        if (top(got) !== top(ref)) flips++;
        rows.push({ q: `${c.name}/${m.qid}`, tokens: m.tokens, ms: Math.round(m.ms), d: +d.toFixed(4) });
        out(`${c.name}/${m.qid} ${m.tokens} tokens ${Math.round(m.ms)} ms | ours ${got.map((v) => v.toFixed(3)).join(" ")} | hf ${ref.map((v) => v.toFixed(3)).join(" ")} | diff ${d.toFixed(4)}`);
      });
    }
    out("RESULT " + JSON.stringify({ device, precision, worst, flips, rows }));
    out("DONE");
  } catch (err) {
    out("ERROR " + ((err as Error)?.stack ?? err));
  }
})();
