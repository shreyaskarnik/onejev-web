# OneJev in the browser

Show it a screenshot or a photo, ask typed questions (yes/no, choice, score), and get a probability for every option from [OneJev-0.8B](https://huggingface.co/OmniJev/OneJev-0.8B), a multimodal System One decision model. It runs on your GPU with WebGPU through Transformers.js; the image never leaves the page.

## How it works

Each question is one chat turn, built exactly as OneJev's own engine builds it (`qev/prompt.py`): a system prompt, the state in `<state>…</state>` with the image inline where `<image:1>` stands, the question with lettered options, "Answer with one letter". One forward pass per question; the answer is the softmax over the option letters. See `web/src/onejev.ts`.

The ONNX files ([onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)) reuse onnx-community's Qwen3.5-0.8B-ONNX-OPT graphs with OneJev's weights, matched by value and re-quantized (`convert/`).

## Checks

- fp32 ONNX vs a Hugging Face forward pass: max probability difference 0.0011 over 9 text and image questions.
- WebGPU fp16: 0.030, no top answer changed. WebGPU q4f16: 0.025 on 8 of 9, one near-tie flipped.
- About 150 ms per text question and 500 ms per image question on an M3 Pro.

## Develop

```bash
cd convert && uv venv .venv && uv pip install onnx onnxruntime onnx_ir torch "transformers>=5" accelerate pillow torchvision safetensors huggingface_hub -e <path to OneJev>
.venv/bin/python map_weights.py   # BASE_REPO/BASE_DIR/MAPPING select the base graphs
.venv/bin/python build_onnx.py && .venv/bin/python build_fp16.py
cd ../web && npm install && npx vite   # /?models=local uses ../models
```

Apache-2.0, as OneJev.
