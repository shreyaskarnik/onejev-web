"""Match every weight of onnx-community's Qwen3.5-0.8B ONNX graphs to the Hugging Face
parameter it was built from (by value, allowing transpose / reshape / bf16 rounding).

The match is what lets us rebuild the same graphs with OneJev-0.8B's weights: OneJev
is a full fine-tune of Qwen3.5-0.8B with the same shapes, so each ONNX initializer is
the same transform of the same parameter, only with different values.
Writes mapping.json: {graph: {initializer: {"param": name, "transform": ...}}}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download
from onnx import numpy_helper
from safetensors import safe_open

import os
BASE_REPO = os.environ.get("BASE_REPO", "onnx-community/Qwen3.5-0.8B-ONNX")
BASE_DIR = os.environ.get("BASE_DIR", "base_onnx")
ONNX_DIR = Path(snapshot_download(BASE_REPO, allow_patterns=["onnx/*.onnx*", "*.json", "*.jinja"], ignore_patterns=["*fp16*", "*q4.*", "*q4.onnx*", "*quantized*"], local_dir=BASE_DIR)) / "onnx"
HF_DIR = Path(snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=["*.safetensors", "*.json"]))


def hf_params():
    out = {}
    for f in sorted(HF_DIR.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                out[k] = st.get_tensor(k).float().numpy()
    return out


def candidates(p):
    yield "identity", p
    if p.ndim == 2:
        yield "transpose", p.T
    if p.ndim >= 2:
        yield "flatten", p.reshape(p.shape[0], -1)


def main():
    params = hf_params()
    print(len(params), "hf params", file=sys.stderr)
    by_size = {}
    for k, v in params.items():
        by_size.setdefault(v.size, []).append(k)
    mapping, unmatched = {}, {}
    for graph in ("embed_tokens", "vision_encoder", "decoder_model_merged"):
        model = onnx.load(str(ONNX_DIR / f"{graph}.onnx"), load_external_data=True)
        mapping[graph] = {}
        unmatched[graph] = []
        for init in model.graph.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL and len(init.raw_data) < 1024 and not init.external_data:
                continue  # small constants are part of the graph, not weights
            w = numpy_helper.to_array(init).astype(np.float32)
            hit = None
            flat = w.reshape(-1)
            for name in by_size.get(w.size, []):
                p = params[name]
                if p.shape == w.shape and np.array_equal(p, w):
                    hit = (name, "identity")
                elif p.ndim == 2 and p.T.shape == w.shape and np.array_equal(p.T, w):
                    hit = (name, "transpose")
                elif np.array_equal(p.reshape(-1), flat):
                    hit = (name, "reshape")
                elif p.shape == w.shape and np.allclose(p + 1.0, w, rtol=0, atol=1e-6):
                    hit = (name, "plus_one")  # zero-centred RMSNorm stores weight - 1
                elif p.ndim > 2 and np.array_equal(p.reshape(p.shape[0], -1).T, w):
                    hit = (name, "flatten_transpose")  # conv patch embedding as a MatMul
                if hit:
                    break
            if hit:
                mapping[graph][init.name] = {"param": hit[0], "transform": hit[1], "shape": list(w.shape)}
            else:
                unmatched[graph].append({"name": init.name, "shape": list(w.shape)})
        print(graph, "matched", len(mapping[graph]), "unmatched", len(unmatched[graph]), file=sys.stderr)
    Path(os.environ.get("MAPPING", "mapping.json")).write_text(json.dumps({"mapping": mapping, "unmatched": unmatched}, indent=1))


if __name__ == "__main__":
    main()
