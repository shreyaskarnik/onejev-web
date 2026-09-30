"""Build OneJev-0.8B ONNX files from onnx-community/Qwen3.5-0.8B-ONNX's graphs.

OneJev-0.8B (OmniJev, Apache-2.0) is a full fine-tune of Qwen3.5-0.8B with the vision
tower frozen: same architecture, same shapes, identical vision weights. So we keep the
base repo's graphs and put OneJev's weights into them:

- fp32 embed_tokens / decoder_model_merged: every weight is replaced by the OneJev
  parameter it maps to (mapping.json, found by value against the base weights), with
  the same transform (transpose, reshape, 1 + weight for zero-centred RMSNorm).
- q4f16 embed_tokens / decoder_model_merged: the same, then 4-bit block-32 asymmetric
  quantization with ONNX Runtime's own kernel (the base graph's MatMulNBits and
  GatherBlockQuantized layout; the error matches the base's own quantization).
- vision_encoder (all variants): copied unchanged; OneJev did not train it.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from onnx import numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors import safe_open

import os
BASE = Path(os.environ.get("BASE_DIR", "base_onnx"))
OUT = Path(os.environ.get("OUT", "out"))
MAPPING = json.loads(Path(os.environ.get("MAPPING", "mapping.json")).read_text())["mapping"]
ONEJEV = Path(snapshot_download("OmniJev/OneJev-0.8B"))


def onejev_params():
    out = {}
    for f in sorted(ONEJEV.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                out[k] = st.get_tensor(k).to(torch.float32).numpy()
    return out


def transformed(params, spec):
    p = params[spec["param"]]
    how = spec["transform"]
    if how == "transpose":
        p = p.T
    elif how == "plus_one":
        p = p + 1.0
    elif how == "flatten_transpose":
        p = p.reshape(p.shape[0], -1).T
    return np.ascontiguousarray(p.reshape(spec["shape"]))


def quant4(w):
    """[K, N] float -> MatMulNBits (packed [N, K/32, 16], fp16 scales [N, K/32], zp [N, K/64])."""
    K, N = w.shape
    kb = (K + 31) // 32
    packed = np.zeros((N, kb, 16), "uint8")
    zp = np.zeros((N, (kb + 1) // 2), "uint8")
    scales = np.zeros((N, kb), np.float16)
    quantize_matmul_4bits(packed, np.ascontiguousarray(w.astype(np.float16)), scales, zp, 32, N, K, False)
    return packed, scales, zp


def set_init(init, array):
    new = numpy_helper.from_array(array.astype(numpy_helper.to_array(init).dtype, copy=False), init.name)
    init.CopyFrom(new)


def save(model, name):
    path = OUT / "onnx" / f"{name}.onnx"
    for stale in path.parent.glob(f"{name}.onnx_data*"):
        stale.unlink()
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=f"{name}.onnx_data", size_threshold=1024)
    print("wrote", path, file=sys.stderr)


def main():
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    params = onejev_params()

    for graph in ("embed_tokens", "decoder_model_merged"):
        spec = MAPPING[graph]
        # fp32: swap every mapped weight.
        model = onnx.load(str(BASE / "onnx" / f"{graph}.onnx"), load_external_data=True)
        for init in model.graph.initializer:
            if init.name in spec:
                set_init(init, transformed(params, spec[init.name]))
        save(model, graph)
        del model

        # q4f16: re-quantize the quantized weights, swap the rest.
        by_flat = {name.replace(".", "_"): name for name in spec}
        model = onnx.load(str(BASE / "onnx" / f"{graph}_q4f16.onnx"), load_external_data=True)
        inits = {i.name: i for i in model.graph.initializer}
        done = 0
        for name, init in inits.items():
            if name.endswith("_quant"):
                stem = name[: -len("_quant")]
                w = transformed(params, spec[by_flat[stem]])
                if graph == "embed_tokens":  # GatherBlockQuantized: blocks along the hidden size of each row
                    w = w.T
                packed, scales, zp = quant4(w)
                set_init(init, packed.reshape(list(init.dims)))
                set_init(inits[stem + "_scales"], scales.reshape(list(inits[stem + "_scales"].dims)))
                set_init(inits[stem + "_zp"], zp.reshape(list(inits[stem + "_zp"].dims)))
                done += 1
            elif name in spec:
                set_init(init, transformed(params, spec[name]))
                done += 1
        print(graph, "q4f16: replaced", done, file=sys.stderr)
        save(model, f"{graph}_q4f16")
        del model

    for f in BASE.glob("onnx/vision_encoder*"):
        shutil.copy(f, OUT / "onnx" / f.name)
    # Model files from OneJev; the Transformers.js settings and vision preprocessor from the base export.
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
        if (ONEJEV / name).exists():
            shutil.copy(ONEJEV / name, OUT / name)
    config = json.loads((OUT / "config.json").read_text())
    config["transformers.js_config"] = {
        "use_external_data_format": {"decoder_model_merged.onnx": 1, "decoder_model_merged_q4f16.onnx": 1, "embed_tokens.onnx": 1,
                                     "embed_tokens_q4f16.onnx": 1, "vision_encoder.onnx": 1, "vision_encoder_fp16.onnx": 1, "vision_encoder_q4f16.onnx": 1},
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(BASE / "preprocessor_config.json", OUT / "preprocessor_config.json")
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
