"""fp16 embed_tokens / decoder_model_merged: OneJev weights into the base fp16 graphs (cast to each initializer's dtype)."""
import json, os, sys
from pathlib import Path
import onnx
from build_onnx import MAPPING, onejev_params, save, set_init, transformed, BASE

params = onejev_params()
for graph in ("embed_tokens", "decoder_model_merged"):
    spec = MAPPING[graph]
    model = onnx.load(str(BASE / "onnx" / f"{graph}_fp16.onnx"), load_external_data=True)
    n = 0
    for init in model.graph.initializer:
        if init.name in spec:
            set_init(init, transformed(params, spec[init.name]))
            n += 1
    print(graph, "fp16: replaced", n, "of", len(spec), file=sys.stderr)
    save(model, f"{graph}_fp16")
