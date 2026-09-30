"""Make the decoder return logits for the last position only.

The graph's head is MatMul/MatMulNBits(final_norm_output, lm_head). A Slice on the
sequence axis in front of it keeps [batch, 1, hidden]; logits become [batch, 1, vocab]
(generation only ever reads the last position). For a 350-token prompt this cuts the
readback from about 175 MB (fp16) to 0.5 MB per question.
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

for name in sys.argv[1:]:
    path = f"out/onnx/{name}.onnx"
    m = onnx.load(path, load_external_data=False)
    head = [n for n in m.graph.node if n.name.startswith("/lm_head/")]
    assert len(head) == 1, [n.name for n in head]
    head = head[0]
    src = head.input[0]
    sliced = src + "/last_token"
    for arr, nm in ((np.array([-1], np.int64), "/lm_head/last/starts"), (np.array([np.iinfo(np.int64).max], np.int64), "/lm_head/last/ends"), (np.array([1], np.int64), "/lm_head/last/axes")):
        m.graph.initializer.append(numpy_helper.from_array(arr, nm))
    node = helper.make_node("Slice", [src, "/lm_head/last/starts", "/lm_head/last/ends", "/lm_head/last/axes"], [sliced], name="/lm_head/LastToken")
    idx = list(m.graph.node).index(head)
    m.graph.node.insert(idx, node)
    head.input[0] = sliced
    onnx.save(m, path)  # external data references are unchanged
    print(name, "head", head.op_type, "now reads", sliced)
