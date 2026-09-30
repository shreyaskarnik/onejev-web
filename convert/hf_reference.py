"""Reference probabilities from a plain Hugging Face forward pass (fp32, CPU) for every
question in fixtures/reference.json. The browser port is checked against these; they
match qev on text, while qev's cached path drifts a little on images."""
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from qev import media as mm
from qev.prompt import build_messages, render_question
from qev.schema import SystemOneRequest

cases = json.load(open("fixtures/reference.json"))
p = AutoProcessor.from_pretrained("OmniJev/OneJev-0.8B")
slots = [p.tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
m = Qwen3_5ForConditionalGeneration.from_pretrained("OmniJev/OneJev-0.8B", dtype=torch.float32).eval()
for c in cases:
    req = SystemOneRequest.model_validate({"model": "x", "state": c["state"], "questions": c["questions"]})
    imgs = [Image.open("fixtures/" + x["file"]).convert("RGB") for x in c["media"]]
    c["hf"] = {}
    for qid, q in req.questions.items():
        r = render_question(qid, q)
        text = mm.render_prompt(p, {"enable_thinking": False}, build_messages(c["state"], r.suffix), [{"type": "image"}] * len(imgs))
        out = p(text=[text], images=imgs or None, return_tensors="pt")
        with torch.inference_mode():
            lg = m(**out).logits[0, -1]
        c["hf"][qid] = torch.softmax(lg[slots[: r.n_slots]], 0).tolist()
    print(c["name"], {k: [round(x, 3) for x in v] for k, v in c["hf"].items()})
json.dump(cases, open("fixtures/reference.json", "w"), indent=1)
