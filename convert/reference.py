"""Reference answers from OneJev's own engine (qev, PyTorch, CPU, fp32) for the parity test.

Writes fixtures/: the test images and reference.json with, per case, the state, the
media, the questions and every question's probabilities (temperature 1, one ordering).
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from qev.media import data_uri
from qev.mm_engine import MMDecisionEngine
from qev.schema import SystemOneRequest

OUT = Path("fixtures")
OUT.mkdir(exist_ok=True)


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def review_card(path):
    img = Image.new("RGB", (640, 352), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 640, 56], fill="#232f3e")
    d.text((20, 16), "Customer reviews", fill="white", font=font(22))
    d.text((20, 76), "★☆☆☆☆  Worst purchase ever", fill="#c45500", font=font(24))
    d.text((20, 114), "Reviewed by Dana K. · Verified purchase", fill="#565959", font=font(16))
    body = ("The blender caught fire the second time I used it.\n"
            "Smoke everywhere, my kitchen still smells. I want a full\n"
            "refund TODAY or I am calling my lawyer and posting\n"
            "videos everywhere. Nobody from support has replied in 5 days.")
    d.multiline_text((20, 150), body, fill="#0f1111", font=font(19), spacing=8)
    img.save(path)


def payment_screen(path):
    img = Image.new("RGB", (576, 384), "#f6f8fa")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 32, 536, 352], fill="white", outline="#d0d7de")
    d.ellipse([250, 70, 310, 130], fill="#1a7f37")
    d.text((262, 84), "✓", fill="white", font=font(34))
    d.text((150, 150), "Payment successful", fill="#1f2328", font=font(30))
    d.text((120, 200), "Invoice INV-2291 from ACME Corp", fill="#57606a", font=font(18))
    d.text((120, 230), "Amount paid: $1,240.00", fill="#57606a", font=font(18))
    d.rectangle([180, 290, 380, 330], fill="#0969da")
    d.text((222, 298), "Back to invoices", fill="white", font=font(18))
    img.save(path)


review_card(OUT / "review.png")
payment_screen(OUT / "payment.png")

CASES = [
    {
        "name": "text-support",
        "state": {"channel": "email", "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
        "media": [],
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, outages", "sales": "pricing, new plans"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "today", "right now"]},
            "churn": {"type": "noul", "instructions": "The customer threatens to cancel."},
        },
    },
    {
        "name": "image-review",
        "state": {"source": "product review screenshot", "review": "<image:1>"},
        "media": [{"type": "image", "file": "review.png"}],
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "This review needs a reply from support today."},
            "sentiment": {"type": "choice", "instructions": "What is the reviewer's sentiment?",
                          "criteria": {"positive": None, "mixed": None, "negative": None, "furious": "angry and threatening action"}},
            "safety": {"type": "noul", "instructions": "The review reports a safety hazard."},
            "stars": {"type": "score", "instructions": "How many stars did the reviewer give?", "criteria": ["1", "2", "3", "4", "5"]},
        },
    },
    {
        "name": "image-payment",
        "state": {"task": "Pay the open invoice from ACME", "screen": "<image:1>"},
        "media": [{"type": "image", "file": "payment.png"}],
        "questions": {
            "done": {"type": "noul", "instructions": "The invoice has been paid."},
            "next": {"type": "choice", "instructions": "What should the agent do next?",
                     "criteria": {"click": "click an element", "type": "type text", "scroll": "scroll the page", "stop": "stop, the task is finished"}},
        },
    },
]

engine = MMDecisionEngine("OmniJev/OneJev-0.8B", device="cpu", dtype="float32", head_dtype="float32", fork_mode="sequential")
for case in CASES:
    media = [{"type": "image", "data": data_uri(str(OUT / m["file"]))} for m in case["media"]]
    request = SystemOneRequest.model_validate({"model": "onejev", "state": case["state"], "questions": case["questions"]})
    response, meta = engine.decide(request, media=media)
    answers = response.model_dump()["answers"]
    case["answers"] = answers
    print(case["name"], json.dumps(answers)[:400])
(OUT / "reference.json").write_text(json.dumps(CASES, indent=1))
