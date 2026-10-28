"""Lazy loaders for CLIP / NLI / BLIP-2 (Hugging Face). Heavy: install `transformers accelerate pillow` first."""
from __future__ import annotations
import numpy as np


def _need(pkg):
    raise ImportError(f"'{pkg}' is not installed. Run: pip install transformers accelerate pillow  (and faiss-cpu optionally)")


class Clip:
    def __init__(self, name, device):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            _need("transformers")
        self.torch, self.device = torch, device
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def image(self, pil_images):
        with self.torch.no_grad():
            x = self.proc(images=pil_images, return_tensors="pt").to(self.device)
            e = self.model.get_image_features(**x)
        return e.cpu().numpy()

    def text(self, texts):
        with self.torch.no_grad():
            x = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            e = self.model.get_text_features(**x)
        return e.cpu().numpy()


class NLI:
    def __init__(self, name, device):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            _need("transformers")
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        lab = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = [k for k, v in lab.items() if "entail" in v][0]

    def entail_prob(self, premise, hypothesis):
        with self.torch.no_grad():
            x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            p = self.torch.softmax(self.model(**x).logits, -1)[0]
        return float(p[self.ent])


class Blip2:
    def __init__(self, name, device):
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError:
            _need("transformers")
        self.torch, self.device = torch, device
        dt = torch.float16 if device.type == "cuda" else torch.float32
        self.proc = Blip2Processor.from_pretrained(name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(name, torch_dtype=dt).to(device).eval()
        self.dt = dt

    def explain(self, pil_image, context_captions):
        if context_captions:
            prompt = f"Question: Similar past incidents: {'; '.join(context_captions)}. In one sentence, what unusual event is happening in this image? Answer:"
        else:
            prompt = "Question: In one sentence, what unusual event is happening in this image? Answer:"
        x = self.proc(images=pil_image, text=prompt, return_tensors="pt").to(self.device, self.dt)
        with self.torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=30)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
