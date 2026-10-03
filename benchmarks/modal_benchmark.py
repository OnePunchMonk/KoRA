"""Modal benchmark harness for the shared-latent KoRA adapter.

Examples:
  modal run benchmarks/modal_benchmark.py --suite smoke
  modal run benchmarks/modal_benchmark.py --suite forward --domain llm
  modal run benchmarks/modal_benchmark.py --suite forward --domain vlm
  modal run benchmarks/modal_benchmark.py --suite forward --domain diffusion

The smoke suite uses tiny locally-created PyTorch projections. The forward
suite loads a named Hugging Face model and measures adapter overhead only; it
does not claim task-quality improvement. End-to-end quality requires a
separate, fixed-data training job.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import modal


app = modal.App("kora-domain-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "diffusers", "accelerate")
    .add_local_dir("src", remote_path="/root/src")
)


@dataclass
class Result:
    domain: str
    method: str
    model: str
    parameters: int
    batch: int
    sequence_or_tokens: int
    mean_ms: float
    p95_ms: float
    device: str


def _timed(fn, warmup: int = 3, repeats: int = 10):
    import torch

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    values.sort()
    return sum(values) / len(values), values[max(0, int(len(values) * .95) - 1)]


@app.function(image=image, gpu="T4", timeout=1800)
def benchmark(domain: str = "llm", model_id: str = "", smoke: bool = False) -> dict:
    import sys
    sys.path.insert(0, "/root/src")
    import torch
    from torch import nn
    from kora import SharedBilinearKoRA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if smoke:
        dimensions = {"llm": (128, 256, 128), "vlm": (256, 256, 64), "diffusion": (320, 640, 77)}
        in_features, out_features, tokens = dimensions[domain]
        model_name = "synthetic"
        projection = nn.Linear(in_features, out_features, bias=False).to(device)
    elif domain == "llm":
        from transformers import AutoModelForCausalLM
        model_id = model_id or "sshleifer/tiny-gpt2"
        projection = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        in_features = projection.config.n_embd
        out_features = in_features
        tokens = 128
        model_name = model_id
    elif domain == "vlm":
        from transformers import CLIPVisionModel
        model_id = model_id or "openai/clip-vit-base-patch32"
        projection = CLIPVisionModel.from_pretrained(model_id).to(device)
        in_features = projection.config.hidden_size
        out_features = in_features
        tokens = 50
        model_name = model_id
    elif domain == "diffusion":
        from diffusers import UNet2DConditionModel
        model_id = model_id or "hf-internal-testing/tiny-stable-diffusion-pipe"
        projection = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        in_features = projection.config.cross_attention_dim
        out_features = in_features
        tokens = 77
        model_name = model_id
    else:
        raise ValueError(f"unknown domain: {domain}")

    x = torch.randn(1, tokens, in_features, device=device)
    rank = min(16, in_features)
    kora = SharedBilinearKoRA(in_features, out_features, rank=rank, interaction_rank=2).to(device)

    class LoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.empty(rank, in_features))
            self.b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        def forward(self, x):
            return (x @ self.a.t()) @ self.b.t()

    lora = LoRA().to(device)
    results = []
    for method, module, params in (
        ("lora", lora, sum(p.numel() for p in lora.parameters())),
        ("shared_bilinear_kora", kora, kora.trainable_parameters),
    ):
        mean_ms, p95_ms = _timed(lambda: module(x))
        results.append(asdict(Result(domain, method, model_name, params, 1, tokens, mean_ms, p95_ms, device)))
    return results


@app.local_entrypoint()
def main(suite: str = "smoke", domain: str = "all", model_id: str = ""):
    domains = ["llm", "vlm", "diffusion"] if domain == "all" else [domain]
    output = []
    for name in domains:
        output.append(benchmark.remote(name, model_id, suite == "smoke"))
    print(json.dumps(output, indent=2))
