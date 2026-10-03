# Modal domain benchmarks

Run the cheap correctness/overhead smoke suite first:

```bash
modal run benchmarks/modal_benchmark.py --suite smoke
```

Run model-backed forward benchmarks independently:

```bash
modal run benchmarks/modal_benchmark.py --suite forward --domain llm
modal run benchmarks/modal_benchmark.py --suite forward --domain vlm
modal run benchmarks/modal_benchmark.py --suite forward --domain diffusion
```

The forward suite reports adapter parameter count and synchronized GPU latency.
It is not a quality benchmark. To claim an improvement over LoRA, add a fixed
dataset and training protocol per domain, then run the same seeds and parameter
budget for LoRA, shared/fused LoRA, DoRA, a matched MLP adapter, and
`SharedBilinearKoRA`. Save configs, checkpoints, predictions, and metrics under
a run ID. The first end-to-end candidates should be:

- LLM: `sshleifer/tiny-gpt2`, language-modeling loss on a pinned text shard.
- VLM: a tiny CLIP vision encoder, image-text contrastive loss on a pinned image
  subset.
- Diffusion: a tiny conditional UNet, fixed-noise denoising loss on a pinned
  image subset.

Do not compare raw latency across domains or model IDs. Record GPU type,
precision, batch size, token/image resolution, warmup count, and model revision.
