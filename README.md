# KoRA: Kolmogorov-inspired Compositional Adapters for Robust Fine-Tuning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/onepunchmonk/kora/actions/workflows/ci.yml/badge.svg)](https://github.com/onepunchmonk/kora/actions/workflows/ci.yml)

KoRA is a novel parameter-efficient fine-tuning (PEFT) strategy that introduces **inter-adapter communication** to learn robust, generalizable representations that transfer across domains, addressing a key limitation in methods like LoRA.

---

## 🎯 The Problem: The Brittleness of Specialization

Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA have been revolutionary. They allow us to adapt massive pre-trained models to specific tasks at a fraction of the cost. However, this efficiency comes with a hidden price: **brittleness**.

Standard LoRA works by injecting small, independent adapters into each layer. These adapters are highly effective at specializing for a single task but often overfit, learning features that don't generalize well. When you take a LoRA adapter trained on one task and move it to another, especially one in a different domain, performance can drop significantly.

Our research shows this trade-off clearly:

| Method | Source Task (CIFAR-100 Classification) | Transfer Task (Tiny ImageNet Classification) |
| :--- | :---: | :---: |
| LoRA | **92.48%** (Excellent Specialization) | 71.04% (Poor Generalization) |
| KoRA | 83.96% (Controlled Specialization) | **97.37%** (Superior Generalization) |

This raises a critical question: Can we create an adapter that learns more fundamental, transferable knowledge, even if it means sacrificing a few points on the source task?

---

## 💡 The Core Idea: From Independent Specialists to a Coordinated Team

The key limitation of LoRA is that its adapters operate in **complete isolation**. The query, key, and value adapters in a transformer block never communicate with each other.

**KoRA changes this paradigm.** Inspired by the Kolmogorov-Arnold Representation Theorem, which states that complex functions can be decomposed into compositions of simpler ones, KoRA introduces a learnable **`CompositionBlock`**.



Think of it like this:
* **LoRA** is a team of three brilliant specialists who write their recommendations on separate notes without ever talking.
* **KoRA** is a team where the same specialists first present their findings to a manager (the `CompositionBlock`). This manager then synthesizes their input into a single, holistic, and more intelligent final decision.

This `CompositionBlock` creates a **functional dependency** between the adapters, forcing them to learn a shared, compositional representation of the task.

---

## 🔧 How It Works

The architecture is a simple but powerful extension of LoRA.

1.  **Generate Low-Rank Signals:** For a given input `x`, we use standard LoRA adapters to compute the initial update deltas for Query, Key, and Value: $\delta_q, \delta_k, \delta_v$.
2.  **Compose the Signals:** Instead of applying these deltas directly, we concatenate them and feed them into a small, trainable MLP—the `CompositionBlock`.
    $$\delta_{comp} = \text{CompositionBlock}([\delta_q ; \delta_k ; \delta_v])$$
3.  **Apply a Holistic Update:** This single, composed delta $\delta_{comp}$ is then applied to the model. To ensure optimization stability, we introduce a **learnable gate `g`**, initialized to zero.
    $$v_{new} = v_{orig} + g \cdot \delta_{comp}$$

This gating mechanism is theoretically grounded and crucial. It allows the model to start from a stable state and learn to "turn up the volume" on the compositional signal only when it's proven to be useful for minimizing the loss.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* Transformers
* `timm`

### Installation
```bash
git clone [https://github.com/onepunchmonk/kora.git](https://github.com/onepunchmonk/kora.git)
cd kora
pip install -r requirements.txt
