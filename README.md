# fused-linear-cross-entropy

[![CI](https://github.com/kishorhange111/fused-linear-cross-entropy/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/fused-linear-cross-entropy/actions/workflows/ci.yml)

In LLM training the single largest activation is often the **logits**. For 8,192 tokens
and Qwen2.5's 151,936-token vocabulary they take 2.5 GB in fp16, 5 GB once upcast to fp32
for the softmax, and as much again for their gradient. This repo fuses the LM head and the
cross-entropy into one autograd function that processes **chunks of tokens**. For each
chunk it computes the logits, the loss, and the gradients `∂L/∂h` and `∂L/∂W` straight
away, then drops the logits. Peak extra memory falls from O(n·V) to O(chunk·V). This is
the idea behind Liger Kernel's fused linear cross-entropy and Apple's Cut Cross-Entropy.

## How ([`loss.py`](src/flce/loss.py))

For a chunk with logits `z = h Wᵀ`, the gradient of the loss is available in closed form:

    ∂L/∂z = softmax(z)·(1 + 2λ·lse) − (1 − ε)·onehot(y) − ε/V      (masked, ÷ n_valid)
    ∂L/∂h = (∂L/∂z) W,     ∂L/∂W += (∂L/∂z)ᵀ h

The forward pass therefore does the backward pass's work chunk by chunk, keeps only `dh`
(n × d) and `dW` (V × d, accumulated in fp32), and the real backward just scales them by
the upstream gradient. It supports `ignore_index` (the mean runs over valid tokens), label
smoothing ε with PyTorch's definition, PaLM's z-loss λ·lse², "sum" or "mean" reduction,
frozen heads (no `dW` is computed) and tied embeddings.

## Tests

There are 18, on CPU, in fp64, all against the naive `cross_entropy(h @ Wᵀ)`:
- loss and both gradients at chunk sizes 1, 5, 16, 37 (= n) and 100 (> n), for both
  reductions;
- label smoothing and z-loss;
- the upstream gradient scale;
- `torch.autograd.gradcheck`;
- an all-ignored batch;
- the no-grad path and 3-D inputs;
- a frozen head;
- **tied embeddings**, where autograd must add the head's gradient to the embedding's.

```
pytest -q                            # 18 tests, CPU
python -m flce.bench --out results   # T4: op-level memory/time + Qwen2.5-0.5B max batch
```

**Results are pending** from the Colab run.
