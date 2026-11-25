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

## Results (Tesla T4, Colab)

Raw numbers are in [`results/bench_Tesla_T4.json`](results/bench_Tesla_T4.json).

**The loss alone: Qwen2.5-0.5B head (d = 896, V = 151,936), fp16, forward + backward.**
Each cell gives the peak extra memory above the inputs (GiB) / the time (ms):

| tokens | fp32 logits alone | naive | fused c=512 | fused c=1024 | fused c=2048 | fused c=4096 |
|---|---|---|---|---|---|---|
| 2,048 | 1.16 | 3.22 / 115 | 1.16 / 215 | 1.70 / 192 | 2.57 / 183 | 2.57 / 184 |
| 4,096 | 2.32 | 6.69 / 252 | 1.16 / 427 | 1.70 / 384 | 3.15 / 366 | 4.89 / 380 |
| 8,192 | 4.64 | 13.64 / 551 | 1.16 / 845 | 1.70 / 749 | 3.15 / 709 | 6.05 / 722 |
| 16,384 | 9.27 | **OOM** | **1.16** / 1649 | 1.70 / 1471 | 3.15 / 1403 | 6.05 / 1447 |

- **The naive loss needs 2.8–2.9× the size of the fp32 logits** in extra memory, for the
  logits and their intermediates. That comes to 13.6 GiB at 8,192 tokens, and it
  runs out of memory at 16,384 on the 15 GB card.
- **The fused loss's memory depends on the chunk size, not the token count.** At c=512 it
  stays at 1.16 GiB from 2,048 to 16,384 tokens, 12× less than the naive loss at 8,192.
- **It costs time.** c=2048 is the fastest chunk size at every length, and it is 1.29× slower
  than the naive loss at 8,192 tokens and 1.59× slower at 2,048. Smaller chunks mean more,
  thinner matmuls: c=512 is 1.53× slower than the naive loss at 8,192.

**Qwen2.5-0.5B training step**: sequence 512, gradient checkpointing, fp16 autocast, SGD,
fused at c=2048. The batch size doubles until it OOMs:

| loss | max batch | peak GiB | tokens/s |
|---|---|---|---|
| naive | 4 | 9.38 | 4003 |
| fused | **8** | 12.91 | 3705 |

**The fused loss doubles the largest batch that fits** (4 → 8 sequences of 512). Throughput is 7%
lower at each loss's own largest batch, so on this card the gain is memory headroom, for
longer sequences or larger batches, not speed.
