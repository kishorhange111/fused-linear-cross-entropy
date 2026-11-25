"""T4 benchmark: memory and time of the LM-head loss, then max batch for a real model.

    python -m flce.bench --out results

1. **Op level.** Qwen2.5-0.5B's head (d = 896, V = 151,936), fp16, forward + backward of the
   loss alone. The table reports peak *extra* memory above the inputs and the time, naive
   against fused at several chunk sizes, for 2,048 to 16,384 tokens.
2. **Training step.** Qwen2.5-0.5B, sequence 512, gradient checkpointing, fp16 autocast,
   SGD (so optimizer state doesn't hide the loss's memory). For each loss, the batch size is
   doubled until it OOMs. The table reports the largest batch that fits, its peak memory and
   its tokens/s.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .loss import fused_linear_cross_entropy, naive_linear_cross_entropy


def _peak(fn) -> tuple[float | None, float | None]:
    """(peak extra GiB, median ms) of fn(), or (None, None) on OOM."""
    torch.cuda.synchronize()
    try:
        fn()  # warm-up
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return peak, sorted(ts)[2]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def _op_row(h, W, y, chunks) -> dict:
    def step(fn):
        def run():
            h.grad = W.grad = None
            fn().backward()

        return run

    n, V = h.shape[0], W.shape[0]
    r = {"tokens": n, "logits_fp32_GiB": n * V * 4 / 2**30}
    r["naive_GiB"], r["naive_ms"] = _peak(step(lambda: naive_linear_cross_entropy(h, W, y)))
    for c in chunks:
        fused = lambda c=c: fused_linear_cross_entropy(h, W, y, chunk=c)
        r[f"fused{c}_GiB"], r[f"fused{c}_ms"] = _peak(step(fused))
    return r


def op_bench(d=896, V=151936, tokens=(2048, 4096, 8192, 16384), chunks=(512, 1024, 2048, 4096)):
    dev, dt = "cuda", torch.float16
    W = (torch.randn(V, d, device=dev, dtype=dt) * 0.02).requires_grad_()
    rows = []
    for n in tokens:
        h = torch.randn(n, d, device=dev, dtype=dt).requires_grad_()
        y = torch.randint(0, V, (n,), device=dev)
        rows.append(_op_row(h, W, y, chunks))
        print(json.dumps(rows[-1]), flush=True)
        del h, y
        W.grad = None
        torch.cuda.empty_cache()
    return rows


def train_bench(model_name="Qwen/Qwen2.5-0.5B", seq=512, chunk=2048):
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    head = model.get_output_embeddings().weight

    def loss_fn(kind, ids):
        with torch.autocast("cuda", torch.float16):
            hid = model.model(input_ids=ids).last_hidden_state[:, :-1]
            y = ids[:, 1:]
            if kind == "naive":
                return naive_linear_cross_entropy(hid, head.to(hid.dtype), y)
            return fused_linear_cross_entropy(hid, head.to(hid.dtype), y, chunk=chunk)

    rows = []
    for kind in ("naive", "fused"):
        best = None
        b = 1
        while b <= 256:
            ids = torch.randint(0, 150000, (b, seq), device=dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                for _ in range(2):  # warm-up + timed
                    t0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    loss_fn(kind, ids).backward()
                    opt.step()
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                best = {
                    "loss": kind,
                    "max_batch": b,
                    "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
                    "tokens_per_s": b * seq / dt,
                }
                print(json.dumps(best), flush=True)
                b *= 2
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
        rows.append(best or {"loss": kind, "max_batch": 0})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.cuda.get_device_name()
    ops = op_bench()
    train = train_bench()
    chunks = [k[5:-4] for k in ops[0] if k.startswith("fused") and k.endswith("_GiB")]
    f = lambda v, fmt: "OOM" if v is None else format(v, fmt)
    lines = [
        f"GPU **{dev}**. Head: d = 896, V = 151,936 (Qwen2.5-0.5B), fp16, loss forward + backward.",
        "",
        "### Peak extra memory (GiB) and time (ms)",
        "",
        "| tokens | fp32 logits alone | naive | "
        + " | ".join(f"fused c={c}" for c in chunks)
        + " |",
        "|---|---|---|" + "---|" * len(chunks),
    ]
    for r in ops:
        cells = [f"{f(r['naive_GiB'], '.2f')} / {f(r['naive_ms'], '.0f')}"] + [
            f"{f(r[f'fused{c}_GiB'], '.2f')} / {f(r[f'fused{c}_ms'], '.0f')}" for c in chunks
        ]
        lines.append(f"| {r['tokens']} | {r['logits_fp32_GiB']:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Qwen2.5-0.5B training step, seq 512, gradient checkpointing, fp16 autocast, SGD",
        "",
        "| loss | max batch (powers of 2) | peak GiB | tokens/s |",
        "|---|---|---|---|",
    ]
    for r in train:
        if r.get("max_batch"):
            tps = f"{r['tokens_per_s']:.0f}"
            lines.append(f"| {r['loss']} | {r['max_batch']} | {r['peak_GiB']:.2f} | {tps} |")
        else:
            lines.append(f"| {r['loss']} | OOM at 1 | – | – |")
    text = "\n".join(lines) + "\n"
    tag = dev.replace(" ", "_")
    with open(os.path.join(a.out, f"bench_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(a.out, f"bench_{tag}.json"), "w") as fh:
        json.dump({"ops": ops, "train": train}, fh, indent=1)
    print(text)


if __name__ == "__main__":
    main()
