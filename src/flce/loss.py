"""Fused LM head + cross-entropy, computed in token chunks so the logits never exist in full.

The naive loss is ``cross_entropy(h @ Wᵀ, y)``. For n tokens and vocabulary V it materialises
an (n, V) logits tensor, its fp32 upcast, and its gradient. At n = 8,192 and V = 151,936
(Qwen2.5) that is 2.5 GB in fp16 and 5 GB per fp32 copy, usually the largest activation in the
whole training step.

Here the forward pass walks over chunks of c tokens. For each chunk it:
1. computes logits ``z = h_c Wᵀ`` (c × V, upcast to fp32) and their log-sum-exp;
2. adds the chunk's loss;
3. *immediately* forms the gradient of the loss with respect to z,

       ∂L/∂z = softmax(z) − onehot(y)              (per token, then scaled)

   and turns it into the two gradients that are actually needed, ``∂L/∂h_c = g W`` and
   ``∂L/∂W += gᵀ h_c``.

The logits are then dropped, so peak extra memory is O(c·V) instead of O(n·V). The backward
pass only multiplies the stored gradients by the incoming scalar. This is the trick used
by Liger Kernel's fused linear cross-entropy. Cut Cross-Entropy goes further, never writing
even the chunk's logits to global memory.

Also supported, each with its exact gradient:
- ``ignore_index`` (masked tokens contribute nothing, and the mean is over valid tokens);
- ``label_smoothing`` ε, with loss ``(1−ε)(lse − z_y) + ε(lse − mean z)``, as in PyTorch;
- ``z_loss`` λ: adds ``λ·lse²`` (PaLM), which keeps the log-normaliser near 0.
"""

from __future__ import annotations

import torch


def _acc(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype: fp32 for half types, otherwise the input's own (e.g. fp64)."""
    return torch.promote_types(dtype, torch.float32)


def _chunk_grads(h, W, y, ignore_index, eps, z_loss, need_grad):
    """Loss sum and, if wanted, ∂(loss sum)/∂z for one chunk, both in the accumulation dtype."""
    z = (h @ W.T).to(_acc(h.dtype))  # (c, V)
    lse = torch.logsumexp(z, -1)
    valid = y != ignore_index
    ys = torch.where(valid, y, torch.zeros_like(y))
    z_y = z.gather(1, ys[:, None])[:, 0]
    V = z.shape[1]
    loss = (1 - eps) * (lse - z_y) + eps * (lse - z.mean(1)) + z_loss * lse**2
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    if not need_grad:
        return loss.sum(), None
    g = torch.softmax(z, -1)  # reuse z's memory footprint: z is dead after this line
    del z
    g *= (1 + 2 * z_loss * lse)[:, None]
    g.scatter_add_(1, ys[:, None], torch.full_like(lse, -(1 - eps))[:, None])
    g -= eps / V
    g *= valid[:, None].to(g.dtype)
    return loss.sum(), g


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, y, chunk, ignore_index, eps, z_loss, reduction):
        need_h, need_W = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need = need_h or need_W
        n = h.shape[0]
        n_valid = (y != ignore_index).sum()
        acc = _acc(h.dtype)
        denom = n_valid.clamp(min=1).to(acc) if reduction == "mean" else 1.0
        total = torch.zeros((), dtype=acc, device=h.device)
        dh = torch.empty_like(h) if need_h else None
        dW = torch.zeros(W.shape, dtype=acc, device=W.device) if need_W else None
        for s in range(0, n, chunk):
            hc, yc = h[s : s + chunk], y[s : s + chunk]
            ls, g = _chunk_grads(hc, W, yc, ignore_index, eps, z_loss, need)
            total += ls
            if need:
                g /= denom
                gd = g.to(h.dtype)
                del g
                if need_h:
                    dh[s : s + chunk] = gd @ W
                if need_W:
                    dW += (gd.T @ hc).to(acc)
        ctx.save_for_backward(dh if need_h else None, dW.to(W.dtype) if need_W else None)
        return total / denom

    @staticmethod
    def backward(ctx, grad_out):
        dh, dW = ctx.saved_tensors
        gh = dh * grad_out.to(dh.dtype) if dh is not None else None
        gW = dW * grad_out.to(dW.dtype) if dW is not None else None
        return gh, gW, None, None, None, None, None, None


def fused_linear_cross_entropy(
    h: torch.Tensor,
    W: torch.Tensor,
    y: torch.Tensor,
    chunk: int = 1024,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """``cross_entropy(h @ W.T, y)`` without materialising the logits.

    ``h`` is (..., d) and is flattened to (n, d); ``W`` is (V, d); ``y`` has h's leading shape.
    ``reduction`` is "mean" (over non-ignored tokens) or "sum".
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(reduction)
    h2 = h.reshape(-1, h.shape[-1])
    return _FusedLinearCE.apply(
        h2, W, y.reshape(-1), chunk, ignore_index, label_smoothing, z_loss, reduction
    )


def naive_linear_cross_entropy(
    h, W, y, ignore_index=-100, label_smoothing=0.0, z_loss=0.0, reduction="mean"
):
    """The reference: full logits, PyTorch's cross-entropy, plus the optional z-loss."""
    z = h.reshape(-1, h.shape[-1]) @ W.T
    z = z.to(torch.promote_types(z.dtype, torch.float32))
    y = y.reshape(-1)
    ce = torch.nn.functional.cross_entropy(
        z, y, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction="none"
    )
    if z_loss:
        valid = (y != ignore_index).to(z.dtype)
        ce = ce + z_loss * torch.logsumexp(z, -1) ** 2 * valid
    if reduction == "sum":
        return ce.sum()
    return ce.sum() / (y != ignore_index).sum().clamp(min=1)
