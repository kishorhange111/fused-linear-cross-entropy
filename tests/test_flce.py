import pytest
import torch

from flce import fused_linear_cross_entropy as fused
from flce import naive_linear_cross_entropy as naive


def data(n=37, d=16, V=53, seed=0, ignore_frac=0.2):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, d, generator=g, dtype=torch.float64) * 2
    W = torch.randn(V, d, generator=g, dtype=torch.float64)
    y = torch.randint(0, V, (n,), generator=g)
    y[torch.rand(n, generator=g) < ignore_frac] = -100
    return h, W, y


def grads(fn, h, W, y, scale=1.0, **kw):
    h, W = h.clone().requires_grad_(), W.clone().requires_grad_()
    loss = fn(h, W, y, **kw)
    (loss * scale).backward()
    return loss.detach(), h.grad, W.grad


@pytest.mark.parametrize("chunk", [1, 5, 16, 37, 100])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_loss_and_grads_match_naive_for_any_chunk(chunk, reduction):
    h, W, y = data()
    a = grads(fused, h, W, y, chunk=chunk, reduction=reduction)
    b = grads(naive, h, W, y, reduction=reduction)
    for x, z in zip(a, b, strict=True):
        torch.testing.assert_close(x, z)


@pytest.mark.parametrize("eps,zl", [(0.1, 0.0), (0.0, 1e-3), (0.2, 1e-2)])
def test_label_smoothing_and_z_loss(eps, zl):
    h, W, y = data(seed=1)
    a = grads(fused, h, W, y, chunk=8, label_smoothing=eps, z_loss=zl)
    b = grads(naive, h, W, y, label_smoothing=eps, z_loss=zl)
    for x, z in zip(a, b, strict=True):
        torch.testing.assert_close(x, z)


def test_upstream_gradient_is_applied():
    h, W, y = data(seed=2)
    a = grads(fused, h, W, y, scale=3.5, chunk=7)
    b = grads(naive, h, W, y, scale=3.5)
    torch.testing.assert_close(a[1], b[1])
    torch.testing.assert_close(a[2], b[2])


def test_gradcheck():
    h, W, y = data(n=9, d=4, V=7, seed=3)
    h.requires_grad_()
    W.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda h, W: fused(h, W, y, chunk=4, label_smoothing=0.1, z_loss=0.01), (h, W)
    )


def test_all_ignored_gives_zero_loss_and_grad():
    h, W, y = data(seed=4)
    y[:] = -100
    loss, gh, gW = grads(fused, h, W, y, chunk=8)
    assert loss.item() == 0 and gh.abs().max() == 0 and gW.abs().max() == 0


def test_no_grad_path_and_3d_inputs():
    h, W, y = data(n=24, seed=5)
    with torch.no_grad():
        a = fused(h.view(4, 6, -1), W, y.view(4, 6), chunk=5)
    torch.testing.assert_close(a, naive(h, W, y))
    # frozen head: only h needs grad
    hh = h.clone().requires_grad_()
    fused(hh, W, y, chunk=5).backward()
    hn = h.clone().requires_grad_()
    naive(hn, W, y).backward()
    torch.testing.assert_close(hh.grad, hn.grad)


def test_tied_embedding_gets_both_gradients():
    """With tied weights W is used for the embedding and the head; autograd must sum both."""
    g = torch.Generator().manual_seed(6)
    E = torch.randn(30, 8, generator=g, dtype=torch.float64)
    ids = torch.randint(0, 30, (12,), generator=g)
    y = torch.randint(0, 30, (12,), generator=g)

    def run(fn):
        W = E.clone().requires_grad_()
        h = torch.tanh(W[ids] * 1.3)
        fn(h, W, y).backward()
        return W.grad

    torch.testing.assert_close(run(lambda h, W, y: fused(h, W, y, chunk=5)), run(naive))
