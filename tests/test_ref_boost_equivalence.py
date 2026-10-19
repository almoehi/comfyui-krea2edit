"""Mask-free ref_boost attention == the original dense-bias masked attention.

Self-contained (no ComfyUI needed): loads ref_boost_attn.py by path and compares, on
random tensors with the real token layout [text | template | character | target]
scaled down, the original path (dense (L, L) bias -> masked SDPA) and both mask-free
forms of make_boost_override — the LSE merge (with a reference log-sum-exp op on CPU;
the real cuDNN / Flash ops when CUDA is available) and the extra-head-dim fallback —
against an exact float64 reference.

Run standalone:  python tests/test_ref_boost_equivalence.py
Or via pytest:   pytest tests/test_ref_boost_equivalence.py
"""
import importlib.util
import os

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "krea2edit_ref_boost_attn", os.path.join(os.path.dirname(_HERE), "ref_boost_attn.py"))
rba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rba)

# Real sheet layout scaled down ~1/40: text, template (ref A), character (ref B), target.
TXT, SLENS, TGT = 23, [144, 76], 144
H, D = 6, 128


def _sdpa(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kw):
    """ComfyUI attention_pytorch call shape (skip_reshape layout only)."""
    assert skip_reshape
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=kw.get("scale"))
    if skip_output_reshape:
        return out
    B, Hh, L, Dd = out.shape
    return out.transpose(1, 2).reshape(B, L, Hh * Dd)


def _ref_lse(q, k, v, scale):
    """Reference (out, lse) op standing in for the fused kernels on CPU."""
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype), torch.logsumexp(s, -1)


REF_OPS = [("reference", _ref_lse)]
FORMS = {"lse_merge": REF_OPS, "extra_dim": []}  # lse_ops=[] -> extra-dim fallback


def _exact(q, k, v, bias):
    s = (q.double() @ k.double().transpose(-1, -2)) * q.shape[-1] ** -0.5 + bias.double()
    return torch.softmax(s, -1) @ v.double()


def _qkv(dtype, kvheads=H, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = TXT + sum(SLENS) + TGT
    q = torch.randn(1, H, L, D, generator=g) * 2.0  # post-qknorm-ish magnitudes
    k = torch.randn(1, kvheads, L, D, generator=g) * 2.0
    v = torch.randn(1, kvheads, L, D, generator=g)
    return q.to(dtype), k.to(dtype), v.to(dtype), L


def _run(boosts, dtype, boost_mask=None, mask_hw=None, kvheads=H, skip_output_reshape=True,
         form="lse_merge"):
    q, k, v, L = _qkv(dtype, kvheads)
    w, rows0 = rba.boost_column_logits(boosts, boost_mask, TXT, SLENS, mask_hw, "cpu")
    bias = rba.dense_bias_from_logits(w, rows0, TGT, dtype)
    kx, vx = k, v
    if kvheads != H:
        kx = k.repeat_interleave(H // kvheads, 1)
        vx = v.repeat_interleave(H // kvheads, 1)
    exact = rba.dense_bias_from_logits(w, rows0, TGT, torch.float64)
    exact = _exact(q, kx, vx, exact)
    dense = _sdpa(q, kx, vx, H, mask=bias, skip_reshape=True, skip_output_reshape=True)
    ov = rba.make_boost_override(w, rows0, L, _sdpa, lse_ops=FORMS[form])
    xdim = ov(_sdpa, q, k, v, H, mask=None, skip_reshape=True,
              skip_output_reshape=skip_output_reshape)
    if not skip_output_reshape:
        xdim = xdim.reshape(1, L, H, D).transpose(1, 2)
    err = lambda a: (a.double() - exact).abs()  # noqa: E731
    e_dense, e_new = err(dense), err(xdim)
    return ((e_dense.max().item(), e_dense.mean().item()), (e_new.max().item(), e_new.mean().item()),
            (dense.double() - xdim.double()).abs().max().item())


def _check(boosts, dtype, **kw):
    for form in FORMS:
        res = _check_form(boosts, dtype, form=form, **kw)
    return res


def _check_form(boosts, dtype, **kw):
    (dmax, dmean), (nmax, nmean), diff = _run(boosts, dtype, **kw)
    info = (boosts, kw.get("form"), dmax, dmean, nmax, nmean, diff)
    if dtype == torch.float32:
        assert nmax < 1e-5 and diff < 1e-5, info
    else:
        # bf16: no worse than today's dense-bias path on average, and every element
        # within 2 bf16 ulps of the exact result (outputs are O(1): ulp = 2^-7..2^-6)
        assert nmean <= 1.25 * dmean + 1e-4, info
        assert nmax <= 2 ** -5, info
    return (dmax, dmean), (nmax, nmean), diff


SHEET_BOOSTS = [[2.5, 4.0], [5.0, 2.0], [4.0, 3.0]]  # [ref_boost_a, ref_boost] from the sheet


def test_sheet_boosts_fp32_and_bf16():
    for boosts in SHEET_BOOSTS:
        for dtype in (torch.float32, torch.bfloat16):
            _check(boosts, dtype)


def test_single_boosted_ref_and_boost_below_one():
    for boosts in ([1.0, 4.0], [2.5, 1.0], [0.5, 0.25], [0.0, 4.0]):
        for dtype in (torch.float32, torch.bfloat16):
            _check(boosts, dtype)


def test_ref_boost_mask_region():
    mask = torch.zeros(1, 76, 64)
    mask[:, 10:40, 20:50] = 1.0  # a "face" region on the subject reference
    for dtype in (torch.float32, torch.bfloat16):
        _check([2.5, 4.0], dtype, boost_mask=mask, mask_hw=[(12, 12), (4, 19)])


def test_unexpanded_gqa_and_flat_output_layout():
    _check([2.5, 4.0], torch.float32, kvheads=2)
    _check([2.5, 4.0], torch.float32, skip_output_reshape=False)


def test_dense_bias_is_the_original_construction():
    """dense_bias_from_logits == the pre-change _ref_attn_bias loop, bit for bit."""
    import math
    boosts, mask = [2.5, 4.0], torch.zeros(1, 76, 64)
    mask[:, 5:30, 10:60] = 1.0
    mask_hw = [(12, 12), (4, 19)]
    w, rows0 = rba.boost_column_logits(boosts, mask, TXT, SLENS, mask_hw, "cpu")
    new = rba.dense_bias_from_logits(w, rows0, TGT, torch.bfloat16)
    offs = [TXT, TXT + SLENS[0], TXT + sum(SLENS)]
    old = torch.zeros(1, 1, offs[-1] + TGT, offs[-1] + TGT, dtype=torch.bfloat16)
    m = F.interpolate(mask[None].float(), size=mask_hw[1], mode="area")[0, 0]
    cols_b = offs[1] + torch.nonzero(m.reshape(-1) > 0.5, as_tuple=True)[0]
    old[:, :, offs[-1]:, torch.arange(offs[0], offs[1])] = math.log(2.5)
    old[:, :, offs[-1]:, cols_b] = math.log(4.0)
    assert torch.equal(new, old)


def test_other_calls_pass_through_untouched():
    q, k, v, L = _qkv(torch.float32)
    w, rows0 = rba.boost_column_logits([2.5, 4.0], None, TXT, SLENS, None, "cpu")
    seen = []

    def func(*a, **kw):
        seen.append(kw.get("mask"))
        return "untouched"

    ov = rba.make_boost_override(w, rows0, L, _sdpa, lse_ops=REF_OPS)
    some_mask = torch.zeros(1, 1, L, L)
    assert ov(func, q, k, v, H, mask=some_mask, skip_reshape=True) == "untouched"
    assert ov(func, q[:, :, :50], k[:, :, :50], v[:, :, :50], H, skip_reshape=True) == "untouched"
    assert ov(func, q, k, v, H, skip_reshape=False) == "untouched"
    assert seen[0] is some_mask


def test_key_groups_cover_every_key_once():
    mask = torch.zeros(1, 76, 64)
    mask[:, 10:40, 20:50] = 1.0
    w, rows0 = rba.boost_column_logits([2.5, 4.0], mask, TXT, SLENS, [(12, 12), (4, 19)], "cpu")
    L = rows0 + TGT
    seen = torch.zeros(L, dtype=torch.long)
    full = torch.zeros(L)
    full[:rows0] = w
    for log_boost, cols in rba.key_groups(w, L):
        seen[cols] += 1
        assert torch.all(full[cols] == log_boost)
    assert torch.all(seen == 1)


def test_fused_lse_ops_on_cuda():
    """The real cuDNN / Flash log-sum-exp kernels, at bf16, when a GPU is present."""
    if not torch.cuda.is_available():
        return
    q, k, v, L = (t.cuda() if torch.is_tensor(t) else t for t in _qkv(torch.bfloat16))
    w, rows0 = rba.boost_column_logits([2.5, 4.0], None, TXT, SLENS, None, "cuda")
    exact = _exact(q, k, v, rba.dense_bias_from_logits(w, rows0, TGT, torch.float64))
    dense = _sdpa(q, k, v, H, mask=rba.dense_bias_from_logits(w, rows0, TGT, torch.bfloat16),
                  skip_reshape=True, skip_output_reshape=True)
    for name, op in rba.LSE_OPS:
        ov = rba.make_boost_override(w, rows0, L, _sdpa, lse_ops=[(name, op)])
        rba._disabled_lse_ops.discard(name)
        out = ov(_sdpa, q, k, v, H, mask=None, skip_reshape=True, skip_output_reshape=True)
        assert name not in rba._disabled_lse_ops, f"{name} op did not run"
        e_dense = (dense.double() - exact).abs().mean().item()
        e_new = (out.double() - exact).abs().mean().item()
        e_new_max = (out.double() - exact).abs().max().item()
        print(f"cuda {name}: mean|dense-exact|={e_dense:.2e} mean|lse-exact|={e_new:.2e} "
              f"max|lse-exact|={e_new_max:.2e}")
        assert e_new <= 1.5 * e_dense + 1e-4 and e_new_max <= 2 ** -5


if __name__ == "__main__":
    for boosts in SHEET_BOOSTS + [[0.5, 0.25]]:
        for dtype in (torch.float32, torch.bfloat16):
            for form in FORMS:
                (dmax, dmean), (nmax, nmean), diff = _check_form(boosts, dtype, form=form)
                print(f"boosts={boosts} {str(dtype):15s} {form:9s} dense-exact max/mean="
                      f"{dmax:.2e}/{dmean:.2e}  new-exact max/mean={nmax:.2e}/{nmean:.2e}  "
                      f"max|dense-new|={diff:.2e}")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
