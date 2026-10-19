"""ref_boost without a dense attention mask.

The reference-fidelity dial (ref_boost / ref_boost_a) is an additive log(b) logit bias
on the [target rows x reference columns] block of the [text | refs... | target] sequence.
Text and reference query rows get no bias, and every target row gets the SAME bias over
the key columns (constant per key column):

    bias[r, c] = t[r] * w[c]     t[r] = 1 for target rows, else 0
                                 w[c] = log(b_i) for c in reference block i (or its masked
                                        region), else 0

Handing that as a dense (L, L) bias to SDPA forces its masked kernels (Flash refuses a
mask, #6). Two mask-free forms of the SAME attention live here:

1. LSE merge (primary, CUDA bf16/fp16). Partition the key columns into groups G_j of
   equal w (at most one per boost value plus the unboosted rest). With o_j, lse_j the
   maskless attention of the target rows over G_j alone and its log-sum-exp,

       out_target = sum_j exp(lse_j + w_j) o_j / sum_j exp(lse_j + w_j)

   exactly (a softmax over the union of key groups, re-weighted per group). The groups
   together cover every key once, so the work is ONE maskless attention's worth; the
   non-target rows are an ordinary maskless call on the active ComfyUI backend. The
   log-sum-exp comes from the fused kernels' own outputs (cuDNN, else Flash).

2. Extra head dims (portable fallback, any SDPA device). A rank-1 additive bias is an
   inner product, so it fits in two extra head dims:

       q' = [q | t, t, 0...]   k' = [k | hi, lo, 0...]   v' = [v | 0...]
       q'.k' * scale = q.k * scale + t * (hi + lo) * scale = q.k * scale + bias

   with hi + lo = w / scale (hi rounded to the dtype, lo the residual — the bias is
   carried at ~2x the dtype's precision). head_dim is padded to the next multiple of 8
   (128 -> 136), the scale stays 1/sqrt(128) and the zero output columns are sliced off.
   One maskless SDPA call, but the fused kernels are slower at 136 than at 128.

This module has no ComfyUI import so it can be tested standalone.
"""
import math

import torch
import torch.nn.functional as F

# Extra head dims of the extra-dim form (hi + lo); the padded head_dim is rounded up to
# this multiple, the alignment SDPA's fused kernels require.
_BIAS_DIMS = 2
_HEAD_DIM_ALIGN = 8


def boost_column_logits(boosts, boost_mask, txtlen, slens, mask_hw, device):
    """The per-KEY-column log-boost w[c] over the [text | refs...] columns.

    boosts align with the reference blocks (last = subject); boost_mask (ComfyUI MASK, in
    the last reference's pixel space) restricts the last reference's boost to a region.
    Returns (w: float32 (rows0,), rows0 = first target row; target columns get no boost).
    This is THE column selection: the dense bias and both mask-free forms use it.
    """
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    w = torch.zeros(rows0, device=device, dtype=torch.float32)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_mask is not None and i == nsrc - 1 and mask_hw is not None:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        w[cols] = math.log(max(b, 1e-4))
    return w, rows0


def dense_bias_from_logits(w, rows0, tgtlen, dtype):
    """The original dense (1, 1, L, L) additive bias: w on the target rows, 0 elsewhere."""
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=w.device, dtype=dtype)
    bias[:, :, rows0:, :rows0] = w.to(dtype)
    return bias


# ---------------------------------------------------------------- 1. LSE merge

def key_groups(w, seqlen):
    """Partition the key columns [0, seqlen) by their log-boost: [(log_boost, cols)].

    cols is a slice when the group is one contiguous run (a view, no copy), else an index
    tensor. Columns >= len(w) (the target) are unboosted."""
    full = torch.zeros(seqlen, dtype=torch.float32)
    full[:w.shape[0]] = w.float().cpu()
    groups = []
    for value in torch.unique(full).tolist():
        idx = torch.nonzero(full == value, as_tuple=True)[0]
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        cols = slice(lo, hi) if hi - lo == idx.numel() else idx.to(w.device)
        groups.append((float(value), cols))
    return groups


def _cudnn_lse(q, k, v, scale):
    r = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q, k, v, None, True, 0.0, False, False, scale=scale)
    return r[0], r[1].reshape(q.shape[:3])  # lse (B, H, Lq, 1) -> (B, H, Lq)


def _flash_lse(q, k, v, scale):
    r = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=scale)
    return r[0], r[1]


# Tried in order; an op that raises once is not tried again in this process (an older
# torch without it, a GPU / dtype it does not support).
LSE_OPS = [("cudnn", _cudnn_lse), ("flash", _flash_lse)]
_disabled_lse_ops = set()


def attention_with_lse(q, k, v, scale, ops=None):
    """(out, lse) of maskless attention via the first fused kernel that runs, else None."""
    if q.device.type != "cuda" and ops is None:
        return None
    for name, op in (ops if ops is not None else LSE_OPS):
        if name in _disabled_lse_ops:
            continue
        try:
            return op(q, k, v, scale)
        except Exception as ex:  # noqa: BLE001
            _disabled_lse_ops.add(name)
            print(f"[krea2edit] ref_boost: {name} log-sum-exp attention unavailable "
                  f"({type(ex).__name__}); trying the next form", flush=True)
    return None


def boosted_rows_lse_merge(q, k, v, groups, scale, ops=None):
    """Attention of q (the target rows) over all keys with log-boost w_j on key group j,
    merged from one maskless call per group. None if no log-sum-exp kernel runs."""
    num = den = m = None
    q = q.contiguous()  # the fused kernels copy strided inputs per call otherwise
    for log_boost, cols in groups:
        if isinstance(cols, slice):
            kg, vg = k[:, :, cols], v[:, :, cols]
        else:
            kg, vg = k.index_select(2, cols), v.index_select(2, cols)
        res = attention_with_lse(q, kg, vg, scale, ops)
        if res is None:
            return None
        o, lse = res
        lse = lse.float().unsqueeze(-1) + log_boost       # (B, H, Lq, 1)
        if m is None:
            m, num, den = lse, o.float(), torch.ones_like(lse)
            continue
        m_new = torch.maximum(m, lse)
        a, b = torch.exp(m - m_new), torch.exp(lse - m_new)
        num = num * a + o.float() * b
        den = den * a + b
        m = m_new
    return (num / den).to(q.dtype)


# ---------------------------------------------------------------- 2. extra head dims

def padded_head_dim(dim_head):
    return -(-(dim_head + _BIAS_DIMS) // _HEAD_DIM_ALIGN) * _HEAD_DIM_ALIGN


def extend_qkv(q, k, v, w, rows0, scale):
    """q, k, v (B, H, L, D) -> (B, H, L, Dp) carrying the rank-1 bias (see module doc)."""
    B, H, L, D = q.shape
    pad = padded_head_dim(D) - D
    t = torch.zeros(L, device=q.device, dtype=q.dtype)
    t[rows0:] = 1
    target = torch.zeros(L, device=q.device, dtype=torch.float32)
    target[:rows0] = w / scale
    hi = target.to(k.dtype)
    lo = (target - hi.float()).to(k.dtype)
    q_ext = torch.zeros(L, pad, device=q.device, dtype=q.dtype)
    q_ext[:, 0] = t
    q_ext[:, 1] = t
    k_ext = torch.zeros(L, pad, device=k.device, dtype=k.dtype)
    k_ext[:, 0] = hi
    k_ext[:, 1] = lo
    qp = torch.cat([q, q_ext.expand(B, H, L, pad)], dim=-1)
    kp = torch.cat([k, k_ext.expand(B, k.shape[1], L, pad)], dim=-1)
    vp = F.pad(v, (0, pad))
    return qp, kp, vp


# ---------------------------------------------------------------- the override

def make_boost_override(w, rows0, seqlen, sdpa_attention, prev_override=None, lse_ops=None):
    """An `optimized_attention_override` computing the boosted block attention without a
    dense mask: the LSE merge when a log-sum-exp kernel runs, else the extra-dim form.

    w, rows0: from boost_column_logits(); seqlen: the [text | refs | target] length.
    sdpa_attention: an attention function with ComfyUI's optimized-attention signature
    that accepts head_dim 136 (ComfyUI's attention_pytorch; Sage / int8 kernels only
    take 64/96/128) — used by the extra-dim fallback. lse_ops: override LSE_OPS (tests).
    Calls that are not the boosted block attention (a mask already set, another layout
    or another sequence length) pass through untouched to the previous override/func.
    """
    call = prev_override or (lambda f, *a, **kw: f(*a, **kw))
    groups = key_groups(w, seqlen)

    def _attention(func, q, k, v, heads, mask=None, skip_reshape=False,
                   skip_output_reshape=False, **kwargs):
        if mask is not None or not skip_reshape or q.shape[-2] != seqlen:
            return call(func, q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)
        B, H, L, D = q.shape
        if k.shape[1] != H:  # unexpanded GQA kv (krea2 expands before the call today)
            rep = H // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        kwargs.pop("enable_gqa", None)
        scale = kwargs.pop("scale", None) or D ** -0.5
        tgt = boosted_rows_lse_merge(q[:, :, rows0:], k, v, groups, scale, lse_ops)
        if tgt is not None:
            # unboosted rows: an ordinary maskless call on the active backend
            out = torch.empty_like(q)
            out[:, :, :rows0] = call(func, q[:, :, :rows0], k, v, heads, mask=None,
                                     skip_reshape=True, skip_output_reshape=True,
                                     scale=scale, **kwargs)
            out[:, :, rows0:] = tgt
        else:
            qp, kp, vp = extend_qkv(q, k, v, w, rows0, scale)
            out = sdpa_attention(qp, kp, vp, heads, mask=None, skip_reshape=True,
                                 skip_output_reshape=True, scale=scale, **kwargs)[..., :D]
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(B, L, H * D)

    return _attention
