import triton
import triton.language as tl

@triton.jit
def _sin_cos(starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    DIM_2: tl.constexpr = 64
    freqs = tl.arange(0, DIM_2) * 2
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.fast_powf(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(freqs), tl.extra.cuda.libdevice.fast_sinf(freqs)




import math
import torch
import triton
import triton.language as tl

# ────────────────────────────────────────────────────────────────────────────────
# Your utility stays unchanged and is called from inside the bigger kernel
# ────────────────────────────────────────────────────────────────────────────────
@triton.jit
def _sin_cos(starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr   = 128          # hard-coded as in your snippet
    DIM_2: tl.constexpr = 64
    freqs = tl.arange(0, DIM_2) * 2
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.fast_powf(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return (
        tl.extra.cuda.libdevice.fast_cosf(freqs),
        tl.extra.cuda.libdevice.fast_sinf(freqs),
    )

# ────────────────────────────────────────────────────────────────────────────────
# One program ⇒ many rows
# ────────────────────────────────────────────────────────────────────────────────
@triton.jit
def _rotary_sin_cos_kernel_multi(
    sin_ptr, cos_ptr,                    # output tensors
    starting_idx,                        # offset added to every row index
    theta: tl.constexpr,                 # rotary base
    SEQLEN: tl.constexpr,                # full length of the sequence
    BLOCK_M: tl.constexpr                # rows handled by every program
):
    pid  = tl.program_id(axis=0)         # token-block index
    row0 = pid * BLOCK_M                 # first row produced by this program

    # Call your helper once for the whole block
    cos_blk, sin_blk = _sin_cos(row0 + starting_idx, theta, NB_TOKENS=BLOCK_M)

    # Mask in case the block straddles the end of the sequence
    valid_rows = row0 + tl.arange(0, BLOCK_M) < SEQLEN
    mask = valid_rows[:, None]           # broadcast to (BLOCK_M, 64)

    # Flat (row, col) → linear index in the output tensors
    DIM_2: tl.constexpr = 64             # keep identical to helper
    row_ptrs = (row0 + tl.arange(0, BLOCK_M))[:, None] * DIM_2
    col_ptrs = tl.arange(0, DIM_2)[None, :]
    ptrs     = row_ptrs + col_ptrs       # shape (BLOCK_M, 64)

    tl.store(sin_ptr + ptrs, sin_blk, mask=mask)
    tl.store(cos_ptr + ptrs, cos_blk, mask=mask)


# ────────────────────────────────────────────────────────────────────────────────
# Python-side convenience
# ────────────────────────────────────────────────────────────────────────────────
def rotary_sin_cos(
    seqlen: int,
    theta: float = 10_000.0,
    starting_idx: int = 0,
    *,
    block_size: int = 16,                # rows per program – tune for your GPU
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    stream: torch.cuda.Stream | None = None,
):
    """
    Compute rotary-embedding sin/cos tables with one Triton program handling
    `block_size` rows.  Works with the hard-coded 128-dim helper function above.
    """
    assert block_size > 0
    dim_2 = 64                           # must match helper's DIM_2

    sin = torch.empty((seqlen, dim_2), dtype=dtype, device=device)
    cos = torch.empty_like(sin)

    grid = (triton.cdiv(seqlen, block_size),)
    _rotary_sin_cos_kernel_multi[grid](
        sin, cos,
        starting_idx, theta,
        SEQLEN=seqlen,
        BLOCK_M=block_size,
    )
    return sin, cos
