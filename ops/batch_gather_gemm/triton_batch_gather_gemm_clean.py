"""
Triton implementation of batch_gather_gemm with RoPE support.

This module provides efficient Triton kernels for:
1. Batch gather GEMM: Sparse gather + batched matrix multiplication
2. RoPE application: Rotary Position Embedding with cache writing
3. Selective processing via cnts parameter
4. Auto-tuning support for optimal performance

Features:
- Clean, well-documented code with type hints
- Triton autotuning for GEMM and RoPE kernels
- Configurable block sizes and tuning parameters
- Correct RoPE position calculation (element-level from chunk-level)

Author: Clean implementation based on triton_batch_gather_gemm_fixed.py
"""

import torch
import triton
import triton.language as tl


def _ceil_div(a: int, b: int) -> int:
    """Compute ceiling division: ⌈a / b⌉"""
    return (a + b - 1) // b


# =============================================================================
# Auto-tuning configurations
# =============================================================================

# GEMM kernel configurations (auto-tuned on M, N, K dimensions)
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    # Higher-occupancy variants for larger K
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
]

# RoPE kernel configurations (auto-tuned on M, N dimensions)
ROPE_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64}, num_warps=8, num_stages=2),
]


# =============================================================================
# Kernel implementations
# =============================================================================

@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _batch_gather_gemm_kernel(
    # Input pointers
    A_ptr, B_ptr, PI_ptr, CNTS_ptr, C_ptr,
    # Shape parameters
    B_size, H_size, S, N, K, C_chunks, CHUNK, M,
    # Strides for A [batch_size, seq_len, rank]
    a_bs, a_s, a_k,
    # Strides for B [batch_size, heads, head_dim, rank]
    b_b, b_h, b_n, b_k,
    # Strides for position_ids [batch_size, heads, num_chunks]
    pi_b, pi_h, pi_c,
    # Strides for output C [batch_size, heads, sparse_budget, head_dim]
    c_b, c_h, c_m, c_n,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Batch gather GEMM kernel with selective chunk processing.

    Performs: C[b, h, :, :] = gathered_A[b, h, :, :] @ B[b, h, :, :]^T
    where gathered_A is gathered from A using position_ids at chunk granularity.

    Args:
        A_ptr: Input matrix A to gather from
        B_ptr: Input matrix B for GEMM
        PI_ptr: Position IDs (chunk-level indices)
        CNTS_ptr: Counts for selective processing (skip first cnts[bh] chunks)
        C_ptr: Output matrix C
        B_size: Batch size
        H_size: Number of heads
        S: Sequence length in A
        N: Head dimension
        K: Rank dimension
        C_chunks: Number of chunks
        CHUNK: Chunk size
        M: Sparse budget (= C_chunks * CHUNK)
    """
    # Program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # Decode batch and head indices
    b = pid_bh // H_size
    h = pid_bh % H_size

    # Compute row and column indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N

    # Load count for this batch-head and compute start position
    cnt = tl.load(CNTS_ptr + pid_bh, mask=True, other=0).to(tl.int32)
    start_row = cnt * CHUNK
    write_mask = m_mask & (rm >= start_row)

    # Compute chunk IDs and offsets within chunks
    chunk_id = rm // CHUNK
    r_in_chunk = rm - chunk_id * CHUNK

    # Load chunk positions from position_ids
    pi_ptr_bh = PI_ptr + b * pi_b + h * pi_h
    chunk_pos = tl.load(pi_ptr_bh + chunk_id * pi_c, mask=m_mask, other=0).to(tl.int32)

    # Compute absolute row indices in A
    row_idx = chunk_pos * CHUNK + r_in_chunk

    # Base pointers for this batch-head
    a_base = A_ptr + b * a_bs
    b_base = B_ptr + b * b_b + h * b_h
    c_base = C_ptr + b * c_b + h * c_h

    # Accumulator for GEMM
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-dimension loop (GEMM computation)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        # Load tile from A (gathered rows)
        a_ptrs = a_base + row_idx[:, None] * a_s + rk[None, :] * a_k
        a_tile = tl.load(a_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0).to(tl.bfloat16)

        # Load tile from B
        b_ptrs = b_base + rn[None, :] * b_n + rk[:, None] * b_k
        b_tile = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0).to(tl.bfloat16)

        # Accumulate: A @ B^T
        acc += tl.dot(a_tile, b_tile)

    # Store output (respecting cnts filtering)
    c_ptrs = c_base + rm[:, None] * c_m + rn[None, :] * c_n
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(write_mask[:, None] & n_mask[None, :]))


@triton.autotune(configs=ROPE_CONFIGS, key=['M', 'N'])
@triton.jit
def _rope_push_cache_kernel(
    # Input pointers
    X_ptr, PI_ptr, CNTS_ptr, COS_SIN_ptr, CACHE_ptr,
    # Shape parameters
    B_size, H_size, M, N, C_chunks, CHUNK, LMAX, SPARSE_START, SPARSE_END,
    # Strides for X [batch_size, heads, sparse_budget, head_dim]
    x_b, x_h, x_m, x_n,
    # Strides for position_ids [batch_size, heads, num_chunks]
    pi_b, pi_h, pi_c,
    # Strides for cache [batch_size, heads, max_seq_len, head_dim]
    cache_b, cache_h, cache_s, cache_d,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Apply RoPE and push results to cache with selective processing.

    Applies rotary position embedding to input X and writes results to cache.
    Respects cnts parameter to skip processing for chunks < cnts[bh].

    Args:
        X_ptr: Input tensor (output from GEMM)
        PI_ptr: Position IDs (chunk-level indices)
        CNTS_ptr: Counts for selective processing
        COS_SIN_ptr: Combined cos/sin cache [max_seq_len, head_dim]
                     Format: [cos_0...cos_{N/2-1}, sin_0...sin_{N/2-1}]
        CACHE_ptr: Output cache tensor
        B_size: Batch size
        H_size: Number of heads
        M: Sparse budget
        N: Head dimension
        C_chunks: Number of chunks
        CHUNK: Chunk size
        LMAX: Maximum sequence length
        SPARSE_START: Starting position in cache
        SPARSE_END: Ending position in cache
    """
    # Program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # Decode batch and head indices
    b = pid_bh // H_size
    h = pid_bh % H_size

    # Compute indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N

    # Load count and compute start position
    cnt = tl.load(CNTS_ptr + pid_bh, mask=True, other=0).to(tl.int32)
    start_row = cnt * CHUNK

    # Half dimension for RoPE
    half = N // 2
    rn_half = rn % half

    # Compute element-level positions (CRITICAL FIX)
    chunk_id = rm // CHUNK
    r_in_chunk = rm - chunk_id * CHUNK
    pi_ptr_bh = PI_ptr + b * pi_b + h * pi_h
    chunk_pos = tl.load(pi_ptr_bh + chunk_id * pi_c, mask=m_mask, other=0).to(tl.int32)
    actual_pos = chunk_pos * CHUNK + r_in_chunk  # Element-level position

    # Load input X
    x_base = X_ptr + b * x_b + h * x_h
    x_ptrs = x_base + rm[:, None] * x_m + rn[None, :] * x_n
    x = tl.load(x_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0).to(tl.bfloat16).to(tl.float32)

    # Compute rotate_half(x) for RoPE
    # rotate_half swaps and negates: [x1, x2] -> [-x2, x1]
    rn_rot = tl.where(rn < half, rn + half, rn - half)
    x_rot_ptrs = x_base + rm[:, None] * x_m + rn_rot[None, :] * x_n
    x_rot = tl.load(x_rot_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(rn < half, -1.0, 1.0)
    x_rot = x_rot * sign[None, :]

    # Load cos/sin from cache using actual_pos
    # COS_SIN layout: [LMAX, N] = [cos(0..half-1) | sin(0..half-1)]
    cos_ptrs = COS_SIN_ptr + actual_pos[:, None] * N + rn_half[None, :]
    sin_ptrs = COS_SIN_ptr + actual_pos[:, None] * N + (rn_half[None, :] + half)
    cos = tl.load(cos_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0).to(tl.bfloat16).to(tl.float32)
    sin = tl.load(sin_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0).to(tl.bfloat16).to(tl.float32)

    # Apply RoPE: x * cos + rotate_half(x) * sin
    y = x * cos + x_rot * sin

    # Write to cache (respecting start row, sparse bounds, and max length)
    sparse_rows = SPARSE_START + rm
    valid_row = (rm >= start_row) & (sparse_rows < SPARSE_END) & (sparse_rows < LMAX)
    cache_base = CACHE_ptr + b * cache_b + h * cache_h
    cache_ptrs = cache_base + sparse_rows[:, None] * cache_s + rn[None, :] * cache_d
    tl.store(cache_ptrs, y.to(tl.bfloat16), mask=(valid_row[:, None] & n_mask[None, :]))


def batch_gather_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_size: int,
    cnts: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Batch gather GEMM operation using Triton with auto-tuning.

    Performs sparse gather from A based on position_ids, then batched GEMM with B.
    Supports selective processing via cnts parameter.

    The kernel automatically tunes BLOCK_M, BLOCK_N, BLOCK_K based on the problem size (M, N, K).
    First call for a given shape will benchmark all configurations and cache the fastest.

    Args:
        a: Input matrix [batch_size, seq_len, rank], dtype=bfloat16
        b: Weight matrix [batch_size, heads, head_dim, rank] or [heads, head_dim, rank], dtype=bfloat16
        position_ids: Chunk indices [batch_size, heads, num_chunks], dtype=int32
        chunk_size: Size of each chunk
        cnts: Skip counts [batch_size * heads], dtype=int32
        out: Optional output buffer [batch_size, heads, sparse_budget, head_dim]

    Returns:
        Output tensor [batch_size, heads, sparse_budget, head_dim]
    """
    assert a.is_contiguous() and a.dtype == torch.bfloat16, "A must be contiguous bfloat16"

    B, S, K = a.shape

    # Handle B with or without batch dimension
    if b.dim() == 3:
        H, N, Kb = b.shape
        assert Kb == K, f"Rank mismatch: {Kb} != {K}"
        b = b.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    else:
        B2, H, N, Kb = b.shape
        assert B2 == B and Kb == K, f"Shape mismatch: B={B2} vs {B}, K={Kb} vs {K}"
        assert b.is_contiguous(), "B must be contiguous"

    # Validate position_ids
    assert position_ids.shape[:2] == (B, H), f"position_ids shape mismatch: {position_ids.shape[:2]} vs {(B, H)}"
    C = position_ids.shape[-1]
    M = C * chunk_size

    # Create output buffer if needed
    if out is None:
        out = torch.zeros((B, H, M, N), dtype=torch.bfloat16, device=a.device)
    else:
        assert out.shape == (B, H, M, N), f"Output shape mismatch: {out.shape} vs {(B, H, M, N)}"

    # Flatten cnts
    cnts_flat = cnts.reshape(-1)
    assert cnts_flat.numel() == B * H, f"cnts size mismatch: {cnts_flat.numel()} vs {B * H}"

    # Launch kernel with auto-tuning
    # Grid function receives META dict with selected BLOCK_M, BLOCK_N from autotuner
    grid = lambda META: (
        _ceil_div(M, META['BLOCK_M']),
        _ceil_div(N, META['BLOCK_N']),
        B * H
    )
    _batch_gather_gemm_kernel[grid](
        a, b, position_ids, cnts_flat, out,
        B, H, S, N, K, C, chunk_size, M,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        position_ids.stride(0), position_ids.stride(1), position_ids.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


def rope_push_cache_triton(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_size: int,
    cache: torch.Tensor,
    sparse_start: int,
    sparse_end: int,
    cnts: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE and write to cache using Triton with auto-tuning.

    The kernel automatically tunes BLOCK_M, BLOCK_N based on the problem size (M, N).
    First call for a given shape will benchmark all configurations and cache the fastest.

    Args:
        x: Input tensor [batch_size, heads, sparse_budget, head_dim]
        cos_sin: Combined cos/sin cache [max_seq_len, head_dim]
        position_ids: Chunk indices [batch_size, heads, num_chunks], dtype=int32
        chunk_size: Size of each chunk
        cache: Output cache [batch_size, heads, max_seq_len, head_dim]
        sparse_start: Starting row in cache
        sparse_end: Ending row in cache
        cnts: Skip counts [batch_size * heads], dtype=int32

    Returns:
        Updated cache tensor
    """
    B, H, M, N = x.shape
    LMAX = cos_sin.shape[0]
    C = position_ids.shape[-1]

    # Flatten cnts
    cnts_flat = cnts.reshape(-1)
    assert cnts_flat.numel() == B * H, f"cnts size mismatch"

    # Launch kernel with auto-tuning
    # Grid function receives META dict with selected BLOCK_M, BLOCK_N from autotuner
    grid = lambda META: (
        _ceil_div(M, META['BLOCK_M']),
        _ceil_div(N, META['BLOCK_N']),
        B * H
    )
    _rope_push_cache_kernel[grid](
        x, position_ids, cnts_flat, cos_sin, cache,
        B, H, M, N, C, chunk_size, LMAX, sparse_start, sparse_end,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        position_ids.stride(0), position_ids.stride(1), position_ids.stride(2),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
    )
    return cache


def batch_gather_gemm_rotary_pos_emb_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    output: torch.Tensor,
    chunk_size: int,
    cache: torch.Tensor,
    sparse_start: int,
    sparse_end: int,
    cnts: torch.Tensor,
    no_rope: bool = False,
) -> torch.Tensor:
    """
    Combined batch gather GEMM + RoPE operation with auto-tuning.

    High-level API that combines GEMM and RoPE operations. Both kernels are auto-tuned
    independently based on their respective problem sizes.

    Args:
        a: Input matrix [batch_size, seq_len, rank]
        b: Weight matrix [batch_size, heads, head_dim, rank]
        cos_sin: Combined cos/sin cache [max_seq_len, head_dim]
        position_ids: Chunk indices [batch_size, heads, num_chunks]
        output: Output buffer [batch_size, heads, sparse_budget, head_dim]
        chunk_size: Size of each chunk
        cache: Cache buffer [batch_size, heads, max_seq_len, head_dim]
        sparse_start: Starting row in cache
        sparse_end: Ending row in cache
        cnts: Skip counts [batch_size * heads]
        no_rope: If True, skip RoPE and return GEMM output directly

    Returns:
        If no_rope=True: GEMM output
        If no_rope=False: Cache with RoPE applied
    """
    # Run GEMM with auto-tuning
    batch_gather_gemm_triton(
        a, b, position_ids, chunk_size, cnts, out=output
    )

    # Optionally apply RoPE with auto-tuning
    if no_rope:
        return output
    else:
        return rope_push_cache_triton(
            output, cos_sin, position_ids, chunk_size, cache,
            sparse_start, sparse_end, cnts
        )


# Expose main APIs
__all__ = [
    'batch_gather_gemm_triton',
    'rope_push_cache_triton',
    'batch_gather_gemm_rotary_pos_emb_triton',
]