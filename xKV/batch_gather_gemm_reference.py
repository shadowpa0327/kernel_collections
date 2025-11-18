"""
Reference implementation of batch_gather_gemm using pure PyTorch operations.
This helps understand what the CUTLASS kernel is doing under the hood.
"""

import torch
import torch.nn.functional as F


def apply_rotary_pos_emb_reference(x, cos_sin, position_ids):
    """
    Apply rotary position embedding to x using cos_sin cache and position_ids.

    This matches the CUDA kernel implementation in rope_new.cu.

    Args:
        x: [batch_size, heads, seq_len, head_dim]
        cos_sin: [max_seq_len, head_dim] where first half is cos, second half is sin
                 Format: [cos_0, ..., cos_{head_dim/2-1}, sin_0, ..., sin_{head_dim/2-1}]
        position_ids: [batch_size, heads, seq_len] - position indices for each element

    Returns:
        x with RoPE applied: [batch_size, heads, seq_len, head_dim]
    """
    batch_size, heads, seq_len, head_dim = x.shape
    half_dim = head_dim // 2

    # Gather cos_sin from cache using position_ids
    # Flatten position_ids for gathering
    position_ids_flat = position_ids.reshape(-1)  # [batch_size * heads * seq_len]

    # Gather cos_sin: [batch_size * heads * seq_len, head_dim]
    cos_sin_gathered = cos_sin[position_ids_flat]

    # Reshape back: [batch_size, heads, seq_len, head_dim]
    cos_sin_gathered = cos_sin_gathered.reshape(batch_size, heads, seq_len, head_dim)

    # Split into cos and sin based on CUDA kernel format
    # cos_sin format: [cos_0, ..., cos_{half_dim-1}, sin_0, ..., sin_{half_dim-1}]
    cos = cos_sin_gathered[..., :half_dim]  # First half
    sin = cos_sin_gathered[..., half_dim:]  # Second half

    # Split x into two halves
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Apply RoPE formula matching CUDA kernel:
    # output[tid] = x1 * cos - x2 * sin
    # output[tid + half_dim] = x2 * cos + x1 * sin
    output_first_half = x1 * cos - x2 * sin
    output_second_half = x2 * cos + x1 * sin

    # Concatenate results
    x_embed = torch.cat([output_first_half, output_second_half], dim=-1)

    return x_embed


def batch_gather_gemm_reference(
    a,              # [batch_size, seq_len, rank]
    b,              # [batch_size, heads, head_dim, rank]
    cos_sin,        # [max_seq_len, head_dim] - RoPE cache (cos and sin are same in kernel)
    position_ids,   # [batch_size, heads, num_chunks] - int32, indices for gathering
    chunk_size,     # int - size of each chunk
    apply_rope=True # whether to apply RoPE
):
    """
    Reference implementation of batch_gather_gemm.

    The operation performs:
    1. Gather rows from 'a' based on position_ids (at chunk granularity)
    2. Perform batched GEMM: gathered_a @ b^T
    3. (Optional) Apply RoPE using cos_sin cache

    Args:
        a: [batch_size, seq_len, rank] - input matrix A (to be gathered)
        b: [batch_size, heads, head_dim, rank] - input matrix B
        cos_sin: [max_seq_len, head_dim] - RoPE cache
        position_ids: [batch_size, heads, num_chunks] - gather indices (int32)
        chunk_size: int - how many consecutive rows form a chunk
        apply_rope: bool - whether to apply rotary position embedding

    Returns:
        output: [batch_size, heads, sparse_budget, head_dim]
        where sparse_budget = num_chunks * chunk_size
    """
    batch_size, seq_len, rank = a.shape
    _, heads, head_dim, _ = b.shape
    _, _, num_chunks = position_ids.shape
    sparse_budget = num_chunks * chunk_size

    print(f"\n{'='*80}")
    print(f"batch_gather_gemm_reference")
    print(f"{'='*80}")
    print(f"Input shapes:")
    print(f"  a: {list(a.shape)} [batch_size={batch_size}, seq_len={seq_len}, rank={rank}]")
    print(f"  b: {list(b.shape)} [batch_size={batch_size}, heads={heads}, head_dim={head_dim}, rank={rank}]")
    print(f"  cos_sin: {list(cos_sin.shape)}")
    print(f"  position_ids: {list(position_ids.shape)} [batch_size={batch_size}, heads={heads}, num_chunks={num_chunks}]")
    print(f"  chunk_size: {chunk_size}")
    print(f"  sparse_budget: {sparse_budget} = num_chunks({num_chunks}) * chunk_size({chunk_size})")
    print(f"{'='*80}\n")

    # Step 1: Expand position_ids from chunk-level to element-level
    # position_ids: [batch_size, heads, num_chunks]
    # We need to expand each chunk index to chunk_size consecutive indices

    # Expand: [batch_size, heads, num_chunks, chunk_size]
    position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, chunk_size)

    # Create offsets within each chunk: [0, 1, 2, ..., chunk_size-1]
    chunk_offsets = torch.arange(chunk_size, device=position_ids.device)

    # Add chunk_size to base position and add offsets
    # position_ids contains the starting position of each chunk
    position_ids_full = position_ids_expanded * chunk_size + chunk_offsets

    # Flatten to [batch_size, heads, sparse_budget]
    position_ids_full = position_ids_full.reshape(batch_size, heads, sparse_budget)

    print(f"Step 1: Expand position_ids from chunks to elements")
    print(f"  position_ids (chunk-level): {list(position_ids.shape)}")
    print(f"  position_ids_full (element-level): {list(position_ids_full.shape)}")
    print(f"  Example position_ids[0,0,:5]: {position_ids[0,0,:5].tolist() if position_ids.shape[-1] >= 5 else position_ids[0,0,:].tolist()}")
    print(f"  Example position_ids_full[0,0,:10]: {position_ids_full[0,0,:10].tolist()}\n")

    # Step 2: Gather rows from 'a' based on position_ids_full
    # a: [batch_size, seq_len, rank]
    # We need to gather for each batch and head combination

    # For gathering, we need to expand 'a' to match the batch dimension
    # a is shared across heads, so we expand it
    a_expanded = a.unsqueeze(1).expand(batch_size, heads, seq_len, rank)
    # a_expanded: [batch_size, heads, seq_len, rank]

    # Gather using position_ids_full
    # position_ids_full: [batch_size, heads, sparse_budget]
    # We need to gather along the seq_len dimension (dim=2)

    # Expand position_ids for gathering along rank dimension
    position_ids_gather = position_ids_full.unsqueeze(-1).expand(-1, -1, -1, rank)
    # position_ids_gather: [batch_size, heads, sparse_budget, rank]

    # Gather from a_expanded
    a_gathered = torch.gather(a_expanded, dim=2, index=position_ids_gather)
    # a_gathered: [batch_size, heads, sparse_budget, rank]

    print(f"Step 2: Gather rows from A")
    print(f"  a_expanded: {list(a_expanded.shape)}")
    print(f"  position_ids_gather: {list(position_ids_gather.shape)}")
    print(f"  a_gathered: {list(a_gathered.shape)}\n")

    # Step 3: Perform batched GEMM: a_gathered @ b^T
    # a_gathered: [batch_size, heads, sparse_budget, rank]
    # b: [batch_size, heads, head_dim, rank]
    # We want: [batch_size, heads, sparse_budget, head_dim]

    # b needs to be transposed: [batch_size, heads, rank, head_dim]
    b_transposed = b.transpose(-2, -1)

    # Perform batched matrix multiplication
    output = torch.matmul(a_gathered, b_transposed)
    # output: [batch_size, heads, sparse_budget, head_dim]

    print(f"Step 3: Perform batched GEMM")
    print(f"  a_gathered: {list(a_gathered.shape)}")
    print(f"  b_transposed: {list(b_transposed.shape)}")
    print(f"  output (before RoPE): {list(output.shape)}\n")

    # Step 4: (Optional) Apply RoPE
    if apply_rope:
        # Apply RoPE using the gathered position indices
        output = apply_rotary_pos_emb_reference(output, cos_sin, position_ids_full)
        print(f"Step 4: Apply RoPE")
        print(f"  output (after RoPE): {list(output.shape)}\n")

    print(f"{'='*80}")
    print(f"Final output shape: {list(output.shape)}")
    print(f"{'='*80}\n")

    return output


def test_batch_gather_gemm():
    """
    Test the reference implementation with a simple example.
    """
    print("\n" + "="*80)
    print("Testing batch_gather_gemm_reference")
    print("="*80 + "\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define dimensions
    batch_size = 2
    heads = 4
    seq_len = 128
    rank = 64
    head_dim = 128
    chunk_size = 8
    num_chunks = 10
    max_seq_len = 256

    # Create input tensors
    a = torch.randn(batch_size, seq_len, rank, dtype=torch.float32)
    b = torch.randn(batch_size, heads, head_dim, rank, dtype=torch.float32)
    cos_sin = torch.randn(max_seq_len, head_dim, dtype=torch.float32)

    # Create position_ids: randomly sample chunk positions
    # Each value should be in range [0, seq_len // chunk_size)
    max_chunk_id = seq_len // chunk_size - 1
    position_ids = torch.randint(0, max_chunk_id + 1, (batch_size, heads, num_chunks), dtype=torch.int32)

    print(f"Test configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  heads: {heads}")
    print(f"  seq_len: {seq_len}")
    print(f"  rank: {rank}")
    print(f"  head_dim: {head_dim}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  num_chunks: {num_chunks}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  sparse_budget: {num_chunks * chunk_size}\n")

    # Run reference implementation
    output_with_rope = batch_gather_gemm_reference(
        a, b, cos_sin, position_ids, chunk_size, apply_rope=True
    )

    output_no_rope = batch_gather_gemm_reference(
        a, b, cos_sin, position_ids, chunk_size, apply_rope=False
    )

    print(f"\nTest completed successfully!")
    print(f"  Output with RoPE shape: {list(output_with_rope.shape)}")
    print(f"  Output without RoPE shape: {list(output_no_rope.shape)}")
    print(f"  Output dtype: {output_with_rope.dtype}")

    # Check that outputs are different when RoPE is applied
    diff = (output_with_rope - output_no_rope).abs().max().item()
    print(f"  Max difference (with vs without RoPE): {diff:.6f}")

    return output_with_rope, output_no_rope


def test_gather_logic_detailed():
    """
    Detailed test to understand the gather operation with chunk_size.
    """
    print("\n" + "="*80)
    print("Detailed test: Understanding gather with chunk_size")
    print("="*80 + "\n")

    # Simple example
    batch_size = 1
    heads = 1
    seq_len = 32
    rank = 8
    chunk_size = 4
    num_chunks = 3

    # Create simple input
    a = torch.arange(seq_len * rank, dtype=torch.float32).reshape(1, seq_len, rank)

    # Position ids: select chunks [2, 5, 7]
    # This means we select rows: [8-11, 20-23, 28-31] (since chunk_size=4)
    position_ids = torch.tensor([[[2, 5, 7]]], dtype=torch.int32)

    print(f"Input A shape: {list(a.shape)}")
    print(f"  A is organized as {seq_len} rows × {rank} columns")
    print(f"  chunk_size = {chunk_size}, so we have {seq_len // chunk_size} chunks\n")

    print(f"position_ids: {position_ids.squeeze().tolist()}")
    print(f"  This selects chunk 2, 5, and 7")
    print(f"  Chunk 2 contains rows [8, 9, 10, 11]")
    print(f"  Chunk 5 contains rows [20, 21, 22, 23]")
    print(f"  Chunk 7 contains rows [28, 29, 30, 31]\n")

    # Expand position_ids
    position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, chunk_size)
    chunk_offsets = torch.arange(chunk_size, device=position_ids.device)
    position_ids_full = position_ids_expanded * chunk_size + chunk_offsets
    position_ids_full = position_ids_full.reshape(batch_size, heads, -1)

    print(f"Expanded position_ids (element-level):")
    print(f"  {position_ids_full.squeeze().tolist()}")
    print(f"  Shape: {list(position_ids_full.shape)}\n")

    # Show what rows we're gathering
    print(f"Gathering rows from A:")
    for i, row_idx in enumerate(position_ids_full.squeeze().tolist()):
        row_data = a[0, row_idx, :].tolist()
        print(f"  Row {row_idx:2d}: {row_data[:4]}... (showing first 4 elements)")

    print(f"\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run detailed gather test first
    test_gather_logic_detailed()

    # Run main test
    output_with_rope, output_no_rope = test_batch_gather_gemm()
