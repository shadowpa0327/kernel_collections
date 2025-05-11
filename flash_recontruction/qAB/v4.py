import torch
import triton
import triton.language as tl
from flash_recontruction.rope.sin_cos import _sin_cos

def get_configs():
    configs = []
    for block_l in [32, 64, 128]:
        for block_r in [16, 32, 64]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3, 4, 8]:
                    configs.append(
                        triton.Config({'BLOCK_SIZE_L': block_l, 'BLOCK_SIZE_R': block_r},
                                num_stages=num_stages, num_warps=num_warps))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["KV_LEN", "RANK", "HEAD_DIM"]
)
@triton.jit
def _qAB_fwd(
    q_ptr, A_ptr, B_ptr, out_ptr,
    stride_q_b, stride_q_g, stride_q_lq, stride_q_d,
    stride_A_b, stride_A_l, stride_A_r,
    stride_B_b, stride_B_g, stride_B_r, stride_B_d,
    stride_o_b, stride_o_g, stride_o_lq, stride_o_lkv,
    Q_LEN, KV_LEN, RANK, HEAD_DIM,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    THETA: tl.constexpr,
):
    pid_b = tl.program_id(axis=2)  # batch size
    pid_hg = tl.program_id(axis=1)  # id of query head group
    pid_l = tl.program_id(axis=0)  # id of block along seq_length dimension

    #offs_qls = tl.arange(0, Q_LEN)
    offs_qls = tl.arange(0, 16) # NOTE: Triton have limitation that tl.dot need size >= 16
    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)

    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)

    Q_ptrs = q_ptr + pid_b * stride_q_b + (pid_hg * stride_q_g + offs_qls[:, None]*stride_q_lq + offs_ds[None, :]*stride_q_d)
    A_ptrs = A_ptr + pid_b * stride_A_b + (offs_ls[:, None]*stride_A_l + offs_rs[None, :]*stride_A_r)
    B_ptrs = B_ptr + pid_b * stride_B_b + (pid_hg * stride_B_g + offs_rs[:, None]*stride_B_r + offs_ds[None, :]*stride_B_d)
    O_ptrs = out_ptr + pid_b * stride_o_b + (pid_hg * stride_o_g + offs_qls[:, None]*stride_o_lq + offs_ls[None, :]*stride_o_lkv)

    # Fix BLOCK_SIZE_D = 64, and head_dim = 128
    #ab_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    #ab_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)

    start_block = pid_l * BLOCK_SIZE_L
    cos, sin = _sin_cos(starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_L)
    out_0 = tl.zeros((16, BLOCK_SIZE_L), dtype=tl.float32)
    out_1 = tl.zeros((16, BLOCK_SIZE_L), dtype=tl.float32)
    q1 = tl.load(Q_ptrs, mask=offs_qls[:, None] < Q_LEN, other=0.0)
    q2 = tl.load(Q_ptrs + BLOCK_SIZE_D * stride_q_d, mask=offs_qls[:, None] < Q_LEN, other=0.0)
    for j in tl.static_range(2):
        for i in range(0, tl.cdiv(RANK, BLOCK_SIZE_R)):
            # Accumulate along R dimension.
            b = tl.load(B_ptrs + j * BLOCK_SIZE_D * stride_B_d + i * BLOCK_SIZE_R * stride_B_r)
            # Load next block of A, B
            a = tl.load(A_ptrs + i * BLOCK_SIZE_R * stride_A_r, mask=offs_ls[:, None] < KV_LEN, other=0.0)
            partial_ab = tl.dot(a, b)
            
            if j == 0:
                partial_ab_1 = partial_ab * cos
                partial_ab_2 = partial_ab * sin
            else:
                partial_ab_1 = partial_ab * -sin
                partial_ab_2 = partial_ab * cos
        
            out_0 = tl.dot(q1, (partial_ab_1.T).to(q1.dtype), out_0)
            out_1 = tl.dot(q2, (partial_ab_2.T).to(q2.dtype), out_1)

            # Advance the pointers to next blocks
            #A_ptrs += BLOCK_SIZE_R * stride_A_r
            #B_ptrs += BLOCK_SIZE_R * stride_B_r

    out = out_0 + out_1
    tl.store(O_ptrs, out, mask=((offs_qls[:, None] < Q_LEN) & (offs_ls[None, :] < KV_LEN)))

def qAB(q: torch.Tensor, Ak: torch.Tensor, Bk: torch.Tensor, theta: float = 10000000.0) -> torch.Tensor:
    """
    Computes the operation q @ (Ak @ Bk) using a custom Triton kernel.
    
    Args:
        q (torch.Tensor): Tensor of shape (bsz, num_heads, q_len, head_dim).
        Ak (torch.Tensor): Tensor of shape (bsz, kv_len, rank).
        Bk (torch.Tensor): Tensor of shape (bsz, num_kv_heads, rank, head_dim).
        
    Returns:
        torch.Tensor: Output tensor of shape (bsz, num_heads, q_len, kv_len).
    """
    assert q.dim() == 4, f"Expected q to be 4D, got {q.dim()}D"
    assert Ak.dim() == 3, f"Expected Ak to be 3D, got {Ak.dim()}D"
    assert Bk.dim() == 4, f"Expected Bk to be 4D, got {Bk.dim()}D"

    bsz, num_heads, q_len, head_dim = q.shape
    bsz_a, kv_len, rank = Ak.shape
    bsz_b, num_kv_heads, rank_b, head_dim_b = Bk.shape
    
    assert bsz == bsz_a == bsz_b, "Batch size mismatch"
    assert rank == rank_b, "Rank mismatch"
    assert head_dim == head_dim_b, "Head dimension mismatch"
    assert q_len == 1, "Only supporting q_len=1 for now"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    
    num_query_heads_per_group = num_heads // num_kv_heads  # number of query heads per group

    # Reshape q from (bsz, num_heads, q_len, head_dim) to (bsz, num_kv_heads, num_query_heads_per_group, q_len, head_dim)
    q = q.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    
    # Allocate output tensor
    out_buffer = torch.empty((bsz, num_kv_heads, num_query_heads_per_group*q_len, kv_len), 
                            dtype=q.dtype, device=q.device)
    # Get strides for tensors
    stride_q_b, stride_q_g, stride_q_lq, stride_q_d = q.stride()
    stride_A_b, stride_A_l, stride_A_r = Ak.stride()
    stride_B_b, stride_B_g, stride_B_r, stride_B_d = Bk.stride()
    stride_o_b, stride_o_g, stride_o_lq, stride_o_lkv = out_buffer.stride()
    

    # Define grid for kernel launch
    grid = lambda META: (triton.cdiv(kv_len, META["BLOCK_SIZE_L"]), num_kv_heads, bsz)
    
    # Launch the kernel
    _qAB_fwd[grid](
        q_ptr=q, A_ptr=Ak, B_ptr=Bk, out_ptr=out_buffer,
        stride_q_b=stride_q_b, stride_q_g=stride_q_g, stride_q_lq=stride_q_lq, stride_q_d=stride_q_d,
        stride_A_b=stride_A_b, stride_A_l=stride_A_l, stride_A_r=stride_A_r,
        stride_B_b=stride_B_b, stride_B_g=stride_B_g, stride_B_r=stride_B_r, stride_B_d=stride_B_d,
        stride_o_b=stride_o_b, stride_o_g=stride_o_g, stride_o_lq=stride_o_lq, stride_o_lkv=stride_o_lkv,
        Q_LEN=num_query_heads_per_group, KV_LEN=kv_len, RANK=rank, HEAD_DIM=head_dim,
        #BLOCK_SIZE_R=32, BLOCK_SIZE_L=16,
        BLOCK_SIZE_D=64,
        THETA=theta
    )
    
    # Reshape output to match expected format (bsz, num_heads, q_len, kv_len)
    return out_buffer.view(bsz, num_heads, q_len, kv_len)