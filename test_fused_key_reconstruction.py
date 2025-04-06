"""We want triton==3.0.0 for this script
"""

import torch
import triton
import triton.language as tl

@triton.jit
def get_freq_multi_tokens(starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    DIM_2: tl.constexpr = 64
    freqs = tl.arange(0, DIM_2) * 2
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.fast_powf(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(freqs), tl.extra.cuda.libdevice.fast_sinf(freqs)


def get_configs():
    configs = []
    for block_l in [32, 64, 128]:
        for block_r in [16, 32, 64]:
            for num_warps in [1, 2, 4]:
                for num_stages in [1, 2, 3]:
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
):
    pid_b = tl.program_id(axis=0)  # batch size
    pid_hg = tl.program_id(axis=1)  # id of query head group
    pid_l = tl.program_id(axis=2)  # id of block along seq_length dimension

    #offs_qls = tl.arange(0, Q_LEN)
    offs_qls = tl.arange(0, 16) # NOTE: Triton have limitation that tl.dot need size >= 16
    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    #offs_ds = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_D), BLOCK_SIZE_D), BLOCK_SIZE_D) # same as offs_bds
    #offs_rs = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_R), BLOCK_SIZE_R), BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)

    Q_ptrs = q_ptr + pid_b * stride_q_b + (pid_hg * stride_q_g + offs_qls[:, None]*stride_q_lq + offs_ds[None, :]*stride_q_d)
    A_ptrs = A_ptr + pid_b * stride_A_b + (offs_ls[:, None]*stride_A_l + offs_rs[None, :]*stride_A_r)
    B_ptrs = B_ptr + pid_b * stride_B_b + (pid_hg * stride_B_g + offs_rs[:, None]*stride_B_r + offs_ds[None, :]*stride_B_d)
    O_ptrs = out_ptr + pid_b * stride_o_b + (pid_hg * stride_o_g + offs_qls[:, None]*stride_o_lq + offs_ls[None, :]*stride_o_lkv)

    # Fix BLOCK_SIZE_D = 64, and head_dim = 128
    ab_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    #ab_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)

    for _ in range(0, tl.cdiv(RANK, BLOCK_SIZE_R)):
        # Accumulate along R dimension.
        b_0 = tl.load(B_ptrs)
        # Load next block of A, B
        a = tl.load(A_ptrs, mask=offs_ls[:, None] < KV_LEN, other=0.0)
        ab_0 = tl.dot(a, b_0, ab_0)
        
        b_1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_B_d)
        ab_0 = tl.dot(a, b_1, ab_0)
        
        # Advance the pointers to next blocks
        A_ptrs += BLOCK_SIZE_R * stride_A_r
        B_ptrs += BLOCK_SIZE_R * stride_B_r

    ab_0 = ab_0.to(tl.float16)
    #ab_1 = ab_1.to(tl.float16)

    q1 = tl.load(Q_ptrs, mask=offs_qls[:, None] < Q_LEN, other=0.0)
    out_0 = tl.dot(q1, ab_0.T)
    q2 = tl.load(Q_ptrs + BLOCK_SIZE_D * stride_q_d, mask=offs_qls[:, None] < Q_LEN, other=0.0)
    out_1 = tl.dot(q2, ab_0.T)
    out = out_0 + out_1

    tl.store(O_ptrs, out, mask=((offs_qls[:, None] < Q_LEN) & (offs_ls[None, :] < KV_LEN)))


@triton.autotune(
    configs=get_configs(),
    key=["KV_LEN", "RANK", "HEAD_DIM"]
)
@triton.jit
def _qAB_fwd_v2(
    q_ptr, A_ptr, B_ptr, out_ptr,
    stride_q_b, stride_q_g, stride_q_lq, stride_q_d,
    stride_A_b, stride_A_l, stride_A_r,
    stride_B_b, stride_B_g, stride_B_r, stride_B_d,
    stride_o_b, stride_o_g, stride_o_lq, stride_o_lkv,
    Q_LEN, KV_LEN, RANK, HEAD_DIM,
    NUM_KV_HEADS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)  # batch size
    pid_hg = tl.program_id(axis=1)  # id of query head group
    pid_l = tl.program_id(axis=2)  # id of block along seq_length dimension

    #offs_qls = tl.arange(0, Q_LEN)
    offs_qls = tl.arange(0, 16) # NOTE: Triton have limitation that tl.dot need size >= 16
    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    #offs_ds = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_D), BLOCK_SIZE_D), BLOCK_SIZE_D) # same as offs_bds
    #offs_rs = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_R), BLOCK_SIZE_R), BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)

    Q_ptrs = q_ptr + pid_b * stride_q_b + (pid_hg * NUM_KV_HEADS_PER_GROUP * stride_q_g + offs_qls[:, None]*stride_q_lq + offs_ds[None, :]*stride_q_d)
    A_ptrs = A_ptr + pid_b * stride_A_b + (offs_ls[:, None]*stride_A_l + offs_rs[None, :]*stride_A_r)
    B_ptrs = B_ptr + pid_b * stride_B_b + (pid_hg * NUM_KV_HEADS_PER_GROUP * stride_B_g + offs_rs[:, None]*stride_B_r + offs_ds[None, :]*stride_B_d)
    O_ptrs = out_ptr + pid_b * stride_o_b + (pid_hg * NUM_KV_HEADS_PER_GROUP * stride_o_g + offs_qls[:, None]*stride_o_lq + offs_ls[None, :]*stride_o_lkv)

    for i in tl.static_range(NUM_KV_HEADS_PER_GROUP):
        # Fix BLOCK_SIZE_D = 64, and head_dim = 128
        ab_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
        ab_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
        for j in range(0, tl.cdiv(RANK, BLOCK_SIZE_R)):
            # Accumulate along R dimension.
            b_0 = tl.load(B_ptrs + j * BLOCK_SIZE_R * stride_B_r)
            # Load next block of A, B
            a = tl.load(A_ptrs + j * BLOCK_SIZE_R * stride_A_r, mask=offs_ls[:, None] < KV_LEN, other=0.0)
            ab_0 = tl.dot(a, b_0, ab_0)
            
            b_1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_B_d + j * BLOCK_SIZE_R * stride_B_r)
            ab_1 = tl.dot(a, b_1, ab_1)
            
            # # Advance the pointers to next blocks
            # A_ptrs += BLOCK_SIZE_R * stride_A_r
            # B_ptrs += BLOCK_SIZE_R * stride_B_r

        ab_0 = ab_0.to(tl.float16)
        ab_1 = ab_1.to(tl.float16)

        q1 = tl.load(Q_ptrs, mask=offs_qls[:, None] < Q_LEN, other=0.0)
        out_0 = tl.dot(q1, ab_0.T)
        q2 = tl.load(Q_ptrs + BLOCK_SIZE_D * stride_q_d, mask=offs_qls[:, None] < Q_LEN, other=0.0)
        out_1 = tl.dot(q2, ab_1.T)
        out = out_0 + out_1

        tl.store(O_ptrs, out, mask=((offs_qls[:, None] < Q_LEN) & (offs_ls[None, :] < KV_LEN)))

        Q_ptrs += stride_q_g
        B_ptrs += stride_B_g
        O_ptrs += stride_o_g


def qAB(q: torch.Tensor, Ak: torch.Tensor, Bk: torch.Tensor) -> torch.Tensor:
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
    grid = lambda META: (bsz, num_kv_heads, triton.cdiv(kv_len, META["BLOCK_SIZE_L"]))
    
    # Launch the kernel
    _qAB_fwd[grid](
        q_ptr=q, A_ptr=Ak, B_ptr=Bk, out_ptr=out_buffer,
        stride_q_b=stride_q_b, stride_q_g=stride_q_g, stride_q_lq=stride_q_lq, stride_q_d=stride_q_d,
        stride_A_b=stride_A_b, stride_A_l=stride_A_l, stride_A_r=stride_A_r,
        stride_B_b=stride_B_b, stride_B_g=stride_B_g, stride_B_r=stride_B_r, stride_B_d=stride_B_d,
        stride_o_b=stride_o_b, stride_o_g=stride_o_g, stride_o_lq=stride_o_lq, stride_o_lkv=stride_o_lkv,
        Q_LEN=num_query_heads_per_group, KV_LEN=kv_len, RANK=rank, HEAD_DIM=head_dim,
        #BLOCK_SIZE_R=32, BLOCK_SIZE_L=16,
        BLOCK_SIZE_D=64
    )
    
    # Reshape output to match expected format (bsz, num_heads, q_len, kv_len)
    return out_buffer.view(bsz, num_heads, q_len, kv_len)


def qAB_v2(q: torch.Tensor, Ak: torch.Tensor, Bk: torch.Tensor) -> torch.Tensor:
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
    grid = lambda META: (bsz, num_kv_heads // 4, triton.cdiv(kv_len, META["BLOCK_SIZE_L"]))
    
    # Launch the kernel
    _qAB_fwd_v2[grid](
        q_ptr=q, A_ptr=Ak, B_ptr=Bk, out_ptr=out_buffer,
        stride_q_b=stride_q_b, stride_q_g=stride_q_g, stride_q_lq=stride_q_lq, stride_q_d=stride_q_d,
        stride_A_b=stride_A_b, stride_A_l=stride_A_l, stride_A_r=stride_A_r,
        stride_B_b=stride_B_b, stride_B_g=stride_B_g, stride_B_r=stride_B_r, stride_B_d=stride_B_d,
        stride_o_b=stride_o_b, stride_o_g=stride_o_g, stride_o_lq=stride_o_lq, stride_o_lkv=stride_o_lkv,
        Q_LEN=num_query_heads_per_group, KV_LEN=kv_len, RANK=rank, HEAD_DIM=head_dim,
        #BLOCK_SIZE_R=32, BLOCK_SIZE_L=16,
        BLOCK_SIZE_D=64,
        NUM_KV_HEADS_PER_GROUP=4
    )
    
    # Reshape output to match expected format (bsz, num_heads, q_len, kv_len)
    return out_buffer.view(bsz, num_heads, q_len, kv_len)


def qAB_ref(q: torch.Tensor, Ak: torch.Tensor, Bk: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, q_len, head_dim = q.shape
    bsz_a, kv_len, rank = Ak.shape
    bsz_b, num_kv_heads, rank, head_dim = Bk.shape
    
    num_query_heads_per_group = num_heads // num_kv_heads
    assert bsz == bsz_a == bsz_b, "Batch size mismatch"

    q = q.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    
    # multiply Ak and Bk
    # Reconstruct the key matrix by multiplying Ak and Bk
    # Ak: (bsz, kv_len, rank)
    # Bk: (bsz, num_kv_heads, rank, head_dim)
    # First compute Ak @ Bk to get reconstructed keys
    reconstructed_k = torch.matmul(Ak.unsqueeze(1), Bk)  # (bsz, num_kv_heads, kv_len, head_dim)
    
    Bk_reshaped = Bk.transpose(1, 2).reshape(bsz, rank, -1)
    reconstructed_k_v2 = torch.matmul(Ak, Bk_reshaped)
    reconstructed_k_v2 = reconstructed_k_v2.reshape(bsz, kv_len, num_kv_heads, head_dim).transpose(1, 2)


    # Now compute q @ reconstructed_k.transpose(-1, -2)
    # q: (bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim)
    # reconstructed_k: (bsz, num_kv_heads, kv_len, head_dim)
    out = torch.matmul(q, reconstructed_k.transpose(-1, -2))  # (bsz, num_kv_heads, num_query_heads_per_group*q_len, kv_len)

    return out.view(bsz, num_heads, q_len, kv_len)

def qK(q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, q_len, head_dim = q.shape
    bsz, num_kv_heads, kv_len, head_dim = K.shape

    num_query_heads_per_group = num_heads // num_kv_heads
    q = q.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    K = K.view(bsz, num_kv_heads, kv_len, head_dim).contiguous()

    out = torch.matmul(q, K.transpose(-1, -2))
    return out.view(bsz, num_heads, q_len, kv_len)
    

@torch.no_grad()
def run_qk_qab_benchmark(
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    rank=192,
    dtype=torch.float16,
    device="cuda",
):
    configs = []
    # Test different sequence lengths
    configs.append(
        triton.testing.Benchmark(
            x_names=['kv_len'],  # What we're varying on x-axis
            x_vals=[2**i for i in range(16, 18)],  # Testing powers of 2 from 4K to 128K
            line_arg='operation',  # Different lines for different operations
            line_vals=['qk', 'qab', 'qab_ref'],
            line_names=['qK (Standard)', 'qAB (Triton)', 'qAB_ref (Eager)'],
            styles=[('red', '--'), ('blue', '-'), ('green', ':')],
            ylabel='ms',
            plot_name=f'qk-qab-comparison-h{num_heads}-d{head_dim}-r{rank}',
            args={
                'dtype': dtype,
                'num_heads': num_heads,
                'num_kv_heads': num_kv_heads,
                'head_dim': head_dim,
                'rank': rank,
                'device': device,
            },
        )
    )

    @triton.testing.perf_report(configs)
    def bench_qk_qab(kv_len, operation, num_heads, num_kv_heads, head_dim, rank, dtype, device):
        # Fixed parameters
        bsz = 1
        q_len = 1
        
        # Create input tensors
        q = torch.randn(bsz, num_heads, q_len, head_dim).to(dtype).to(device)
        K = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(dtype).to(device)
        
        # For factorized approach
        Ak = torch.randn(bsz, kv_len, rank).to(dtype).to(device)
        Bk = torch.randn(bsz, num_kv_heads, rank, head_dim).to(dtype).to(device)

        quantiles = [0.5, 0.2, 0.8]
        warmup = 25
        rep = 100

        if operation == 'qk':
            # Standard qK operation
            def fn():
                return qK(q, K)
        elif operation == 'qab':
            # Triton qAB operation
            def fn():
                return qAB(q, Ak, Bk)
        else:  # qab_ref
            # Reference qAB operation
            def fn():
                return qAB_ref(q, Ak, Bk)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms, min_ms, max_ms

    # Run benchmarks and save results
    bench_qk_qab.run(print_data=True, show_plots=True, save_path='results/')

if __name__ == "__main__":
    q_len = 1
    bsz = 1
    num_heads = 32
    num_kv_heads = 8
    rank = 192
    head_dim = 128
    kv_len = 65536

    q = torch.randn(bsz, num_heads, q_len, head_dim).to(torch.float16).cuda()
    Ak = torch.randn(bsz, kv_len, rank).to(torch.float16).cuda()
    Bk = torch.randn(bsz, num_kv_heads, rank, head_dim).to(torch.float16).cuda()
    K = torch.randn(bsz, num_kv_heads, kv_len, head_dim).to(torch.float16).cuda()

    out = qAB(q, Ak, Bk)
    out_ref = qAB_ref(q, Ak, Bk)
    out_v2 = qAB_v2(q, Ak, Bk)
    print(out - out_ref)
    print(out_v2 - out_ref)
    # print(torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2))

    range = torch.cuda.nvtx.range_start("qAB_benchmark")
    qAB(q, Ak, Bk)
    qAB_v2(q, Ak, Bk)
    torch.cuda.nvtx.range_pop()

    # Run the benchmark
    #run_qk_qab_benchmark()
    
