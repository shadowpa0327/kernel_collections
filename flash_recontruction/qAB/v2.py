import torch
import triton
import triton.language as tl

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

###############################################################################
# 2.  TRITON KERNEL – uses the pre-computed table                             #
###############################################################################

@triton.autotune(configs=get_configs(), key=["KV_LEN", "RANK", "HEAD_DIM"])
@triton.jit
def _qAB_fwd_precomp(
    # --- matrices ---
    q_ptr, A_ptr, B_ptr,  out_ptr,
    sincos_ptr,                        # <-- NEW: pre-computed table
    # --- strides ---
    stride_q_b, stride_q_g, stride_q_lq, stride_q_d,
    stride_A_b, stride_A_l, stride_A_r,
    stride_B_b, stride_B_g, stride_B_r, stride_B_d,
    stride_o_b, stride_o_g, stride_o_lq, stride_o_lkv,
    stride_s_l, stride_s_d,            # <-- NEW:   sincos strides
    # --- sizes ---
    Q_LEN, KV_LEN, RANK, HEAD_DIM,
    # --- meta-parameters ---
    BLOCK_SIZE_D: tl.constexpr,        # == head_dim // 2   (64 in your tests)
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    ###############
    #  grid ids   #
    ###############
    pid_b  = tl.program_id(axis=2)   # batch
    pid_hg = tl.program_id(axis=1)   # group of query heads
    pid_l  = tl.program_id(axis=0)   # block along KV sequence

    #########################
    #  offsets & pointers   #
    #########################
    offs_ds  = tl.arange(0, BLOCK_SIZE_D)                   # 0‥63
    offs_rs  = tl.arange(0, BLOCK_SIZE_R)
    offs_ls  = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_qls = tl.arange(0, 16) # NOTE: Triton have limitation that tl.dot need size >= 16
    # q
    q_ptrs = (q_ptr  + pid_b * stride_q_b
                      + pid_hg * stride_q_g
                      + offs_qls[:, None] * stride_q_lq
                      + offs_ds[None, :]      * stride_q_d)

    # A & B (same as before)
    A_ptrs = (A_ptr + pid_b * stride_A_b
                     + offs_ls[:, None] * stride_A_l
                     + offs_rs[None, :] * stride_A_r)

    B_ptrs = (B_ptr + pid_b * stride_B_b
                     + pid_hg * stride_B_g
                     + offs_rs[:, None] * stride_B_r
                     + offs_ds[None, :] * stride_B_d)

    O_ptrs = (out_ptr + pid_b * stride_o_b
                       + pid_hg * stride_o_g
                       + offs_qls[:, None] * stride_o_lq
                       + offs_ls[None, :]     * stride_o_lkv)

    ############################
    #  1)  reconstruct K = A·B #
    ############################
    ab_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    ab_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)

    for _ in range(0, tl.cdiv(RANK, BLOCK_SIZE_R)):
        a  = tl.load(A_ptrs, mask=offs_ls[:, None] < KV_LEN, other=0.0)
        b0 = tl.load(B_ptrs)
        ab_0 = tl.dot(a, b0, ab_0)

        b1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_B_d)
        ab_1 = tl.dot(a, b1, ab_1)

        A_ptrs += BLOCK_SIZE_R * stride_A_r
        B_ptrs += BLOCK_SIZE_R * stride_B_r

    # fp32->fp16
    ab_0, ab_1 = ab_0.to(tl.float16), ab_1.to(tl.float16)

    ############################################
    #  2)  load pre-computed sin / cos for RoPE #
    ############################################

    sin_ptrs = (sincos_ptr
                + offs_ls[:, None] * stride_s_l
                + offs_ds[None, :]* stride_s_d)

    cos_ptrs = sin_ptrs + (HEAD_DIM // 2) * stride_s_d   # second half of dim

    sin = tl.load(sin_ptrs)               # [L,D/2]
    cos = tl.load(cos_ptrs)               # [L,D/2]

    ###################################
    #  3)  apply rotary embedding     #
    ###################################
    ab_0_pe = ab_0 * cos - ab_1 * sin
    ab_1_pe = ab_1 * cos + ab_0 * sin

    ###################################
    #  4)  Q x Kᵀ                     #
    ###################################
    q0   = tl.load(q_ptrs, mask=offs_qls[:, None] < Q_LEN, other=0.0)
    out0 = tl.dot(q0, ab_0_pe.T).to(tl.float16)

    q1   = tl.load(q_ptrs + BLOCK_SIZE_D * stride_q_d,
                   mask=offs_qls[:, None] < Q_LEN, other=0.0)
    out1 = tl.dot(q1, ab_1_pe.T).to(tl.float16)

    tl.store(O_ptrs, out0 + out1,
             mask=((offs_qls[:, None] < Q_LEN) &
                   (offs_ls[None, :]      < KV_LEN)))


###############################################################################
# 3.  PYTHON WRAPPER – identical signature plus the sin_cos argument          #
###############################################################################

def qAB_precomp(q:   torch.Tensor,
                Ak:  torch.Tensor,
                Bk:  torch.Tensor,
                sin_cos: torch.Tensor,
            ) -> torch.Tensor:
    """
    Same contract as `qAB`, but uses a pre-computed sin/cos table of shape
    [kv_len, head_dim] where [:, :H/2] = sin, [:, H/2:] = cos.
    """
    # ---------- shape sanity ----------
    bsz, num_heads, q_len, head_dim = q.shape
    kv_len, head_dim_sc = sin_cos.shape
    assert head_dim_sc == head_dim, "sin_cos.head_dim mismatch"
    assert q_len == 1, "kernel still restricted to q_len = 1"
    bsz_a, kv_len_a, rank = Ak.shape
    bsz_b, num_kv_heads, rank_b, head_dim_b = Bk.shape
    assert bsz == bsz_a == bsz_b
    assert rank == rank_b and head_dim == head_dim_b
    assert num_heads % num_kv_heads == 0

    # ---------- reshape q ----------
    num_q_per_group = num_heads // num_kv_heads
    q = q.view(bsz, num_kv_heads, num_q_per_group * q_len, head_dim).contiguous()

    # ---------- output ----------
    out = torch.empty((bsz, num_kv_heads, num_q_per_group * q_len, kv_len),
                      dtype=q.dtype, device=q.device)

    # ---------- strides ----------
    stride_q_b, stride_q_g, stride_q_lq, stride_q_d = q.stride()
    stride_A_b, stride_A_l, stride_A_r              = Ak.stride()
    stride_B_b, stride_B_g, stride_B_r, stride_B_d  = Bk.stride()
    stride_o_b, stride_o_g, stride_o_lq, stride_o_lkv = out.stride()
    stride_s_l, stride_s_d = sin_cos.stride()

    # ---------- launch grid ----------
    grid = lambda META: (triton.cdiv(kv_len, META["BLOCK_SIZE_L"]), num_kv_heads, bsz)

    _qAB_fwd_precomp[grid](
        # pointers
        q_ptr=q, A_ptr=Ak, B_ptr=Bk, out_ptr=out, sincos_ptr=sin_cos,
        # strides
        stride_q_b=stride_q_b, stride_q_g=stride_q_g,
        stride_q_lq=stride_q_lq, stride_q_d=stride_q_d,
        stride_A_b=stride_A_b, stride_A_l=stride_A_l, stride_A_r=stride_A_r,
        stride_B_b=stride_B_b, stride_B_g=stride_B_g,
        stride_B_r=stride_B_r, stride_B_d=stride_B_d,
        stride_o_b=stride_o_b, stride_o_g=stride_o_g,
        stride_o_lq=stride_o_lq, stride_o_lkv=stride_o_lkv,
        stride_s_l=stride_s_l, stride_s_d=stride_s_d,
        # sizes
        Q_LEN=num_q_per_group, KV_LEN=kv_len, RANK=rank, HEAD_DIM=head_dim,
        # meta
        BLOCK_SIZE_D=head_dim // 2,   # 64 with your defaults
    )
    return out.view(bsz, num_heads, q_len, kv_len)
