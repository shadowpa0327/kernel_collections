import torch
from flash_recontruction import (
    gqa_xKV_no_pe_k_only_ref, 
    gqa_xKV_no_pe_k_only,
    gqa_xKV_no_pe_ref,
    gqa_xKV_no_pe,
    gqa_xKV_no_pe_v2
)

if __name__ == "__main__":
   # ---------------------------
    # 1) BASIC PARAMETERS
    # ---------------------------
    batch       = 1
    q_len       = 1
    kv_len      = 256*1024
    num_q_heads = 32
    num_kv_heads = 8
    head_dim    = 128
    sm_scale    = 1.0
    k_rank      = 256
    v_rank      = 384
    # Create random Q, K, V, O with the shapes you specified.
    # Here we assume you want them on CUDA, half-precision (float16):
    dtype = torch.float16
    device = 'cuda'

    # q:  (q_len, num_q_heads, head_dim) --> reshape to (batch, num_q_heads, head_dim)
    q = torch.randn(batch, q_len*num_q_heads, head_dim, dtype=dtype, device=device)/3

    # k_buffer, v_buffer: (kv_len, num_kv_heads, head_dim)
    k_A_buffer = torch.randn(batch, kv_len, k_rank, dtype=dtype, device=device)/3
    k_B_buffer = torch.randn(batch, num_kv_heads, k_rank, head_dim, dtype=dtype, device=device)/3
    v_A_buffer = torch.randn(batch, kv_len, v_rank, dtype=dtype, device=device)/3
    v_B_buffer = torch.randn(batch, num_kv_heads, v_rank, head_dim, dtype=dtype, device=device)/3
    v_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)/3
    # o:  (q_len, num_q_heads, head_dim) or similarly (batch, heads, head_dim)
    o = torch.zeros_like(q)


    # --------------------------- (Testing K only)
    num_kv_splits = torch.tensor([128], dtype=torch.int32, device=device)
    max_kv_splits = 128

    o_ref = gqa_xKV_no_pe_k_only_ref(q, k_A_buffer, k_B_buffer, v_buffer)
    print(o_ref.shape)

    o_fused = gqa_xKV_no_pe_k_only(q, k_A_buffer, k_B_buffer, v_buffer, o, num_kv_splits, max_kv_splits, sm_scale)
    print(o_fused.shape)
    print(torch.max(torch.abs(o_ref - o_fused)))
    print(o_fused)
    print(o_ref)
    print(torch.allclose(o_ref, o_fused, atol=5e-2, rtol=5e-2))
    

    # --------------------------- (Testing K and V)
    o_ref = gqa_xKV_no_pe_ref(
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer
    )
    o = torch.zeros_like(q)
    o_fused = gqa_xKV_no_pe( 
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, o, num_kv_splits, max_kv_splits, sm_scale
    )

    print(o_fused.shape)
    print(torch.max(torch.abs(o_ref - o_fused)))


    # --------------------------- (Testing K and V with v2)
    o_fused = gqa_xKV_no_pe_v2( 
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, num_kv_splits, max_kv_splits, sm_scale
    )
    print(o_fused.shape)
    print(torch.max(torch.abs(o_ref - o_fused)))



a = torch.cuda.nvtx.range_start("gqa_xKV_fused")
gqa_xKV_no_pe(q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, o, num_kv_splits, max_kv_splits, sm_scale)
gqa_xKV_no_pe_v2(q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, num_kv_splits, max_kv_splits, sm_scale)
gqa_xKV_no_pe_k_only(q, k_A_buffer, k_B_buffer, v_buffer, o, num_kv_splits, max_kv_splits, sm_scale)
torch.cuda.nvtx.range_end(a)