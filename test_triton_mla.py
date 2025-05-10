from flash_decoding.mla import decode_attention_fwd, decode_attention_reference
import torch

if __name__ == "__main__":
   # ---------------------------
    # 1) BASIC PARAMETERS
    # ---------------------------
    batch       = 1
    q_len       = 1
    kv_len      = 2048*64+256
    num_q_heads = 32
    num_kv_heads = 1
    head_dim    = 384
    sm_scale    = 1.0

    # Create random Q, K, V, O with the shapes you specified.
    # Here we assume you want them on CUDA, half-precision (float16):
    dtype = torch.float16
    device = 'cuda'

    # q:  (q_len, num_q_heads, head_dim) --> reshape to (batch, num_q_heads, head_dim)
    q = torch.randn(batch, q_len*num_q_heads, head_dim, dtype=dtype, device=device) # shape: (1, 32, 128)

    # k_buffer, v_buffer: (kv_len, num_kv_heads, head_dim)
    k_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # o:  (q_len, num_q_heads, head_dim) or similarly (batch, heads, head_dim)
    o = torch.zeros_like(q)

    # ---------------------------
    # 2) CREATE kv_indptr, kv_indices
    # ---------------------------
    # Since batch=1, we only need to say that:
    #   - kv_indptr[0] = 0
    #   - kv_indptr[1] = kv_len (4096)
    #
    # This means the entire [0..4096) range of kv_indices belongs to batch 0.
    #kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

    # kv_indices will be a simple range [0..4096).
    # kv_indices = torch.arange(kv_len, dtype=torch.int32, device=device)

    # ---------------------------
    # 3) DEFINE THE "SPLITS" AND INTERMEDIATE BUFFERS
    # ---------------------------
    # Suppose we do 1 KV-split for this entire chunk.
    num_kv_splits = torch.tensor([130], dtype=torch.int32, device=device)
    max_kv_splits = 130

    # If your kernel is using "attn_logits" to store partial results per split,
    # you typically want shape = (batch, num_q_heads, max_kv_splits, head_dim_of_V).
    # Here head_dim_of_V = 128, same as K, so:
    attn_logits = torch.zeros(
        (batch, num_q_heads, max_kv_splits, head_dim),
        dtype=dtype,
        device=device,
    )

    # Likewise for log-sum-exp array, shape = (batch, num_q_heads, max_kv_splits)
    attn_lse = torch.zeros(
        (batch, num_q_heads, max_kv_splits),
        dtype=dtype,
        device=device,
    )

    # Example usage:
    decode_attention_fwd(
        q, 
        k_buffer, 
        v_buffer, 
        o,
        sm_scale,
        logit_cap=0.0,
    )

    # Now 'o' would contain the final result after "attention" (in real code).
    print("o shape:", o.shape)            # Should be [1, 32, 128]
    print("attn_logits shape:", attn_logits.shape)  # [1, 32, 1, 128]
    print("attn_lse shape:", attn_lse.shape)        # [1, 32, 1]

    o_ref = decode_attention_reference(q, k_buffer, v_buffer, o, sm_scale)
    print(torch.max(torch.abs(o_ref-o)))
    print("Ouput aligned with reference:", torch.allclose(o, o_ref, atol=5e-2, rtol=5e-2))

    breakpoint()
    # Test performance
    import time
    
    # Warm-up runs
    for _ in range(10):
        decode_attention_fwd(
            q, 
            k_buffer, 
            k_buffer, 
            o,
            sm_scale,
            logit_cap=0.0,
        )
    
    # Measure time for triton implementation
    torch.cuda.synchronize()
    start_time = time.time()
    num_runs = 100
    
    for _ in range(num_runs):
        decode_attention_fwd(
            q, 
            k_buffer, 
            k_buffer, 
            o,
            sm_scale,
            logit_cap=0.0,
        )
    
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs
    print(f"Triton implementation average time: {triton_time*1000:.4f} ms")
    
    # Measure time for reference implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        decode_attention_reference(
            q, 
            k_buffer, 
            v_buffer, 
            o,
            sm_scale,
        )
    
    torch.cuda.synchronize()
    reference_time = (time.time() - start_time) / num_runs
    print(f"Reference implementation average time: {reference_time*1000:.4f} ms")
    print(f"Speedup: {reference_time/triton_time:.2f}x")
    
