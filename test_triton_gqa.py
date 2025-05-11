import torch
from flash_decoding.gqa import decode_attention_fwd

def decode_attention_reference(q, k_buffer, v_buffer, o, sm_scale):
    # Step 1: Store the size 
    batch, num_q_heads, head_dim = q.shape
    batch, kv_len, num_kv_heads, head_dim = k_buffer.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    k_buffer = k_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]
    v_buffer = v_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]

    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)

    # Step 2: Compute q@K
    qk = torch.matmul(q, k_buffer.transpose(-1, -2))
    # Step 3: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    # Step 4: Compute o = softmax(q@K) @ V
    o = torch.matmul(qk, v_buffer) # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    o = o.view(batch, num_q_heads, head_dim)
    return o

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
    k_rank      = 384
    v_rank      = 384
    # Create random Q, K, V, O with the shapes you specified.
    # Here we assume you want them on CUDA, half-precision (float16):
    dtype = torch.float16
    device = 'cuda'

    # q:  (q_len, num_q_heads, head_dim) --> reshape to (batch, num_q_heads, head_dim)
    q = torch.randn(batch, q_len*num_q_heads, head_dim, dtype=dtype, device=device)

    # k_buffer, v_buffer: (kv_len, num_kv_heads, head_dim)
    k_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # o:  (q_len, num_q_heads, head_dim) or similarly (batch, heads, head_dim)
    o = torch.zeros_like(q)


    # ---------------------------
    # 2) DEFINE THE "SPLITS" AND INTERMEDIATE BUFFERS
    # ---------------------------
    # Suppose we do 1 KV-split for this entire chunk.
    num_kv_splits = torch.tensor([128], dtype=torch.int32, device=device)
    max_kv_splits = 128

    # Example usage:
    decode_attention_fwd(
        q, 
        k_buffer, 
        v_buffer, 
        o,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )
    o_ref = decode_attention_reference(q, k_buffer, v_buffer, o, sm_scale)
    print(torch.max(torch.abs(o_ref-o)))
    print("Ouput aligned with reference:", torch.allclose(o, o_ref, atol=5e-2, rtol=5e-2))