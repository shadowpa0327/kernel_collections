import torch

def mla_decode_attention_fwd_ref(
    q_nope,                  # shape: [batch, num_q_heads, lora_rank]
    q_rope,            # shape: [batch, num_q_heads, rope_dim]
    kv_nope,           # shape: [batch, kv_len, num_kv_heads, lora_rank]
    k_rope,            # shape: [batch, kv_len, num_kv_heads, rope_dim]
    sm_scale=1.0,           # float: softmax scaling factor
):
    # Step 1: Store the size 
    batch, num_q_heads, lora_rank = q_nope.shape
    _, _, q_rope_dim = q_rope.shape
    batch, kv_len, num_kv_heads, _ = kv_nope.shape
    assert num_kv_heads == 1
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads

    print(f"q_nope.shape: {q_nope.shape}")
    print(f"q_rope.shape: {q_rope.shape}")
    print(f"kv_nope.shape: {kv_nope.shape}")
    print(f"k_rope.shape: {k_rope.shape}")

    kv_nope = kv_nope.squeeze(2) # [batch, kv_len, lora_rank]
    k_rope = k_rope.squeeze(2) # [batch, kv_len, rope_dim]
    # Step 2: Compute q@K
    qk = torch.matmul(q_nope, kv_nope.transpose(-1, -2))  # [batch, num_q_heads, kv_len]
    qk += torch.matmul(q_rope, k_rope.transpose(-1, -2))  # [batch, num_q_heads, kv_len]
    # Step 3: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_q_heads, kv_len]
    # Step 4: Compute o = softmax(q@K) @ V
    o = torch.matmul(qk, kv_nope) # [batch, num_q_heads, lora_rank]
    o = o.view(batch, num_q_heads, lora_rank)    
    return o