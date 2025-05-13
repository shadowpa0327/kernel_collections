import torch
from .stage1 import decoding_stage1
from .stage2 import decoding_stage2


def mla_decode_attention_fwd(
    q_nope,           # shape: [batch, num_q_heads, lora_rank]
    q_rope,           # shape: [batch, num_q_heads, rope_dim]
    kv_nope,          # shape: [batch, kv_len, num_kv_heads, lora_rank]
    k_rope,           # shape: [batch, kv_len, num_kv_heads, rope_dim]
    sm_scale=1.0,     # float: softmax scaling factor
):
    batch, num_q_heads, lora_rank = q_nope.shape
    _, kv_len, num_kv_heads, _ = kv_nope.shape
    
    assert num_kv_heads == 1
    
    # Determine number of KV splits if not provided
    # FIXME: Determine the optimal number of KV splits on the fly
    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=q_nope.device) * 128
    max_kv_splits = 128

    # Create intermediate tensors for attention computation
    attn_logits = torch.zeros(
        (batch, num_q_heads, max_kv_splits, lora_rank),
        dtype=torch.float32, 
        device=q_nope.device
    )
    
    attn_lse = torch.zeros(
        (batch, num_q_heads, max_kv_splits),
        dtype=torch.float32,
        device=q_nope.device
    )

    # Stage 1: Compute attention logits and LSE
    decoding_stage1(
        q_nope,
        q_rope,
        kv_nope,
        k_rope,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )

    # Create output tensor
    o = torch.empty_like(q_nope)

    # Stage 2: Combine attention outputs
    decoding_stage2(
        attn_logits,
        attn_lse,
        o,
        kv_seq_len=kv_len,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
    )

    return o



def mla_decode_attention_xKV(
    q_nope,           # shape: [batch, num_q_heads, lora_rank]
    q_rope,           # shape: [batch, num_q_heads, rope_dim]
    kv_A_nope,        # shape: [batch, kv_len, 1, rank]
    kv_B_nope,        # shape: [batch, num_kv_heads, rank, lora_rank]
    k_rope,           # shape: [batch, kv_len, num_kv_heads, rope_dim]
    sm_scale=1.0,     # float: softmax scaling factor
):
    batch, num_q_heads, lora_rank = q_nope.shape
    _, kv_len, _, xKV_rank = kv_A_nope.shape
    assert kv_A_nope.shape[2] == 1
    _, num_kv_heads, xKV_rank, _ = kv_B_nope.shape
    assert num_kv_heads == 1
    
    #NOTE(brian1009): for xKV, we fuse kv_B_nope and q_nope
    kv_B_nope = kv_B_nope.squeeze(1)
    q_nope_fused = torch.matmul(q_nope, kv_B_nope.transpose(1, 2))
    
    # Determine number of KV splits if not provided
    # FIXME: Determine the optimal number of KV splits on the fly
    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=q_nope.device) * 128
    max_kv_splits = 128

    # Create intermediate tensors for attention computation
    attn_logits = torch.zeros(
        (batch, num_q_heads, max_kv_splits, xKV_rank),
        dtype=torch.float32, 
        device=q_nope.device
    )
    
    attn_lse = torch.zeros(
        (batch, num_q_heads, max_kv_splits),
        dtype=torch.float32,
        device=q_nope.device
    )

    # Stage 1: Compute attention logits and LSE
    decoding_stage1(
        q_nope_fused,
        q_rope,
        kv_A_nope,
        k_rope,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )

    # Create output tensor
    o = torch.empty_like(q_nope_fused)

    # Stage 2: Combine attention outputs
    decoding_stage2(
        attn_logits,
        attn_lse,
        o,
        kv_seq_len=kv_len,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
    )

    o = torch.matmul(o, kv_B_nope)

    return o

    
    