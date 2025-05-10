import torch
from .stage1 import decoding_stage1
from .stage2 import decoding_stage2


def mla_decode_attention_fwd(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, kv_len, num_kv_heads, head_dim]
    v_buffer,           # shape: [batch, kv_len, num_kv_heads, head_dim]
    sm_scale=1.0,       # float: softmax scaling factor
    logit_cap=0.0,      # float: cap for logits
):
    batch, num_q_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k_buffer.shape
    
    # Calculate kv_group_num (ratio of query heads to key/value heads)
    kv_group_num = num_q_heads // num_kv_heads
    assert kv_group_num >= 1
    
    # Determine number of KV splits if not provided
    # FIXME(brian1009): Determine the optimal number of KV splits on the fly
    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=q.device) * 64
    max_kv_splits = 64

    # Create intermediate tensors for attention computation
    attn_logits = torch.zeros(
        (batch, num_q_heads, max_kv_splits, head_dim),
        dtype=torch.float32, 
        device=q.device
    )
    
    attn_lse = torch.zeros(
        (batch, num_q_heads, max_kv_splits),
        dtype=torch.float32,
        device=q.device
    )

    decoding_stage1(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )
    o = torch.zeros_like(q)

    decoding_stage2(
        attn_logits,
        attn_lse,
        q,
        o,
        v_dim=v_buffer.shape[-1],
        kv_seq_len=kv_len,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
    )

    return o



def mla_decode_attention_fwd_pd_sep(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, prefill_kv_len, num_kv_heads, head_dim]
    v_buffer,           # shape: [batch, prefill_kv_len, num_kv_heads, head_dim]
    k_buffer_decoded,   # shape: [batch, decode_kv_len, num_kv_heads, head_dim]
    v_buffer_decoded,   # shape: [batch, decode_kv_len, num_kv_heads, head_dim]
    sm_scale=1.0,       # float: softmax scaling factor
    logit_cap=0.0,      # float: cap for logits
):
    batch, num_q_heads, head_dim = q.shape
    _, prefill_kv_len, num_kv_heads, _ = k_buffer.shape
    _, decode_kv_len, _, _ = k_buffer_decoded.shape
    v_dim = v_buffer.shape[-1]

    # Calculate kv_group_num (ratio of query heads to key/value heads)
    kv_group_num = num_q_heads // num_kv_heads
    assert kv_group_num >= 1

    # Handle prefill buffer
    ## FIXME(brian1009): Determine the optimal number of KV splits on the fly
    num_kv_splits_prefill = torch.ones(batch, dtype=torch.int32, device=q.device) *64
    max_kv_splits_prefill = 64

    max_kv_splits_decode = 2
    num_kv_splits_decode = torch.ones(batch, dtype=torch.int32, device=q.device) * 2
    # Create intermediate tensors for attention computation
    attn_logits_prefill = torch.empty(
        (batch, num_q_heads, max_kv_splits_prefill, head_dim),
        dtype=torch.float32, 
        device=q.device
    )
    attn_lse_prefill = torch.empty(
        (batch, num_q_heads, max_kv_splits_prefill),
        dtype=torch.float32,
        device=q.device
    )
    
    attn_logits_decode = torch.empty(
        (batch, num_q_heads, max_kv_splits_decode, head_dim),
        dtype=torch.float32,
        device=q.device
    )
    attn_lse_decode = torch.empty(
        (batch, num_q_heads, max_kv_splits_decode),
        dtype=torch.float32,
        device=q.device
    )
    
    # Process prefill KV cache
    decoding_stage1(
        q,
        k_buffer,
        v_buffer,
        attn_logits_prefill,
        attn_lse_prefill,
        num_kv_splits_prefill,
        max_kv_splits_prefill,
        sm_scale,
        logit_cap,
    ) 
    # Process decode KV cache
    decoding_stage1(
        q,
        k_buffer_decoded,
        v_buffer_decoded,
        attn_logits_decode,
        attn_lse_decode,
        num_kv_splits_decode,
        max_kv_splits_decode,
        sm_scale,
        logit_cap,
    )

    # Merge prefill and decode results
    attn_logits = torch.cat([attn_logits_prefill, attn_logits_decode], dim=2)
    attn_lse = torch.cat([attn_lse_prefill, attn_lse_decode], dim=2)
    o = torch.zeros_like(q)

    decoding_stage2(
        attn_logits,
        attn_lse,
        q,
        o,
        v_dim=v_dim,
        kv_seq_len=prefill_kv_len+decode_kv_len,
        num_kv_splits=num_kv_splits_prefill+num_kv_splits_decode,
        max_kv_splits=max_kv_splits_prefill+max_kv_splits_decode,
    )
    
    return o



def mla_decode_attention_fwd_pd_sep_xKV(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    k_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    v_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    v_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    k_buffer_decoded,   # shape: [batch, decode_kv_len, num_kv_heads, head_dim]
    v_buffer_decoded,   # shape: [batch, decode_kv_len, num_kv_heads, head_dim]
    sm_scale=1.0,       # float: softmax scaling factor
    logit_cap=0.0,      # float: cap for logits
):
    batch, num_q_heads, head_dim = q.shape
    _, prefill_kv_len, rank = k_A_buffer.shape
    _, num_kv_heads, _, _ = k_B_buffer.shape
    _, decode_kv_len, _, _ = k_buffer_decoded.shape
    v_dim = v_buffer_decoded.shape[-1]

    assert num_kv_heads == 1, "Only one KV head is supported for now"

    q_fused_KB = torch.matmul(q, k_B_buffer.squeeze(1).transpose(-1, -2)) # [batch, num_q_heads, rank]
    k_A_buffer = k_A_buffer.unsqueeze(2)  # [batch, prefill_kv_len, 1, rank]
    v_A_buffer = v_A_buffer.unsqueeze(2)  # [batch, prefill_kv_len, 1, rank]

    # Handle prefill buffer
    ## FIXME(brian1009): Determine the optimal number of KV splits on the fly
    num_kv_splits_prefill = torch.ones(batch, dtype=torch.int32, device=q.device) * 128
    max_kv_splits_prefill = 128

    max_kv_splits_decode = 2
    num_kv_splits_decode = torch.ones(batch, dtype=torch.int32, device=q.device) * 2
    # Create intermediate tensors for attention computation
    attn_logits_prefill = torch.empty(
        (batch, num_q_heads, max_kv_splits_prefill, rank),
        dtype=torch.float32, 
        device=q.device
    )
    attn_lse_prefill = torch.empty(
        (batch, num_q_heads, max_kv_splits_prefill),
        dtype=torch.float32,
        device=q.device
    )

    attn_logits_decode = torch.empty(
        (batch, num_q_heads, max_kv_splits_decode, head_dim),
        dtype=torch.float32,
        device=q.device
    )
    attn_lse_decode = torch.empty(
        (batch, num_q_heads, max_kv_splits_decode),
        dtype=torch.float32,
        device=q.device
    )

    decoding_stage1(
        q_fused_KB,
        k_A_buffer,
        v_A_buffer,
        attn_logits_prefill,
        attn_lse_prefill,
        num_kv_splits_prefill,
        max_kv_splits_prefill,
        sm_scale,
        logit_cap,
    )

    attn_logits_prefill = torch.matmul(attn_logits_prefill.view(batch, -1, rank).to(torch.float16), v_B_buffer.squeeze(1)).to(torch.float32) # [batch, num_q_heads, head_dim]
    attn_logits_prefill = attn_logits_prefill.reshape(batch, num_q_heads, max_kv_splits_prefill, head_dim)

    decoding_stage1(
        q,
        k_buffer_decoded,
        v_buffer_decoded,
        attn_logits_decode,
        attn_lse_decode,
        num_kv_splits_decode,
        max_kv_splits_decode,
        sm_scale,
        logit_cap,
    )

    breakpoint()

    # Merge prefill and decode results
    attn_logits = torch.cat([attn_logits_prefill, attn_logits_decode], dim=2)
    attn_lse = torch.cat([attn_lse_prefill, attn_lse_decode], dim=2)
    o = torch.empty_like(q)
    decoding_stage2(
        attn_logits,
        attn_lse,
        q,
        o,
        v_dim=v_dim,
        kv_seq_len=prefill_kv_len+decode_kv_len,
        num_kv_splits=num_kv_splits_prefill+num_kv_splits_decode,
        max_kv_splits=max_kv_splits_prefill+max_kv_splits_decode,
    )

    return o