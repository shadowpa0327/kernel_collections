import torch
from torch.autograd.profiler import record_function

def group_query_attention_xKV(
    query,                  # (bsz, num_heads, q_len, head_dim)
    key_A,                  # (bsz, kv_len, k_rank)
    key_B,                  # (bsz, num_kv_heads, k_rank, head_dim)
    value_A,                # (bsz, kv_len, v_rank)
    value_B,                # (bsz, num_kv_heads, v_rank, head_dim)
    attn_mask=None,         # (q_len, kv_len) or (bsz, num_heads, q_len, kv_len)
    dropout_p=0.0,          # dropout probability
):
    """
    Factorized version of group query attention where both K and V are factorized.
    K = key_A @ key_B and V = value_A @ value_B
    
    Supports different ranks for K and V factorizations.
    """
    # Get dimensions
    bsz, num_q_heads, head_dim = query.shape
    _, kv_len, k_rank = key_A.shape
    _, num_kv_heads, k_rank2, _ = key_B.shape
    _, _, v_rank = value_A.shape
    _, _, v_rank2, _ = value_B.shape
    
    q_len = 1

    # Validate dimensions
    assert k_rank == k_rank2, "Rank mismatch between key_A and key_B"
    assert v_rank == v_rank2, "Rank mismatch between value_A and value_B"
    
    # Ensure num_heads is divisible by num_kv_heads
    assert num_q_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    group_size = num_q_heads // num_kv_heads # how many Q-heads attend per KV-head
    assert q_len == 1, "q_len must be 1"
    
    # Reshape query to group the heads
    # (bsz, num_heads, q_len, head_dim) -> (bsz, num_kv_heads, group_size*q_len, head_dim)
    query = query.view(bsz, num_kv_heads, group_size*q_len, head_dim)
    
    # 3) Compute attention weights = Q * K^T / sqrt(head_dim)
    with record_function("q@K"):
        # Compute Q @ key_B^T -> (bsz, num_kv_heads, group_size*q_len, k_rank)
        latents = torch.matmul(query, key_B.transpose(-1, -2))
        latents = latents.view(bsz, -1, k_rank) # (bsz, num_heads*q_len, k_rank)
        
        # Compute latents @ key_A^T -> (bsz, num_heads*q_len, kv_len)
        attn_weights = torch.matmul(latents, key_A.transpose(-1, -2))
        attn_weights = attn_weights.reshape(bsz, num_q_heads, q_len, kv_len)
        
        # Scale attention weights
        #attn_weights = attn_weights / (head_dim ** 0.5)
    
    # 4) Optionally apply an attention mask
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    # 5) Softmax over kv_len dimension
    with torch.autograd.profiler.record_function("softmax"):
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # 6) (Optional) apply dropout
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)
    
    # 7) Compute attention output
    with torch.autograd.profiler.record_function("attn_weights@V"):
        # Factorized computation:
        # First reshape attention weights to (bsz, num_heads*q_len, kv_len)
        # for easier matrix multiplication with value_A
        attn_score = attn_weights.view(bsz, num_q_heads*q_len, kv_len)
        
        # Compute: attn_score @ value_A
        # (bsz, num_heads*q_len, kv_len) @ (bsz, kv_len, v_rank) -> (bsz, num_heads*q_len, v_rank)
        temp = torch.matmul(attn_score, value_A)
        # Reshape to (bsz, num_kv_heads, group_size*q_len, v_rank)
        temp = temp.view(bsz, num_kv_heads, group_size*q_len, v_rank)
        
        # Compute: temp @ value_B
        # (bsz, num_kv_heads, group_size*q_len, v_rank) @ (bsz, num_kv_heads, v_rank, head_dim)
        # -> (bsz, num_kv_heads, group_size*q_len, head_dim)
        attn_output = torch.matmul(temp, value_B)
    
    # 8) Reshape back to (bsz, num_heads, q_len, head_dim)
    attn_output = attn_output.view(bsz, num_q_heads*q_len, head_dim)
    
    return attn_output




def group_query_attention_xKV_k_only(
    query,                  # (bsz, num_heads, q_len, head_dim)
    key_A,                  # (bsz, kv_len, k_rank)
    key_B,                  # (bsz, num_kv_heads, k_rank, head_dim)
    value,                # (bsz, kv_len, num_kv_heads, head_dim)
    attn_mask=None,         # (q_len, kv_len) or (bsz, num_heads, q_len, kv_len)
    dropout_p=0.0,          # dropout probability
):
    """
    Factorized version of group query attention where both K and V are factorized.
    K = key_A @ key_B and V = value_A @ value_B
    
    Supports different ranks for K and V factorizations.
    """
    # Get dimensions
    bsz, num_q_heads, head_dim = query.shape
    _, kv_len, k_rank = key_A.shape
    _, num_kv_heads, k_rank2, _ = key_B.shape
    
    q_len = 1

    value = value.transpose(1, 2) # (bsz, num_kv_heads, kv_len, head_dim)

    # Validate dimensions
    assert k_rank == k_rank2, "Rank mismatch between key_A and key_B"
    
    # Ensure num_heads is divisible by num_kv_heads
    assert num_q_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    group_size = num_q_heads // num_kv_heads # how many Q-heads attend per KV-head
    assert q_len == 1, "q_len must be 1"
    
    # Reshape query to group the heads
    # (bsz, num_heads, q_len, head_dim) -> (bsz, num_kv_heads, group_size*q_len, head_dim)
    query = query.view(bsz, num_kv_heads, group_size*q_len, head_dim)
    
    # 3) Compute attention weights = Q * K^T / sqrt(head_dim)
    with record_function("q@K"):
        # Compute Q @ key_B^T -> (bsz, num_kv_heads, group_size*q_len, k_rank)
        latents = torch.matmul(query, key_B.transpose(-1, -2))
        latents = latents.view(bsz, -1, k_rank) # (bsz, num_heads*q_len, k_rank)
        
        # Compute latents @ key_A^T -> (bsz, num_heads*q_len, kv_len)
        attn_weights = torch.matmul(latents, key_A.transpose(-1, -2))
        attn_weights = attn_weights.reshape(bsz, num_q_heads, q_len, kv_len)
        
        # Scale attention weights
        #attn_weights = attn_weights / (head_dim ** 0.5)
    
    # 4) Optionally apply an attention mask
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    # 5) Softmax over kv_len dimension
    with torch.autograd.profiler.record_function("softmax"):
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # 6) (Optional) apply dropout
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)
    
    # 7) Compute attention output
    with torch.autograd.profiler.record_function("attn_weights@V"):
        attn_score = attn_weights.view(bsz, num_kv_heads, group_size*q_len, kv_len)
        attn_output = torch.matmul(attn_score, value)
    
    
    # 8) Reshape back to (bsz, num_heads, q_len, head_dim)
    attn_output = attn_output.view(bsz, num_q_heads*q_len, head_dim)
    
    return attn_output
