import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from flash_attn import flash_attn_with_kvcache

def group_query_attention(query, 
                          key, 
                          value, 
                          attn_mask=None, 
                          dropout_p=0.0):
    """
    Args:
        query: Tensor of shape (bsz, num_heads, q_len, head_dim)
        key:   Tensor of shape (bsz, num_kv_heads, kv_len, head_dim)
        value: Tensor of shape (bsz, num_kv_heads, kv_len, head_dim)
        attn_mask: optional mask broadcastable to (bsz, 1, q_len, kv_len) 
                   or (bsz, num_heads, q_len, kv_len), etc.
        dropout_p: dropout probability to apply on attention weights.

    Returns:
        attn_output: (bsz, num_heads, q_len, head_dim)
        attn_weights: (bsz, num_heads, q_len, kv_len) for inspection if needed
    """
    bsz, num_heads, q_len, head_dim = query.shape
    bsz2, num_kv_heads, kv_len, head_dim2 = key.shape
    bsz3, num_kv_heads2, kv_len2, head_dim3 = value.shape

    # Basic shape checks
    assert bsz == bsz2 == bsz3,           "Batch size mismatch among Q, K, V."
    assert num_kv_heads == num_kv_heads2, "num_kv_heads mismatch between K, V."
    assert head_dim == head_dim2 == head_dim3, "head_dim mismatch among Q, K, V."
    assert kv_len == kv_len2,            "kv_len mismatch between K, V."
    assert num_heads % num_kv_heads == 0, "num_heads must be multiple of num_kv_heads."

    # Compute group size: how many Q-heads attend per KV-head
    group_size = num_heads // num_kv_heads

    # 1) Reshape Q to split out group dimension:
    #    (bsz, num_heads, q_len, head_dim)
    # -> (bsz, num_kv_heads, group_size, q_len, head_dim)
    query = query.view(bsz, num_kv_heads, group_size, head_dim).contiguous()

    # 2) We will do a scaled dot-product within each group:
    #    => each group of Q-heads corresponds to exactly one KV-head index.

    # 3) Compute attention weights = Q * K^T / sqrt(head_dim)
    #    Q: (bsz, num_kv_heads, group_size, q_len, head_dim)
    #    K^T: (bsz, num_kv_heads, group_size, head_dim, kv_len) after transpose
    # -> attn_weights shape: (bsz, num_kv_heads, group_size, q_len, kv_len)
    with record_function("q@K"):
        attn_weights = torch.matmul(
            query, 
            key.transpose(-1, -2)
        ) / (head_dim ** 0.5)
    

    # 4) Optionally apply an attention mask (e.g., for causal or padding)
    #    attn_mask should be broadcastable to attn_weights' shape (..., q_len, kv_len).
    if attn_mask is not None:
        # Expand attn_mask if needed to match (bsz, num_kv_heads, group_size, q_len, kv_len)
        # Usually you'll have shape (bsz, 1, 1, q_len, kv_len) or (bsz, 1, q_len, kv_len),
        # so a broadcast is enough as long as the left dims match.
        attn_weights = attn_weights + attn_mask

    # 5) Softmax over kv_len dimension
    with record_function("softmax"):
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # 6) (Optional) apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # 7) Multiply by values to get final attention output
    #    attn_weights: (bsz, num_kv_heads, group_size*q_len, kv_len)
    #    value:        (bsz, num_kv_heads, kv_len, head_dim)
    # -> attn_output:  (bsz, num_kv_heads, group_size*q_len, head_dim)
    with record_function("attn_weights@V"):
        attn_output = torch.matmul(attn_weights, value)
    # 8) Finally, reshape back: 
    #    (bsz, num_kv_heads, group_size, q_len, kv_len)
    # -> (bsz, num_heads, q_len, head_dim) by merging num_kv_heads * group_size
    attn_output = attn_output.view(bsz, num_heads, q_len, head_dim)

    # # If you want to return attention weights in the standard shape 
    # # (bsz, num_heads, q_len, kv_len) for inspection/logging, 
    # # you also reshape that from (bsz, num_kv_heads, group_size, q_len, kv_len).
    # attn_weights = attn_weights.view(bsz, num_heads, q_len, kv_len)

    return attn_output


def group_query_attentionv2(query, 
                          key, 
                          value, 
                          attn_mask=None, 
                          dropout_p=0.0):
    """
    Args:
        query: Tensor of shape (bsz, num_heads, q_len, head_dim)
        key:   Tensor of shape (bsz, num_kv_heads, kv_len, head_dim)
        value: Tensor of shape (bsz, num_kv_heads, kv_len, head_dim)
        attn_mask: optional mask broadcastable to (bsz, 1, q_len, kv_len) 
                   or (bsz, num_heads, q_len, kv_len), etc.
        dropout_p: dropout probability to apply on attention weights.

    Returns:
        attn_output: (bsz, num_heads, q_len, head_dim)
        attn_weights: (bsz, num_heads, q_len, kv_len) for inspection if needed
    """
    bsz, num_heads, q_len, head_dim = query.shape
    bsz2, num_kv_heads, kv_len, head_dim2 = key.shape
    bsz3, num_kv_heads2, kv_len2, head_dim3 = value.shape

    # Basic shape checks
    assert bsz == bsz2 == bsz3,           "Batch size mismatch among Q, K, V."
    assert num_kv_heads == num_kv_heads2, "num_kv_heads mismatch between K, V."
    assert head_dim == head_dim2 == head_dim3, "head_dim mismatch among Q, K, V."
    assert kv_len == kv_len2,            "kv_len mismatch between K, V."
    assert num_heads % num_kv_heads == 0, "num_heads must be multiple of num_kv_heads."

    # Compute group size: how many Q-heads attend per KV-head
    group_size = num_heads // num_kv_heads

    # 1) Reshape Q to split out group dimension:
    #    (bsz, num_heads, q_len, head_dim)
    # -> (bsz, num_kv_heads, group_size, q_len, head_dim)
    query = query.view(bsz, num_kv_heads, group_size, q_len, head_dim)

    # 2) We will do a scaled dot-product within each group:
    #    => each group of Q-heads corresponds to exactly one KV-head index.

    #    key:   (bsz, num_kv_heads, kv_len, head_dim)
    #    For batched matmul below, we can add a group dimension of size 1
    #    and let broadcasting handle the group_size dimension of Q.
    key = key.unsqueeze(2)  # (bsz, num_kv_heads, 1, kv_len, head_dim)
    # -> after broadcasting, effectively (bsz, num_kv_heads, group_size, kv_len, head_dim)

    #    value: (bsz, num_kv_heads, kv_len, head_dim)
    value = value.unsqueeze(2)  # (bsz, num_kv_heads, 1, kv_len, head_dim)

    # 3) Compute attention weights = Q * K^T / sqrt(head_dim)
    #    Q: (bsz, num_kv_heads, group_size, q_len, head_dim)
    #    K^T: (bsz, num_kv_heads, group_size, head_dim, kv_len) after transpose
    # -> attn_weights shape: (bsz, num_kv_heads, group_size, q_len, kv_len)
    attn_weights = torch.matmul(
        query, 
        key.transpose(-1, -2)
    ) / (head_dim ** 0.5)

    # 4) Optionally apply an attention mask (e.g., for causal or padding)
    #    attn_mask should be broadcastable to attn_weights' shape (..., q_len, kv_len).
    if attn_mask is not None:
        # Expand attn_mask if needed to match (bsz, num_kv_heads, group_size, q_len, kv_len)
        # Usually you'll have shape (bsz, 1, 1, q_len, kv_len) or (bsz, 1, q_len, kv_len),
        # so a broadcast is enough as long as the left dims match.
        attn_weights = attn_weights + attn_mask

    # 5) Softmax over kv_len dimension
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # 6) (Optional) apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # 7) Multiply by values to get final attention output
    #    attn_weights: (bsz, num_kv_heads, group_size, q_len, kv_len)
    #    value:        (bsz, num_kv_heads, 1, kv_len, head_dim)
    # -> attn_output:  (bsz, num_kv_heads, group_size, q_len, head_dim)
    attn_output = torch.matmul(attn_weights, value)

    # 8) Finally, reshape back: 
    #    (bsz, num_kv_heads, group_size, q_len, head_dim)
    # -> (bsz, num_heads, q_len, head_dim) by merging num_kv_heads * group_size
    attn_output = attn_output.view(bsz, num_heads, q_len, head_dim)

    # # If you want to return attention weights in the standard shape 
    # # (bsz, num_heads, q_len, kv_len) for inspection/logging, 
    # # you also reshape that from (bsz, num_kv_heads, group_size, q_len, kv_len).
    # attn_weights = attn_weights.view(bsz, num_heads, q_len, kv_len)

    return attn_output

def group_query_attention_fa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0
):
    return flash_attn_with_kvcache(query, key, value, causal=True)

@torch.inference_mode()
def group_query_attention_factorized_v_only(
    query,
    key,
    value_A,  # Factorized value component A: (bsz, kv_len, rank)
    value_B,  # Factorized value component B: (bsz, num_kv_heads, rank, head_dim)
    attn_mask=None,
    dropout_p=0.0,
    use_factorized=True  # Flag to switch between factorized and standard computation
):
    """
    Group Query Attention with factorized value matrices.
    
    Args:
        query: Tensor of shape (bsz, num_heads, q_len, head_dim)
        key: Tensor of shape (bsz, num_kv_heads, kv_len, head_dim)
        value_A: Tensor of shape (bsz, kv_len, rank) - first factor of value matrix
        value_B: Tensor of shape (bsz, num_kv_heads, rank, head_dim) - second factor of value matrix
        attn_mask: optional mask broadcastable to (bsz, 1, q_len, kv_len) 
                   or (bsz, num_heads, q_len, kv_len), etc.
        dropout_p: dropout probability to apply on attention weights.
        use_factorized: whether to use factorized computation (True) or standard (False)
                        If False, value_A is treated as the full value tensor and value_B is ignored

    Returns:
        attn_output: (bsz, num_heads, q_len, head_dim)
    """
    bsz, num_heads, q_len, head_dim = query.shape
    bsz2, num_kv_heads, kv_len, head_dim2 = key.shape
    
    # Basic shape checks
    assert bsz == bsz2, "Batch size mismatch between Q and K."
    assert head_dim == head_dim2, "head_dim mismatch between Q and K."
    assert num_heads % num_kv_heads == 0, "num_heads must be multiple of num_kv_heads."
    assert q_len == 1, "q_len must be 1"

    if use_factorized:
        # Check factorized value shapes
        bsz3, kv_len2, rank = value_A.shape
        bsz4, num_kv_heads2, rank2, head_dim3 = value_B.shape
        
        assert bsz == bsz3 == bsz4, "Batch size mismatch in value factors."
        assert kv_len == kv_len2, "KV length mismatch between K and value_A."
        assert num_kv_heads == num_kv_heads2, "num_kv_heads mismatch between K and value_B."
        assert rank == rank2, "Rank mismatch between value_A and value_B."
        assert head_dim == head_dim3, "head_dim mismatch between Q and value_B."
    else:
        # Treat value_A as the full value tensor
        bsz3, num_kv_heads2, kv_len2, head_dim3 = value_A.shape
        assert bsz == bsz3, "Batch size mismatch between Q and V."
        assert num_kv_heads == num_kv_heads2, "num_kv_heads mismatch between K and V."
        assert kv_len == kv_len2, "KV length mismatch between K and V."
        assert head_dim == head_dim3, "head_dim mismatch between Q and V."

    # Compute group size: how many Q-heads attend per KV-head
    group_size = num_heads // num_kv_heads
    assert q_len == 1, "q_len must be 1"
    # 1) Reshape Q to split out group dimension:
    # -> (bsz, num_kv_heads, group_size, q_len*group_size, head_dim)
    query = query.view(bsz, num_kv_heads, q_len*group_size, head_dim).contiguous()

    # 2) We will do a scaled dot-product within each group:
    #    => each group of Q-heads corresponds to exactly one KV-head index.

    # 3) Compute attention weights = Q * K^T / sqrt(head_dim)
    #    Q: (bsz, num_kv_heads, group_size, q_len, head_dim)
    #    K^T: (bsz, num_kv_heads, group_size, head_dim, kv_len) after transpose
    # -> attn_weights shape: (bsz, num_kv_heads, group_size, q_len, kv_len)
    with record_function("q@K"):
        attn_weights = torch.matmul(
            query, 
            key.transpose(-1, -2)
        ) / (head_dim ** 0.5)

    # 3) Optionally apply an attention mask
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    # 4) Softmax over kv_len dimension
    with torch.autograd.profiler.record_function("softmax"):
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # 5) (Optional) apply dropout
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)

    # 6) Compute attention output
    with torch.autograd.profiler.record_function("attn_weights@V"):
        if use_factorized:
            # Factorized computation:
            # First reshape attention weights to (bsz, num_heads, q_len, kv_len)
            # for easier matrix multiplication with value_A
            attn_score = attn_weights.view(bsz, num_heads*q_len, kv_len)
            
            # Compute: attn_score @ value_A
            # (bsz, num_heads, q_len, kv_len) @ (bsz, kv_len, rank) -> (bsz, num_heads, q_len, rank)
            temp = torch.matmul(attn_score, value_A)
            # Reshape to (bsz, num_kv_heads, group_size, q_len, rank)
            temp = temp.view(bsz, num_kv_heads, group_size*q_len, rank)
            
            # Compute: temp @ value_B
            # (bsz, num_kv_heads, group_size, q_len, rank) @ (bsz, num_kv_heads, rank, head_dim)
            # -> (bsz, num_kv_heads, group_size, q_len, head_dim)
            attn_output = torch.matmul(temp, value_B)
        else:
            raise NotImplementedError("Standard computation not implemented")

    # 7) Reshape back to (bsz, num_heads, q_len, head_dim)
    attn_output = attn_output.view(bsz, num_heads, q_len, head_dim)

    return attn_output

def group_query_attention_factorized(
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
    bsz, num_heads, q_len, head_dim = query.shape
    _, kv_len, k_rank = key_A.shape
    _, num_kv_heads, k_rank2, _ = key_B.shape
    _, _, v_rank = value_A.shape
    _, _, v_rank2, _ = value_B.shape
    
    # Validate dimensions
    assert k_rank == k_rank2, "Rank mismatch between key_A and key_B"
    assert v_rank == v_rank2, "Rank mismatch between value_A and value_B"
    
    # Ensure num_heads is divisible by num_kv_heads
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    group_size = num_heads // num_kv_heads # how many Q-heads attend per KV-head
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
        attn_weights = attn_weights.reshape(bsz, num_heads, q_len, kv_len)
        
        # Scale attention weights
        attn_weights = attn_weights / (head_dim ** 0.5)
    
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
        attn_score = attn_weights.view(bsz, num_heads*q_len, kv_len)
        
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
    attn_output = attn_output.view(bsz, num_heads, q_len, head_dim)
    
    return attn_output
