import torch 

def decode_attention_pd_sep_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, num_kv_heads, prefill_kv_len, head_dim]
    v_buffer,           # shape: [batch, num_kv_heads, prefill_kv_len, head_dim]
    k_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    v_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    sm_scale,           # float: softmax scaling factor
):
    # Step 1: Store the size 
    batch, num_q_heads, head_dim = q.shape
    batch, num_kv_heads, prefill_kv_len, head_dim = k_buffer.shape
    batch, num_kv_heads, decode_kv_len, head_dim = k_buffer_decoded.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
 
    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)

    # Step 2: Compute q@K
    qk_prefill = torch.matmul(q, k_buffer.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    qk_decoded = torch.matmul(q, k_buffer_decoded.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, decode_kv_len]
    qk = torch.cat([qk_prefill, qk_decoded], dim=-1) # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len + decode_kv_len]
    # Step 3: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len + decode_kv_len]
    # Step 4: Compute o = softmax(q@K) @ V 
    o = torch.matmul(qk, torch.cat([v_buffer, v_buffer_decoded], dim=-2)) # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    o = o.view(batch, num_q_heads, head_dim)
    return o

def decode_attention_pd_sep_xKV_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    k_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    v_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    v_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    k_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    v_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    sm_scale,           # float: softmax scaling factor   
):
    # Step 1: Store the sizes
    batch, num_q_heads, head_dim = q.shape
    batch, prefill_kv_len, rank = k_A_buffer.shape
    batch, num_kv_heads, rank, head_dim = k_B_buffer.shape
    batch, num_kv_heads, decode_kv_len, head_dim = k_buffer_decoded.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads

    # Step 2: Reconstruct the original k_buffer and v_buffer from their factorized components
    # For each KV head, multiply A_buffer with the corresponding B_buffer
    k_buffer = torch.matmul(k_A_buffer.unsqueeze(1), k_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    v_buffer = torch.matmul(v_A_buffer.unsqueeze(1), v_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
        
    # Reshape q to match the KV head grouping
    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)
    
    # Step 3: Compute q@K for both prefill and decode buffers
    qk_prefill = torch.matmul(q, k_buffer.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, prefill_kv_len]
    qk_decoded = torch.matmul(q, k_buffer_decoded.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, decode_kv_len]
    
    # Concatenate the attention scores
    qk = torch.cat([qk_prefill, qk_decoded], dim=-1) # [batch, num_kv_heads, num_q_heads_per_kv_group, prefill_kv_len + decode_kv_len]
    
    # Step 4: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_kv_heads, num_q_heads_per_kv_group, prefill_kv_len + decode_kv_len]
    
    # Step 5: Compute o = softmax(q@K) @ V
    o = torch.matmul(qk, torch.cat([v_buffer, v_buffer_decoded], dim=-2)) # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    
    # Reshape o back to the original shape
    o = o.view(batch, num_q_heads, head_dim)
    
    return o

#NOTE(brian1009): This version, do not reconstruction directly, it fuse B matrices elsewhere.
def decode_attention_pd_sep_xKV_reference_v2(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    k_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    v_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    v_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    k_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    v_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    sm_scale,           # float: softmax scaling factor   
):
    # Step 1: Store the sizes
    batch, num_q_heads, head_dim = q.shape
    batch, prefill_kv_len, rank = k_A_buffer.shape
    batch, num_kv_heads, rank, head_dim = k_B_buffer.shape
    batch, num_kv_heads, decode_kv_len, head_dim = k_buffer_decoded.shape
    
    assert num_kv_heads == 1, "num_kv_heads must be 1"
    num_q_heads_per_kv_group = num_q_heads
  
    q_fused_kB = torch.matmul(q, k_B_buffer.squeeze(1).transpose(-1, -2)) # [batch, num_q_heads, rank]
    qK_prefill = torch.matmul(q_fused_kB, k_A_buffer.transpose(-1, -2)) # [batch, num_q_heads, prefill_kv_len]
    qK_decodeed = torch.matmul(q, k_buffer_decoded.squeeze(1).transpose(-1, -2)) # [batch, num_q_heads, decode_kv_len]
    qK = torch.cat([qK_prefill, qK_decodeed], dim=-1) # [batch, num_q_heads, prefill_kv_len + decode_kv_len]
    
    qK = qK / sm_scale
    qK = torch.nn.functional.softmax(qK, dim=-1, dtype=torch.float32).to(qK.dtype)  # [batch, num_q_heads, prefill_kv_len + decode_kv_len]
    
    qK_prefill = qK[:, :, :prefill_kv_len]
    qK_decodeed = qK[:, :, prefill_kv_len:]

    o_prefill = torch.matmul(qK_prefill, v_A_buffer) # [batch, num_q_heads, rank]
    o_prefill = torch.matmul(o_prefill, v_B_buffer.squeeze(1)) # [batch, num_q_heads, head_dim]
    o_decodeed = torch.matmul(qK_decodeed, v_buffer_decoded.squeeze(1)) # [batch, num_q_heads, head_dim]

    o = o_prefill + o_decodeed

    return o

def decode_attention_partial_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, num_kv_heads, kv_len, head_dim]
    v_buffer,           # shape: [batch, num_kv_heads, kv_len, head_dim]
    sm_scale,           # float: softmax scaling factor
):
    """
    Compute partial attention for a segment of KV cache using PyTorch native operations.
    Returns both the partial output and the LSE values for later merging.
    """
    # Step 1: Store the sizes
    batch, num_q_heads, head_dim = q.shape
    batch, num_kv_heads, kv_len, head_dim = k_buffer.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    
    # Reshape q to match the KV head grouping
    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)
    
    # Step 2: Compute q@K
    qk = torch.matmul(q, k_buffer.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    
    # Step 3: Compute scaled logits
    qk = qk / sm_scale
    
    # Step 4: Compute LSE (log-sum-exp) values before softmax
    # First get the max value for numerical stability
    qk_max = torch.max(qk, dim=-1, keepdim=True)[0]  # [batch, num_kv_heads, num_q_heads_per_kv_group, 1]
    
    # Compute exp(qk - max) for softmax
    exp_qk = torch.exp(qk - qk_max)  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    
    # Compute sum of exp values for softmax denominator
    exp_sum = torch.sum(exp_qk, dim=-1, keepdim=True)  # [batch, num_kv_heads, num_q_heads_per_kv_group, 1]
    
    # Compute softmax
    attention_probs = exp_qk / exp_sum  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    
    # Step 5: Compute partial output
    partial_output = torch.matmul(attention_probs, v_buffer)  # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    partial_output = partial_output.view(batch, num_q_heads, head_dim)
    
    # Compute LSE values for later merging
    # LSE = max + log(sum(exp(x - max)))
    lse = qk_max.squeeze(-1) + torch.log(exp_sum.squeeze(-1))  # [batch, num_kv_heads, num_q_heads_per_kv_group]
    lse = lse.view(batch, num_q_heads)
    
    return partial_output, lse

def merge_attention_partials_reference(
    partial_outputs,    # list of tensors with shape [batch, num_q_heads, head_dim]
    partial_lses,       # list of tensors with shape [batch, num_q_heads]
):
    """
    Merge partial attention outputs using the LSE values with PyTorch native operations.
    """
    batch, num_q_heads, head_dim = partial_outputs[0].shape
    device = partial_outputs[0].device
    dtype = partial_outputs[0].dtype
    
    # Find the maximum LSE value for each query
    max_lse, _ = torch.max(torch.stack(partial_lses, dim=-1), dim=-1)  # [batch, num_q_heads]
    
    # Initialize output tensor and denominator
    final_output = torch.zeros((batch, num_q_heads, head_dim), dtype=dtype, device=device)
    denominator = torch.zeros((batch, num_q_heads), dtype=torch.float32, device=device)
    
    # Merge the partial outputs
    for partial_output, partial_lse in zip(partial_outputs, partial_lses):
        # Scale factor based on the difference between this partial's LSE and the max LSE
        scale_factor = torch.exp(partial_lse - max_lse).unsqueeze(-1)  # [batch, num_q_heads, 1]
        
        # Add the scaled partial output to the final output
        final_output += partial_output * scale_factor
        
        # Add to the denominator
        denominator += scale_factor.squeeze(-1)
    
    # Normalize the final output
    final_output = final_output / denominator.unsqueeze(-1)
    
    return final_output

def decode_attention_pd_sep_merged_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, num_kv_heads, prefill_kv_len, head_dim]
    v_buffer,           # shape: [batch, num_kv_heads, prefill_kv_len, head_dim]
    k_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    v_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    sm_scale,           # float: softmax scaling factor
):
    """
    Compute attention by separately processing prefill and decode buffers,
    then merging the results using PyTorch native operations.
    """
    # Compute partial attention for prefill buffer
    prefill_output, prefill_lse = decode_attention_partial_reference(
        q, k_buffer, v_buffer, sm_scale
    )
    
    # Compute partial attention for decode buffer
    decode_output, decode_lse = decode_attention_partial_reference(
        q, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    
    # Merge the partial results
    final_output = merge_attention_partials_reference(
        [prefill_output, decode_output],
        [prefill_lse, decode_lse]
    )
    
    return final_output

def decode_attention_pd_sep_merged_xKV_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    k_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    v_A_buffer,         # shape: [batch, prefill_kv_len, rank]
    v_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    k_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    v_buffer_decoded,   # shape: [batch, num_kv_heads, decode_kv_len, head_dim]
    sm_scale,           # float: softmax scaling factor
):
    """
    Compute attention by separately processing prefill and decode buffers using the xKV approach,
    then merging the results using the decode_attention_partial_reference and 
    merge_attention_partials_reference functions.
    """
    
    # Get the shape, num_q_heads, kv_heads etc.
    batch, num_q_heads, head_dim = q.shape
    batch, prefill_kv_len, rank = k_A_buffer.shape
    batch, num_kv_heads, rank, head_dim = k_B_buffer.shape
    batch, num_kv_heads, decode_kv_len, head_dim = k_buffer_decoded.shape
    
    assert num_kv_heads == 1, "num_kv_heads must be 1"
    q_fused_kB = torch.matmul(q, k_B_buffer.squeeze(1).transpose(-1, -2)) # [batch, num_q_heads, rank]
    
    # Reshape k_A_buffer and v_A_buffer to match the expected format for partial attention
    k_A_reshaped = k_A_buffer.unsqueeze(1)  # [batch, 1, prefill_kv_len, rank]
    v_A_reshaped = v_A_buffer.unsqueeze(1)  # [batch, 1, prefill_kv_len, rank]
    
    # Compute partial attention for prefill buffer
    prefill_output, prefill_lse = decode_attention_partial_reference(
        q_fused_kB, k_A_reshaped, v_A_reshaped, sm_scale
    )
    prefill_output = torch.matmul(prefill_output, v_B_buffer.squeeze(1)) # [batch, num_q_heads, head_dim]

    # Compute partial attention for decode buffer
    decode_output, decode_lse = decode_attention_partial_reference(
        q, k_buffer_decoded, v_buffer_decoded, sm_scale
    )

    # Merge the partial results
    final_output = merge_attention_partials_reference(
        [prefill_output, decode_output],
        [prefill_lse, decode_lse]
    )
    
    return final_output

if __name__ == '__main__':
    # Create random tensors with shapes matching the function signature
    batch = 2
    num_q_heads = 32
    num_kv_heads = 1
    head_dim = 128
    prefill_kv_len = 512
    decode_kv_len = 128
    rank = 96

    # Calculate derived values
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    
    # Create random tensors (using uniform [-0.5, 0.5])
    q = torch.rand(batch, num_q_heads, head_dim, dtype=torch.float32) - 0.5
    k_A_buffer = torch.rand(batch, prefill_kv_len, rank, dtype=torch.float32) - 0.5
    k_B_buffer = torch.rand(batch, num_kv_heads, rank, head_dim, dtype=torch.float32) - 0.5
    v_A_buffer = torch.rand(batch, prefill_kv_len, rank, dtype=torch.float32) - 0.5
    v_B_buffer = torch.rand(batch, num_kv_heads, rank, head_dim, dtype=torch.float32) - 0.5
    k_buffer_decoded = torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=torch.float32) - 0.5
    v_buffer_decoded = torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=torch.float32) - 0.5
    k_buffer = torch.matmul(k_A_buffer.unsqueeze(1), k_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    v_buffer = torch.matmul(v_A_buffer.unsqueeze(1), v_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    sm_scale = float(1.0 / (head_dim ** 0.5))
    
    print("\n" + "="*80)
    print("TESTING ATTENTION IMPLEMENTATIONS")
    print("="*80)
    
    print("\nInput shapes:")
    print(f"q: {q.shape}")
    print(f"k_A_buffer: {k_A_buffer.shape}")
    print(f"k_B_buffer: {k_B_buffer.shape}")
    print(f"v_A_buffer: {v_A_buffer.shape}")
    print(f"v_B_buffer: {v_B_buffer.shape}")
    print(f"k_buffer_decoded: {k_buffer_decoded.shape}")
    print(f"v_buffer_decoded: {v_buffer_decoded.shape}")
    print(f"k_buffer: {k_buffer.shape}")
    print(f"v_buffer: {v_buffer.shape}")
     
    golden = decode_attention_pd_sep_reference(
        q, k_buffer, v_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    
    print("\n" + "-"*80)
    print("TEST 1: decode_attention_pd_sep_xKV_reference (reconstruct full KV buffers in Attention)")
    print("-"*80)
    result = decode_attention_pd_sep_xKV_reference(
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    print(f"Result shape: {result.shape}")
    match = torch.allclose(result, golden, atol=5e-4, rtol=5e-4)
    print(f"Test passed: {'✓' if match else '✗'}")
    print(f"Max difference: {torch.max(torch.abs(result - golden))}")
    
    print("\n" + "-"*80)
    print("TEST 2: decode_attention_pd_sep_xKV_reference_v2 (Fused B matrices elsewhere)")
    print("-"*80)
    result_v2 = decode_attention_pd_sep_xKV_reference_v2(
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    print(f"Result shape: {result_v2.shape}")
    match_v2 = torch.allclose(result, result_v2, atol=5e-4, rtol=5e-4)
    print(f"Result matched with TEST 1: {'✓' if match_v2 else '✗'}")
    print(f"Max difference: {torch.max(torch.abs(result - result_v2))}")
    
    print("\n" + "-"*80)
    print("TEST 3: decode_attention_pd_sep_merged_reference (Testing the seperated chunk then merge)")
    print("-"*80)
    result_merged = decode_attention_pd_sep_merged_reference(
        q, k_buffer, v_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    print(f"Result shape: {result_merged.shape}")
    match_merged = torch.allclose(result, result_merged, atol=5e-4, rtol=5e-4)
    print(f"Result matched with TEST 1: {'✓' if match_merged else '✗'}")
    print(f"Max difference: {torch.max(torch.abs(result - result_merged))}")

    print("\n" + "-"*80)
    print("TEST 4: decode_attention_pd_sep_merged_xKV_reference (Testing the seperated chunk then merge, under xKV with B matrices fused elsewhere)")
    print("-"*80)
    result_merged_xKV = decode_attention_pd_sep_merged_xKV_reference(
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    print(f"Result shape: {result_merged_xKV.shape}")
    match_merged_xKV = torch.allclose(result, result_merged_xKV, atol=5e-4, rtol=5e-4)
    print(f"Result matched with TEST 1: {'✓' if match_merged_xKV else '✗'}")
    print(f"Max difference: {torch.max(torch.abs(result - result_merged_xKV))}")

   
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"All tests passed: {'✓' if match and match_v2 and match_merged_xKV and match_merged else '✗'}")
    

    
