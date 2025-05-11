from flash_decoding.deepseek_mla import mla_decode_attention_fwd, mla_decode_attention_fwd_pd_sep, mla_decode_attention_fwd_pd_sep_xKV
import torch
import time

def decode_attention_reference(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_buffer,           # shape: [batch, kv_len, num_kv_heads, head_dim]
    v_buffer,           # shape: [batch, kv_len, num_kv_heads, head_dim]
    sm_scale,           # float: softmax scaling factor
):
    # Step 1: Store the size 
    batch, num_q_heads, head_dim = q.shape
    batch, kv_len, num_kv_heads, head_dim = k_buffer.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    #assert batch== 1
    #k_buffer = k_buffer.unsqueeze(0) # [batch, kv_len, num_kv_heads, head_dim]
    k_buffer = k_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]
    
    #v_buffer = v_buffer.unsqueeze(0) # [batch, kv_len, num_kv_heads, head_dim]
    v_buffer = v_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]

    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)

    # Step 2: Compute q@K
    qk = torch.matmul(q, k_buffer.transpose(-1, -2))  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    # Step 3: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    # Step 4: Compute o = softmax(q@K) @ V
    o = torch.matmul(qk, v_buffer) # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    o = o.view(batch, num_q_heads, head_dim)    
    return o


def measure_time(func, *args, warmup=10, repeat=100):
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        args: Arguments to pass to the function
        warmup: Number of warmup iterations
        repeat: Number of measurement iterations
    
    Returns:
        Average execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(repeat):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) * 1000 / repeat  # Convert to ms

if __name__ == '__main__':
    # Create random tensors with shapes matching the function signature
    batch = 1
    num_q_heads = 32
    num_kv_heads = 1
    head_dim = 512
    prefill_kv_len = 1024*128*2
    decode_kv_len = 128
    rank = 512
    
    dtype = torch.float16

    # Calculate derived values
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    
    # Create random tensors (using uniform [-0.5, 0.5])
    q = (torch.rand(batch, num_q_heads, head_dim, dtype=dtype, device='cuda') - 0.5)
    k_A_buffer = (torch.rand(batch, prefill_kv_len, rank, dtype=dtype, device='cuda') - 0.5)
    k_B_buffer = (torch.rand(batch, num_kv_heads, rank, head_dim, dtype=dtype, device='cuda') - 0.5)
    v_A_buffer = (torch.rand(batch, prefill_kv_len, rank, dtype=dtype, device='cuda') - 0.5)
    v_B_buffer = (torch.rand(batch, num_kv_heads, rank, head_dim, dtype=dtype, device='cuda') - 0.5)
    k_buffer_decoded = (torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=dtype, device='cuda') - 0.5)*20
    v_buffer_decoded = (torch.rand(batch, num_kv_heads, decode_kv_len, head_dim, dtype=dtype, device='cuda') - 0.5)*20
    k_buffer_prefill = torch.matmul(k_A_buffer.unsqueeze(1), k_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    v_buffer_prefill = torch.matmul(v_A_buffer.unsqueeze(1), v_B_buffer) # [batch, num_kv_heads, prefill_kv_len, head_dim]
    sm_scale = 1.0


    k_buffer_prefill = k_buffer_prefill.transpose(1, 2)
    v_buffer_prefill = v_buffer_prefill.transpose(1, 2)
    k_buffer_decoded = k_buffer_decoded.transpose(1, 2)
    v_buffer_decoded = v_buffer_decoded.transpose(1, 2)
    
    k_buffer = torch.cat([k_buffer_prefill, k_buffer_decoded], dim=1)
    v_buffer = torch.cat([v_buffer_prefill, v_buffer_decoded], dim=1)


    
    o = mla_decode_attention_fwd(q, k_buffer, v_buffer, sm_scale)
    o_sep= mla_decode_attention_fwd_pd_sep(q, k_buffer_prefill, v_buffer_prefill, k_buffer_decoded, v_buffer_decoded, sm_scale)
    print("Maximum difference between separated and non-separated implementation:", torch.max(torch.abs(o_sep - o)))
    o_xkv = mla_decode_attention_fwd_pd_sep_xKV(
        q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    o_ref = decode_attention_reference(q, k_buffer, v_buffer, sm_scale)
    print("Maximum difference between separated and xKV implementation:", torch.max(torch.abs(o_sep - o_xkv)))
    print("Maximum difference between separated and reference implementation:", torch.max(torch.abs(o_sep - o_ref)))
    print("Maximum difference between triton and reference implementation:", torch.max(torch.abs(o - o_ref)))
    # Measure performance
    print("\n=== Performance Measurements ===")
    
    # Measure non-compiled functions
    time_regular = measure_time(
        mla_decode_attention_fwd, q, k_buffer, v_buffer, sm_scale
    )
    time_sep = measure_time(
        mla_decode_attention_fwd_pd_sep, q, k_buffer_prefill, v_buffer_prefill, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    time_xkv = measure_time(
        mla_decode_attention_fwd_pd_sep_xKV, q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer, k_buffer_decoded, v_buffer_decoded, sm_scale
    )
    
    print(f"Regular implementation: {time_regular:.3f} ms")
    print(f"Separated implementation: {time_sep:.3f} ms")
    print(f"xKV implementation: {time_xkv:.3f} ms")
    print(f"Speedup: {time_regular/time_sep:.2f}x")
    
    