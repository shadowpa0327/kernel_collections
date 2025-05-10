from flash_decoding.mla_xKV import decode_attention_partial_reference
from flash_decoding.mla import _decode_grouped_att_m_fwd
import torch
import time
import numpy as np


if __name__ == "__main__":
    # Create random tensors with the correct shapes
    batch = 1
    num_q_heads = 32
    num_kv_heads = 1
    head_dim = 512
    kv_len = 64
    sm_scale = 1.0
    
    # Create tensors according to the function signature
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.float16, device="cuda")
    k_buffer = torch.randn(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16, device="cuda")
    v_buffer = torch.randn(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16, device="cuda")
    
    decode_attention_partial_reference_compiled = torch.compile(decode_attention_partial_reference)

    # Warm-up runs
    for _ in range(10):
        decode_attention_partial_reference(q, k_buffer, v_buffer, sm_scale)
    
    torch.cuda.synchronize()
    
    # Measure latency
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        output, lse = decode_attention_partial_reference(q, k_buffer, v_buffer, sm_scale)
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Average latency over {num_runs} runs: {avg_time_ms:.2f} ms")
    
    # Measure latency for compiled version
    decode_attention_partial_reference_compiled = torch.compile(decode_attention_partial_reference)
    
    # Warm-up runs for compiled version
    for _ in range(10):
        decode_attention_partial_reference_compiled(q, k_buffer, v_buffer, sm_scale)
    
    torch.cuda.synchronize()
    
    # Measure latency for compiled version
    start_time = time.time()
    
    for _ in range(num_runs):
        output_compiled, lse_compiled = decode_attention_partial_reference_compiled(q, k_buffer, v_buffer, sm_scale)
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time_compiled_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Average latency for compiled version over {num_runs} runs: {avg_time_compiled_ms:.2f} ms")
    print(f"Speedup from compilation: {avg_time_ms / avg_time_compiled_ms:.2f}x")
    

    k_buffer_T = k_buffer.transpose(1, 2)
    v_buffer_T = v_buffer.transpose(1, 2)

    num_kv_splits = 2
    prefill_output = torch.empty(
        (batch, num_q_heads, num_kv_splits, head_dim),
        dtype=q.dtype, 
        device=q.device
    )
    prefill_lse = torch.empty(
        (batch, num_q_heads, num_kv_splits),
        dtype=torch.float32,
        device=q.device
    )

    from functools import partial

    fun = partial(torch.compile(_decode_grouped_att_m_fwd),
        q,
        k_buffer_T,
        v_buffer_T,
        prefill_output,
        prefill_lse,
        torch.ones(batch, dtype=torch.int32, device=q.device) * num_kv_splits,
        num_kv_splits,
        sm_scale,
        0.0,
    )
    
    # testing time
    for _ in range(10):
        fun()

    # Measure latency for compiled version
    start_time = time.time()
    
    for _ in range(num_runs):
        fun()
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time_compiled_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Average latency for compiled version over {num_runs} runs: {avg_time_compiled_ms:.2f} ms")
    print(f"Speedup from kernel: {avg_time_ms / avg_time_compiled_ms:.2f}x")

    print(prefill_output)
    print(prefill_lse)