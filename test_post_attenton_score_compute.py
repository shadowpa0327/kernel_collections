import torch

def run_attention_benchmark(
    batch_size=1,
    num_heads=32,
    num_kv_heads=8,
    kv_len=65536,
    head_dim=128,
    rank=256,
    num_trials=100,
    warmup_trials=10
):
    """Run benchmark comparing standard attention vs. optimized factorized attention."""
    
    # Calculate derived parameters
    group_size = num_heads // num_kv_heads
    
    # Create input tensors
    attn_score = torch.randn(batch_size, num_heads, kv_len).half().cuda()  # (bsz, q_head, kv_len)
    Ak = torch.randn(batch_size, kv_len, rank).half().cuda()  # (bsz, kv_len, rank)
    Bk = torch.randn(batch_size, num_kv_heads, rank, head_dim).half().cuda()  # (bsz, num_kv_heads, rank, d_v)
    value = torch.randn(batch_size, num_kv_heads, kv_len, head_dim).half().cuda()  # (bsz, num_kv_heads, kv_len, d_v)

    def normal_attn(attn_score, value):
        """Standard attention computation.
        
        Args:
            attn_score: Tensor of shape (bsz, q_head, kv_len)
            value: Tensor of shape (bsz, num_kv_heads, kv_len, d_v)
        
        Returns:
            Tensor of shape (bsz, num_kv_heads, q_len*group_size, d_v)
        """
        return torch.matmul(attn_score.reshape(batch_size, num_kv_heads, group_size, kv_len), value)

    def our_attn(attn_score, Ak, Bv):
        """Optimized attention computation using factorized matrices.
        
        Args:
            attn_score: Tensor of shape (bsz, q_head, kv_len)
            Ak: Tensor of shape (bsz, kv_len, rank)
            Bv: Tensor of shape (bsz, num_kv_heads, rank, d_v)
        
        Returns:
            Tensor of shape (bsz, num_kv_heads, q_len*group_size, d_v)
        """
        temp = torch.matmul(attn_score, Ak)
        temp = temp.reshape(batch_size, num_kv_heads, group_size, rank)  # (bsz, num_kv_heads, group_size, rank)
        return torch.matmul(temp, Bv)

    # Example usage
    attn_out = our_attn(attn_score, Ak, Bk)
    print(f"Optimized attention output shape: {attn_out.shape}")

    attn_out2 = normal_attn(attn_score, value)
    print(f"Standard attention output shape: {attn_out2.shape}")

    # Warmup
    for _ in range(warmup_trials):
        _ = our_attn(attn_score, Ak, Bk)
        _ = normal_attn(attn_score, value)
    torch.cuda.synchronize()

    # Measure our_attn latency
    our_attn_times = []
    for _ in range(num_trials):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        _ = our_attn(attn_score, Ak, Bk)
        end_time.record()
        torch.cuda.synchronize()
        our_attn_times.append(start_time.elapsed_time(end_time))

    # Measure normal_attn latency
    normal_attn_times = []
    for _ in range(num_trials):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        _ = normal_attn(attn_score, value)
        end_time.record()
        torch.cuda.synchronize()
        normal_attn_times.append(start_time.elapsed_time(end_time))

    avg_our_attn_time = sum(our_attn_times) / len(our_attn_times)
    avg_normal_attn_time = sum(normal_attn_times) / len(normal_attn_times)

    print(f"Optimized attention time: {avg_our_attn_time:.4f} ms")
    print(f"Standard attention time: {avg_normal_attn_time:.4f} ms")
    print(f"Speedup: {avg_normal_attn_time/avg_our_attn_time:.2f}x")
    
    return {
        "optimized_time_ms": avg_our_attn_time,
        "standard_time_ms": avg_normal_attn_time,
        "speedup": avg_normal_attn_time/avg_our_attn_time
    }

# Run the benchmark with default parameters
if __name__ == "__main__":
    run_attention_benchmark()
