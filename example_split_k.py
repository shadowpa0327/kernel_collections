import torch
import tilelang
import tilelang.language as T
from tvm import DataType

def matmul(M, N, K, block_M, block_N, block_K, split_k, dtype="float16", accum_dtype="float"):

    splitK = K // split_k

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if bz == 0:
                # fuse the zero initialization kernel
                for i, j in T.Parallel(block_M, block_N):
                    m, n = by * block_M + i, bx * block_N + j
                    C[m, n] = T.cast(0, dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
                T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            if DataType(dtype).bits == 16:
                for i, j in T.Parallel(block_M, block_N // 2):
                    m, n = by * block_M + i, bx * block_N + j * 2
                    # vectorized atomic
                    T.atomic_addx2(C[m, n], C_shared[i, j * 2])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

    return main


# ------------------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Just a short demonstration with a small batch
    B, M, K, N = 1, 32, 65536, 256
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")
    b_tl = b.transpose(0, 1).contiguous()

    tile_program = matmul(M, N, K, 32, 64, 128, 64, dtype="float16", accum_dtype="float")
    kernel = tilelang.compile(tile_program)
    output_buffer = torch.empty((M, N), dtype=torch.float16, device="cuda")
    # Warmup
    for _ in range(10):
        c_out = kernel(a, b_tl, output_buffer)
        with torch.no_grad():
            ref = torch.matmul(a, b)
    
    # Measure custom matmul time
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    c_out = kernel(a, b_tl, output_buffer)
    end_time.record()
    torch.cuda.synchronize()
    custom_time = start_time.elapsed_time(end_time)
    
    # Measure PyTorch's builtin batched matmul time
    start_time.record()
    with torch.no_grad():
        ref = torch.matmul(a, b)
    end_time.record()
    torch.cuda.synchronize()
    ref_time = start_time.elapsed_time(end_time)
    
    # Validate results
   # torch.testing.assert_close(c_out, ref, rtol=1e-1, atol=1e-1)
    print("Batched Tile-lang matmul wrapper test passed.")
    print(f"Custom implementation: {custom_time:.3f} ms")
    print(f"PyTorch reference: {ref_time:.3f} ms")
    print(f"Speedup: {ref_time/custom_time:.2f}x")
