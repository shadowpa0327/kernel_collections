
import torch

def qAB_ref(q: torch.Tensor, Ak: torch.Tensor, Bk: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, q_len, head_dim = q.shape
    bsz_a, kv_len, rank = Ak.shape
    bsz_b, num_kv_heads, rank, head_dim = Bk.shape
    
    num_query_heads_per_group = num_heads // num_kv_heads
    assert bsz == bsz_a == bsz_b, "Batch size mismatch"

    q = q.view(bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim).contiguous()
    
    # multiply Ak and Bk
    # Reconstruct the key matrix by multiplying Ak and Bk
    # Ak: (bsz, kv_len, rank)
    # Bk: (bsz, num_kv_heads, rank, head_dim)
    # First compute Ak @ Bk to get reconstructed keys
    reconstructed_k = torch.matmul(Ak.unsqueeze(1), Bk)  # (bsz, num_kv_heads, kv_len, head_dim)
    
    # Now compute q @ reconstructed_k.transpose(-1, -2)
    # q: (bsz, num_kv_heads, num_query_heads_per_group*q_len, head_dim)
    # reconstructed_k: (bsz, num_kv_heads, kv_len, head_dim)
    out = torch.matmul(q, reconstructed_k.transpose(-1, -2))  # (bsz, num_kv_heads, num_query_heads_per_group*q_len, kv_len)

    return out.view(bsz, num_heads, q_len, kv_len)