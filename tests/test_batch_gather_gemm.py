
import torch
import pytest
import sys
import os

# Add the parent directory to sys.path to import ops
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ops.batch_gather_gemm.batch_gather_gemm_reference import batch_gather_gemm_reference
from ops.batch_gather_gemm.triton_batch_gather_gemm_clean import (
    batch_gather_gemm_triton,
    batch_gather_gemm_rotary_pos_emb_triton
)

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBatchGatherGemm:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.device = 'cuda'
        self.dtype = torch.bfloat16
        
        # Common configuration
        self.batch_size = 2
        self.heads = 4
        self.seq_len = 128
        self.rank = 64
        self.head_dim = 128
        self.chunk_size = 8
        self.num_chunks = 10
        self.max_seq_len = 256
        self.sparse_budget = self.num_chunks * self.chunk_size

    def _create_inputs(self):
        a = torch.randn(self.batch_size, self.seq_len, self.rank, dtype=self.dtype, device=self.device)
        b = torch.randn(self.batch_size, self.heads, self.head_dim, self.rank, dtype=self.dtype, device=self.device)
        
        # RoPE cache
        half_dim = self.head_dim // 2
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32, device=self.device) / half_dim))
        freqs = torch.outer(positions, inv_freq)
        cos_cache = torch.cos(freqs).to(self.dtype)
        sin_cache = torch.sin(freqs).to(self.dtype)
        cos_sin = torch.cat([cos_cache, sin_cache], dim=-1)
        
        # Position IDs
        max_chunk_id = self.seq_len // self.chunk_size - 1
        position_ids = torch.randint(0, max_chunk_id + 1, (self.batch_size, self.heads, self.num_chunks), dtype=torch.int32, device=self.device)
        
        # Cnts (all chunks valid by default)
        cnts = torch.zeros(self.batch_size * self.heads, dtype=torch.int32, device=self.device)
        
        return a, b, cos_sin, position_ids, cnts

    def test_batch_gather_gemm_correctness(self):
        a, b, _, position_ids, cnts = self._create_inputs()
        
        # Run Triton
        output_triton = batch_gather_gemm_triton(
            a=a.contiguous(),
            b=b.contiguous(),
            position_ids=position_ids,
            chunk_size=self.chunk_size,
            cnts=cnts
        )
        
        # Run Reference
        output_ref = batch_gather_gemm_reference(
            a=a.float(),
            b=b.float(),
            cos_sin=torch.empty(0), # Not used when apply_rope=False
            position_ids=position_ids,
            chunk_size=self.chunk_size,
            apply_rope=False
        ).to(self.dtype)
        
        # Compare
        # bfloat16 precision can be tricky, using a generous tolerance
        assert torch.allclose(output_triton.float(), output_ref.float(), atol=1e-2, rtol=1e-2)

    def test_batch_gather_gemm_with_rope(self):
        a, b, cos_sin, position_ids, cnts = self._create_inputs()
        
        cache_triton = torch.zeros(self.batch_size, self.heads, self.max_seq_len, self.head_dim, dtype=self.dtype, device=self.device)
        output_triton = torch.zeros(self.batch_size, self.heads, self.sparse_budget, self.head_dim, dtype=self.dtype, device=self.device)
        
        # Run Triton
        batch_gather_gemm_rotary_pos_emb_triton(
            a=a,
            b=b,
            cos_sin=cos_sin,
            position_ids=position_ids,
            output=output_triton,
            chunk_size=self.chunk_size,
            cache=cache_triton,
            sparse_start=0,
            sparse_end=self.max_seq_len,
            cnts=cnts,
            no_rope=False
        )
        
        # Run Reference
        output_ref = batch_gather_gemm_reference(
            a=a.float(),
            b=b.float(),
            cos_sin=cos_sin.float(),
            position_ids=position_ids,
            chunk_size=self.chunk_size,
            apply_rope=True
        ).to(self.dtype)
        
        # Check cache content (which should match output_ref for the written parts)
        # The reference returns the output *after* RoPE.
        # The Triton kernel writes this same output to the cache.
        # We need to extract the relevant parts from the cache to compare.
        # However, the Triton kernel *also* returns the cache.
        # Wait, the reference implementation returns [batch, heads, sparse_budget, head_dim]
        # The Triton implementation writes to [batch, heads, max_seq_len, head_dim] at specific indices.
        # But for verification, we can just check if the values written to cache match the reference output.
        
        # Actually, let's look at how test_triton_fixed.py did it.
        # It compared cache_triton[:, :, :sparse_budget, :] with output_ref.
        # This assumes sparse_start=0 and contiguous writing which matches the reference logic?
        # In reference: output is [batch, heads, sparse_budget, head_dim]
        # In Triton: it writes to cache based on sparse_rows = SPARSE_START + rm
        # rm goes from 0 to M (sparse_budget).
        # So if SPARSE_START=0, it writes to cache[:, :, 0:sparse_budget, :]
        
        result_triton = cache_triton[:, :, :self.sparse_budget, :]
        
        # Compare
        # RoPE involves more ops, so tolerance might need to be slightly higher
        # But test_triton_fixed used rtol=2e-1, atol=2.5 which is very loose.
        # Let's try something reasonable first.
        diff = (result_triton.float() - output_ref.float()).abs().max().item()
        print(f"Max diff with RoPE: {diff}")
        
        assert torch.allclose(result_triton.float(), output_ref.float(), atol=2.0, rtol=1e-1)

    def test_batch_gather_gemm_cnts(self):
        a, b, cos_sin, position_ids, _ = self._create_inputs()
        
        # Set some cnts
        # cnts controls how many chunks to SKIP from the beginning.
        # Let's skip 2 chunks for the first head of first batch.
        cnts = torch.zeros(self.batch_size * self.heads, dtype=torch.int32, device=self.device)
        cnts[0] = 2 
        
        # Run Triton
        output_triton = batch_gather_gemm_triton(
            a=a.contiguous(),
            b=b.contiguous(),
            position_ids=position_ids,
            chunk_size=self.chunk_size,
            cnts=cnts
        )
        
        # Run Reference
        output_ref = batch_gather_gemm_reference(
            a=a.float(),
            b=b.float(),
            cos_sin=torch.empty(0),
            position_ids=position_ids,
            chunk_size=self.chunk_size,
            apply_rope=False,
            cnts=cnts
        ).to(self.dtype)
        
        assert torch.allclose(output_triton.float(), output_ref.float(), atol=1e-2, rtol=1e-2)
