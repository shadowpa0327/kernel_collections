import transformers
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from flash_recontruction.rope.sin_cos import rotary_sin_cos

sin_triton, cos_triton = rotary_sin_cos(
    seqlen=4096,
    theta=10000000.0,
    starting_idx=0,
    block_size=32,   # try 16 / 32 / 64, benchmark and pick the best
)

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
qwen2_rope = Qwen2RotaryEmbedding(config)

position_ids = torch.arange(4096).unsqueeze(0).to("cuda")
x = torch.randn(1, 4096, 128, dtype=torch.float32, device="cuda")
cos, sin = qwen2_rope(x, position_ids)
cos = cos.squeeze(0).cuda().float()
sin = sin.squeeze(0).cuda().float()

# Check if cosine values match between Triton and Qwen2 implementations
cos_match = torch.allclose(cos_triton[:, :64], cos[:, :64], atol=1e-3, rtol=1e-3)
print(f"Cosine values match: {cos_match}")

# Check if sine values match between Triton and Qwen2 implementations
sin_match = torch.allclose(sin_triton[:, :64], sin[:, :64], atol=1e-3, rtol=1e-3)
print(f"Sine values match: {sin_match}")

# Maximum absolute difference in cosine values
cos_max_diff = torch.max(torch.abs(cos_triton[:, :64]-cos[:, :64]))
print(f"Maximum absolute difference in cosine values: {cos_max_diff:.6f}")

# Maximum absolute difference in sine values
sin_max_diff = torch.max(torch.abs(sin_triton[:, :64]-sin[:, :64]))
print(f"Maximum absolute difference in sine values: {sin_max_diff:.6f}")