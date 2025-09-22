################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import torch
from kernels import shadowkv

last_q = 64
arange = torch.arange(last_q, device="cuda")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    # if position_ids shape is (batch_size, num_heads, seq_len), then reshape it to (batch_size*num_heads, seq_len)
    if len(position_ids.shape) == 3:
        position_ids = position_ids.view(-1, position_ids.size(-1))
        cos = cos[position_ids]
        sin = sin[position_ids]
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def apply_rotary_pos_emb_cuda(x, cos_sin, position_ids):
    batch_size, heads, seq_len, embed_dim = x.shape
    half_dim = embed_dim // 2

    # NOTE(brian1009): If position_ids is 2D [batch_size, seq_len], expand it to 3D [batch_size, heads, seq_len]
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(1).expand(-1, heads, -1).contiguous()

    output = torch.empty_like(x)
    shadowkv.apply_rotary_pos_emb_new(
        x, cos_sin, position_ids, output,
        int(batch_size), int(heads), int(seq_len), int(embed_dim),
        int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
        int(cos_sin.stride(0)),
        int(position_ids.stride(0)), int(position_ids.stride(1)), int(position_ids.stride(2)),
        int(half_dim)
    )
    
    return output

def apply_rotary_pos_emb_cuda_push_cache(x, cos_sin, position_ids, chunk_size, cache, sparse_start, sparse_end, cnts):
    batch_size, heads, seq_len, embed_dim = x.shape
    half_dim = embed_dim // 2

    if cos_sin.shape[-1] == 128:
        shadowkv.apply_rotary_pos_emb_push_cache_opt(
            x, cos_sin, position_ids, cache, cnts,
            int(batch_size), int(heads), int(seq_len), int(embed_dim),
            int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
            int(cos_sin.stride(0)),
            int(position_ids.stride(0)), int(position_ids.stride(1)), int(position_ids.stride(2)),
            int(cache.stride(0)), int(cache.stride(1)), int(cache.stride(2)),
            int(sparse_start), int(sparse_end),
            int(half_dim), int(chunk_size)
        )
    elif cos_sin.shape[-1] == 64:
        shadowkv.apply_rotary_pos_emb_push_cache_opt_glm(
            x, cos_sin, position_ids, cache, cnts,
            int(batch_size), int(heads), int(seq_len), int(embed_dim),
            int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
            int(cos_sin.stride(0)),
            int(position_ids.stride(0)), int(position_ids.stride(1)), int(position_ids.stride(2)),
            int(cache.stride(0)), int(cache.stride(1)), int(cache.stride(2)),
            int(sparse_start), int(sparse_end),
            int(half_dim), int(chunk_size)
        )
    else:
        raise ValueError(f"Invalid cos_sin shape {cos_sin.shape}")
    
    return cache

def batch_gather_gemm_rotary_pos_emb_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    output: torch.Tensor,
    chunk_size: int,
    cache: torch.Tensor,
    sparse_start: int,
    sparse_end: int,
    cnts: torch.Tensor,
    no_rope: bool = False
):
    batch_size, seq_len, rank = a.shape
    _, heads, head_dim, _ = b.shape
    max_seq_len, _ = cos_sin.shape
    _, _, num_chunks = position_ids.shape
    sparse_budget = num_chunks * chunk_size
    position_ids = position_ids.to(torch.int32).contiguous()
    
    # print(f"a shape: {a.shape}")
    # print(f"b shape: {b.shape}")
    # print(f"cos_sin shape: {cos_sin.shape}")
    # print(f"position_ids shape: {position_ids.shape}")
    # print(f"output shape: {output.shape}")
    # print(f"cache shape: {cache.shape}")
    # print(f"sparse_start: {sparse_start}, sparse_end: {sparse_end}")
    # print(f"cnts shape: {cnts.shape}, cnts: {cnts}")
    # exit()

    # NOTE(max410011): cos_sin is the same afer the kernel
    shadowkv.batch_gather_gemm(
        a.contiguous(),
        b.contiguous(),
        cos_sin.contiguous(),
        cos_sin.contiguous(),
        position_ids,
        output,
        batch_size,
        heads,
        seq_len,
        head_dim,
        rank,
        sparse_budget,
        max_seq_len,
        chunk_size,
        cnts,
    )
    if no_rope:
        return output
    else:
        return apply_rotary_pos_emb_cuda_push_cache(output, cos_sin, position_ids, chunk_size, cache, sparse_start, sparse_end, cnts)

def sample(probs : torch.Tensor, num_samples=1):
    idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    return idx_next

