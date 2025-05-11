# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

import triton
import triton.language as tl

import torch

logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
logger.warning(
    "The following error message 'operation scheduled before its operands' can be ignored."
)


_MIN_BLOCK_KV = 32

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_A_Buffer, # shape: [batch, kv_len, rank]
    K_B_Buffer, # shape: [batch, num_kv_heads, rank, head_dim]
    V_A_Buffer, # shape: [batch, kv_len, rank]
    sm_scale,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kA_bs,
    stride_buf_kA_ls,
    stride_buf_kB_bs,
    stride_buf_kB_hs,
    stride_buf_kB_rs,
    stride_buf_vbs,
    stride_buf_vls,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_seq_len: tl.constexpr,
    kv_group_num: tl.constexpr,
    rank_k: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_r = tl.arange(0, BLOCK_R)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]


    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    partial_qk_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    partial_exp_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # reconstruct K here.
            ## Load A_K_Buffer and B_K_Buffer
            k = tl.zeros((BLOCK_DMODEL, BLOCK_N), dtype=tl.float32)
            for i in range(0, tl.cdiv(rank_k, BLOCK_R)):
                offs_buf_kA = (
                    cur_batch * stride_buf_kA_bs
                    + offs_n[None, :] * stride_buf_kA_ls
                    + offs_r[:, None] + i * BLOCK_R
                ) 
                
                kA = tl.load(
                    K_A_Buffer + offs_buf_kA, 
                    mask=(offs_n[None, :] < split_kv_end), 
                    other=0.0
                ) # [BLOCK_R, BLOCK_N]
                offs_buf_kB = (
                    cur_batch * stride_buf_kB_bs
                    + cur_kv_head * stride_buf_kB_hs
                    + (offs_r + i * BLOCK_R)[None, :] * stride_buf_kB_rs
                    + offs_d[:, None]
                ) #[BLOCK_D, BLOCK_R]

                kB = tl.load(
                    K_B_Buffer + offs_buf_kB,
                ) # [BLOCK_D, BLOCK_R]
                k = tl.dot(kB, kA, k) # [BLOCK_D, BLOCK_N]
            
            qk = tl.dot(q, k.to(q.dtype))
            qk *= sm_scale
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                cur_batch * stride_buf_vbs  
                + offs_n[:, None] * stride_buf_vls
                + offs_dv[None, :]
            )
            v = tl.load(
                V_A_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            new_partial_qk_max = tl.maximum(tl.max(qk, 1), partial_qk_max)
            re_scale = tl.exp(partial_qk_max - new_partial_qk_max)
            p = tl.exp(qk - new_partial_qk_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            partial_exp_sum = partial_exp_sum * re_scale + tl.sum(p, 1)
            partial_qk_max = new_partial_qk_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / partial_exp_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            partial_qk_max + tl.log(partial_exp_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_A_buffer,
    k_B_buffer,
    v_A_buffer,
    att_out,
    att_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
):
    BLOCK = 32
    Dk = k_B_buffer.shape[-1]
    Dv = v_A_buffer.shape[-1]

    BLOCK_DMODEL = triton.next_power_of_2(Dk)
    BLOCK_DV = triton.next_power_of_2(Dv)
    
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_B_buffer.shape[1]
    kv_len = k_A_buffer.shape[1]
    rank_k = k_A_buffer.shape[-1]
    BLOCK_R = 64 #NOTE(brian1009): Not sure whether we will get oom or not
    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_A_buffer,
        k_B_buffer,
        v_A_buffer,
        sm_scale,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_A_buffer.stride(0),
        k_A_buffer.stride(1),
        k_B_buffer.stride(0),
        k_B_buffer.stride(1),
        k_B_buffer.stride(2),
        v_A_buffer.stride(0),
        v_A_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        kv_seq_len=kv_len,
        rank_k=rank_k,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        num_warps=4,
        num_stages=num_stages,
        Lk=Dk,
        Lv=Dv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    attn_logits,
    attn_lse,
    o,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    kv_seq_len: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    global_exp_sum = 0.0
    global_lse = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                attn_logits + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            in_coming_lse = tl.load(attn_lse + offs_logic + split_kv_id * stride_mid_os // Lv)
            new_global_lse = tl.maximum(in_coming_lse, global_lse)
            old_scale = tl.exp(global_lse - new_global_lse)
            acc *= old_scale
            exp_logic = tl.exp(in_coming_lse - new_global_lse)
            acc += exp_logic * tv

            global_exp_sum = global_exp_sum * old_scale + exp_logic
            global_lse = new_global_lse

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / global_exp_sum,
        mask=mask_d,
    )

def _decode_softmax_reducev_fwd(
    attn_logits,
    attn_lse,
    q,
    o,
    v_A_buffer,
    v_B_buffer,
    num_kv_splits,
    max_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_B_buffer.shape[-1]
    kv_seq_len = v_A_buffer.shape[1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits

    extra_kargs = {}
    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        attn_logits,
        attn_lse,
        o,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        kv_seq_len=kv_seq_len,
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )

def decode_attention_fwd(
    q,                  # shape: [batch, num_q_heads, head_dim]
    k_A_buffer,         # shape: [batch, kv_len, rank]
    k_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    v_A_buffer,         # shape: [batch, kv_len, rank]
    v_B_buffer,         # shape: [batch, num_kv_heads, rank, head_dim]
    o,                  # shape: [batch, num_q_heads, head_dim]
    num_kv_splits,      # shape: [batch]
    max_kv_splits,      # int
    sm_scale=1.0,       # float: softmax scaling factor
):    
    """
    Memory-efficient grouped query attention implementation for decoding.
    
    This function implements GQA (Grouped Query Attention) in a memory-efficient way by
    splitting the KV cache into multiple chunks and processing them sequentially.
    The implementation follows a two-stage approach:
    1. Compute partial attention scores and outputs for each KV split
    2. Combine the partial results to produce the final attention output
    
    Parameters:
    -----------
    q : torch.Tensor
        Query tensor of shape [batch, num_q_heads, head_dim]
    k_A_buffer : torch.Tensor
        Key cache buffer of shape [batch, kv_len, rank]
    k_B_buffer : torch.Tensor
        Key cache buffer of shape [batch, num_kv_heads, rank, head_dim]
    v_A_buffer : torch.Tensor
        Value cache buffer of shape [batch, kv_len, rank]
    v_B_buffer : torch.Tensor
        Value cache buffer of shape [batch, num_kv_heads, rank, head_dim]
    o : torch.Tensor
        Output tensor of shape [batch, num_q_heads, head_dim]
        Will be filled with the result of attention computation
    num_kv_splits : torch.Tensor
        Tensor of shape [batch] containing the number of KV splits for each batch item
        Used to handle different sequence lengths efficiently
    max_kv_splits : int
        Maximum number of KV splits across all batches
        Determines the size of intermediate buffers
    sm_scale : float, optional
        Softmax scaling factor (1/sqrt(head_dim)), defaults to 1.0
        
    Returns:
    --------
    torch.Tensor
        The same tensor as the input 'o', now filled with attention output
        
    Notes:
    ------
    - The implementation uses Triton kernels for efficient parallel computation
    - KV splitting allows processing of very long sequences with limited memory
    - Each KV split is processed independently, and results are combined with proper normalization
    - For GQA, kv_group_num = num_q_heads // num_kv_heads (each KV head serves multiple Q heads)
    """
    
    # Calculate kv_group_num (ratio of query heads to key/value heads)
    assert v_A_buffer.shape[1] == k_A_buffer.shape[1]
    assert v_B_buffer.shape[1] == k_B_buffer.shape[1]
    num_kv_heads = v_B_buffer.shape[1]
    kv_group_num = q.shape[1] // num_kv_heads
    assert kv_group_num >= 1
    batch, num_q_heads, head_dim = q.shape
    rank_v = v_A_buffer.shape[-1]

    # Create intermediate tensors for attention computation
    # attn_logits: Stores partial attention outputs for each KV split
    # attn_lse: Stores log-sum-exp values for numerical stability
    attn_logits = torch.empty(
        (batch, num_q_heads, max_kv_splits, rank_v),
        dtype=torch.float32, 
        device=q.device
    )
    attn_lse = torch.empty(
        (batch, num_q_heads, max_kv_splits),
        dtype=torch.float32,
        device=q.device
    )

    # Stage 1: Compute attention scores and partial output for each KV split
    _decode_grouped_att_m_fwd(
        q,
        k_A_buffer,
        k_B_buffer,
        v_A_buffer,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )
    # Stage 2: Multiply with v_B_buffer
    attn_logits = attn_logits.reshape(batch, num_kv_heads, -1, rank_v)
    attn_logits = torch.matmul(attn_logits.to(v_B_buffer.dtype), v_B_buffer).to(torch.float32) # [batch, num_kv_heads, kv_group_num*max_kv_splits, head_dim]
    attn_logits = attn_logits.reshape(batch, num_q_heads, -1, head_dim)
    
    # Stage 3: Combine partial outputs from all KV splits
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_A_buffer,
        v_B_buffer,
        num_kv_splits,
        max_kv_splits,
    )
    
    return o