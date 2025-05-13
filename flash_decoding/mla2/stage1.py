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
# https://github.com/sgl-project/sglang/blob/f8194b267c71d79fca30e0bd8d01c086969abd4e/python/sglang/srt/layers/attention/triton_ops/decode_attention.py



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
    Q_nope,
    Q_rope,
    KV_nope,
    K_rope,
    sm_scale,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_kv_bs,
    stride_kv_ls,
    stride_kv_h,
    stride_k_rope_bs,
    stride_k_rope_ls,
    stride_k_rope_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_seq_len: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    DMODEL: tl.constexpr,
    DPE: tl.constexpr,
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

    offs_d_nope = tl.arange(0, BLOCK_DMODEL)
    offs_d_rope = tl.arange(0, BLOCK_DPE)
    mask_d_nope = offs_d_nope < DMODEL
    mask_d_rope = offs_d_rope < DPE

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q_nope = cur_batch * stride_q_nope_bs + cur_head[:, None] * stride_q_nope_h + offs_d_nope[None, :]
    off_q_rope = (
        cur_batch * stride_q_rope_bs + cur_head[:, None] * stride_q_rope_h + offs_d_rope[None, :]
    )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)
    partial_qk_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    partial_exp_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q_nope = tl.load(Q_nope + offs_q_nope, mask=(mask_h[:, None]) & (mask_d_nope[None, :]), other=0.0)
        q_rope = tl.load(
            Q_rope + off_q_rope, mask=(mask_h[:, None]) & (mask_d_rope[None, :]), other=0.0
        )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # Load KV to KV^T
            offs_buf_k = (
                #kv_loc[None, :] * stride_buf_kbs
                cur_batch * stride_kv_bs
                + offs_n[None, :] * stride_kv_ls 
                + cur_kv_head * stride_kv_h
                + offs_d_nope[:, None]
            )
            kv_nope = tl.load(
                KV_nope + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d_nope[:, None]),
                other=0.0,
            )
            qk = tl.dot(q_nope, kv_nope.to(q_nope.dtype))
            offs_buf_k_rope = (
                cur_batch * stride_k_rope_bs
                + offs_n[None, :] * stride_k_rope_ls
                + cur_kv_head * stride_k_rope_h
                + offs_d_rope[:, None]
            )
            k_rope = tl.load(
                K_rope + offs_buf_k_rope,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d_rope[:, None]),
                other=0.0,
            )
            qk += tl.dot(q_rope, k_rope.to(q_rope.dtype))
            qk *= sm_scale
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            new_partial_qk_max = tl.maximum(tl.max(qk, 1), partial_qk_max)
            re_scale = tl.exp(partial_qk_max - new_partial_qk_max)
            p = tl.exp(qk - new_partial_qk_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(kv_nope.dtype), tl.trans(kv_nope))

            partial_exp_sum = partial_exp_sum * re_scale + tl.sum(p, 1)
            partial_qk_max = new_partial_qk_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d_nope[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / partial_exp_sum[:, None],
            mask=(mask_h[:, None]) & (mask_d_nope[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // DMODEL

        tl.store(
            Att_Lse + offs_mid_o_1,
            partial_qk_max + tl.log(partial_exp_sum),
            mask=mask_h,
        )


def decoding_stage1(
    q_nope,
    q_rope,
    kv_nope,
    k_rope,
    attn_out,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
):
    BLOCK = 32
    DMODEL = q_nope.shape[-1]
    DPE = q_rope.shape[-1]

    
    BLOCK_DMODEL = triton.next_power_of_2(DMODEL)
    BLOCK_DPE = triton.next_power_of_2(DPE)

    batch, head_num = q_nope.shape[0], q_nope.shape[1]
    kv_group_num = q_nope.shape[1] // kv_nope.shape[2]
    kv_seq_len = kv_nope.shape[1]
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
        q_nope,
        q_rope,
        kv_nope,
        k_rope,
        sm_scale,
        attn_out,
        attn_lse,
        num_kv_splits,
        q_rope.stride(0),  # stride_q_rope_bs
        q_rope.stride(1),  # stride_q_rope_h
        q_nope.stride(0),  # stride_q_nope_bs
        q_nope.stride(1),  # stride_q_nope_h
        kv_nope.stride(0),  # stride_kv_bs
        kv_nope.stride(1),  # stride_kv_ls
        kv_nope.stride(2),  # stride_kv_h
        k_rope.stride(0),  # stride_k_rope_bs
        k_rope.stride(1),  # stride_k_rope_ls
        k_rope.stride(2),  # stride_k_rope_h
        attn_out.stride(0),  # stride_mid_ob
        attn_out.stride(1),  # stride_mid_oh
        attn_out.stride(2),  # stride_mid_os
        kv_seq_len=kv_seq_len,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        DMODEL=DMODEL,
        DPE=DPE,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )
