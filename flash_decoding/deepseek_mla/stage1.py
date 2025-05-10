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
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kls,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vls,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_seq_len: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
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
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

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
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # kv_loc = tl.load(
            #     kv_indices + cur_batch_kv_start_idx + offs_n,
            #     mask=offs_n < split_kv_end,
            #     other=0,
            # )
            # Load K to K^T
            offs_buf_k = (
                #kv_loc[None, :] * stride_buf_kbs
                cur_batch * stride_buf_kbs
                + offs_n[None, :] * stride_buf_kls 
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    #kv_loc[None, :] * stride_buf_kbs
                    cur_batch * stride_buf_kbs
                    + offs_n[None, :] * stride_buf_kls
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                #kv_loc[:, None] * stride_buf_vbs
                cur_batch * stride_buf_vbs
                + offs_n[:, None] * stride_buf_vls
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
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


def decoding_stage1(
    q,
    k_buffer,
    v_buffer,
    attn_out,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    Dk = k_buffer.shape[-1]
    Dv = v_buffer.shape[-1]

    if Dk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Dk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Dk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Dv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[2]
    kv_seq_len = k_buffer.shape[1]
    BLOCK_H = 32
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
        k_buffer,
        v_buffer,
        sm_scale,
        attn_out,
        attn_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        kv_seq_len=kv_seq_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Dk,
        Lv=Dv,
        **extra_kargs,
    )
