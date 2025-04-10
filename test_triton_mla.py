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


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
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
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
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
    v_buffer,
    num_kv_splits,
    max_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    kv_seq_len = v_buffer.shape[1]
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

def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= attn_logits.shape[0]

    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        num_kv_splits,
        max_kv_splits,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[2]
    assert kv_group_num >= 1

    # GQA/MQA/MLA
    decode_attention_fwd_grouped(
        q,
        k_buffer,
        v_buffer,
        o,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )


def decode_attention_reference(q, k_buffer, v_buffer, o, sm_scale):
    # Step 1: Store the size 
    batch, num_q_heads, head_dim = q.shape
    batch, kv_len, num_kv_heads, head_dim = k_buffer.shape
    
    num_q_heads_per_kv_group = num_q_heads // num_kv_heads
    assert batch== 1
    #k_buffer = k_buffer.unsqueeze(0) # [batch, kv_len, num_kv_heads, head_dim]
    k_buffer = k_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]
    
    #v_buffer = v_buffer.unsqueeze(0) # [batch, kv_len, num_kv_heads, head_dim]
    v_buffer = v_buffer.transpose(1, 2) # [batch, num_kv_heads, kv_len, head_dim]

    q = q.view(batch, num_kv_heads, num_q_heads_per_kv_group, head_dim)

    # Step 2: Compute q@K
    qk = torch.matmul(q, k_buffer.transpose(-1, -2))
    # Step 3: Compute softmax
    qk = qk / sm_scale
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(qk.dtype)  # [batch, num_kv_heads, num_q_heads_per_kv_group, kv_len]
    # Step 4: Compute o = softmax(q@K) @ V
    o = torch.matmul(qk, v_buffer) # [batch, num_kv_heads, num_q_heads_per_kv_group, head_dim]
    o = o.view(batch, num_q_heads, head_dim)
    return o

if __name__ == "__main__":
   # ---------------------------
    # 1) BASIC PARAMETERS
    # ---------------------------
    batch       = 1
    q_len       = 1
    kv_len      = 256*1024
    num_q_heads = 32
    num_kv_heads = 1
    head_dim    = 384
    sm_scale    = 1.0

    # Create random Q, K, V, O with the shapes you specified.
    # Here we assume you want them on CUDA, half-precision (float16):
    dtype = torch.float16
    device = 'cuda'

    # q:  (q_len, num_q_heads, head_dim) --> reshape to (batch, num_q_heads, head_dim)
    q = torch.randn(batch, q_len*num_q_heads, head_dim, dtype=dtype, device=device) # shape: (1, 32, 128)

    # k_buffer, v_buffer: (kv_len, num_kv_heads, head_dim)
    k_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # o:  (q_len, num_q_heads, head_dim) or similarly (batch, heads, head_dim)
    o = torch.zeros_like(q)

    # ---------------------------
    # 2) CREATE kv_indptr, kv_indices
    # ---------------------------
    # Since batch=1, we only need to say that:
    #   - kv_indptr[0] = 0
    #   - kv_indptr[1] = kv_len (4096)
    #
    # This means the entire [0..4096) range of kv_indices belongs to batch 0.
    #kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

    # kv_indices will be a simple range [0..4096).
    # kv_indices = torch.arange(kv_len, dtype=torch.int32, device=device)

    # ---------------------------
    # 3) DEFINE THE "SPLITS" AND INTERMEDIATE BUFFERS
    # ---------------------------
    # Suppose we do 1 KV-split for this entire chunk.
    num_kv_splits = torch.tensor([128], dtype=torch.int32, device=device)
    max_kv_splits = 128

    # If your kernel is using "attn_logits" to store partial results per split,
    # you typically want shape = (batch, num_q_heads, max_kv_splits, head_dim_of_V).
    # Here head_dim_of_V = 128, same as K, so:
    attn_logits = torch.zeros(
        (batch, num_q_heads, max_kv_splits, head_dim),
        dtype=dtype,
        device=device,
    )

    # Likewise for log-sum-exp array, shape = (batch, num_q_heads, max_kv_splits)
    attn_lse = torch.zeros(
        (batch, num_q_heads, max_kv_splits),
        dtype=dtype,
        device=device,
    )

    # Example usage:
    decode_attention_fwd(
        q, 
        k_buffer, 
        v_buffer, 
        o,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap=0.0,
    )

    # Now 'o' would contain the final result after "attention" (in real code).
    print("o shape:", o.shape)            # Should be [1, 32, 128]
    print("attn_logits shape:", attn_logits.shape)  # [1, 32, 1, 128]
    print("attn_lse shape:", attn_lse.shape)        # [1, 32, 1]

    o_ref = decode_attention_reference(q, k_buffer, v_buffer, o, sm_scale)
    print(torch.max(torch.abs(o_ref-o)))
    print("Ouput aligned with reference:", torch.allclose(o, o_ref, atol=5e-2, rtol=5e-2))

    # Test performance
    import time
    
    # Warm-up runs
    for _ in range(10):
        decode_attention_fwd(
            q, 
            k_buffer, 
            k_buffer, 
            o,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=0.0,
        )
    
    # Measure time for triton implementation
    torch.cuda.synchronize()
    start_time = time.time()
    num_runs = 100
    
    for _ in range(num_runs):
        decode_attention_fwd(
            q, 
            k_buffer, 
            k_buffer, 
            o,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=0.0,
        )
    
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs
    print(f"Triton implementation average time: {triton_time*1000:.4f} ms")
    
    # Measure time for reference implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        decode_attention_reference(
            q, 
            k_buffer, 
            v_buffer, 
            o,
            sm_scale,
        )
    
    torch.cuda.synchronize()
    reference_time = (time.time() - start_time) / num_runs
    print(f"Reference implementation average time: {reference_time*1000:.4f} ms")
    print(f"Speedup: {reference_time/triton_time:.2f}x")
    
    
    # # flash attention
    # from flash_attn import flash_attn_with_kvcache
    
    # q = q.reshape(batch, 1, num_q_heads, head_dim)
    # k_buffer = k_buffer.reshape(batch, kv_len, num_kv_heads, head_dim)
    # v_buffer = v_buffer.reshape(batch, kv_len, num_kv_heads, head_dim)
    
    # print("flash attention")
    # # Warm-up runs
    # for _ in range(10):
    #     flash_attn_with_kvcache(q, k_buffer, v_buffer)
    
    # # Measure time for triton implementation
    # torch.cuda.synchronize()
    # start_time = time.time()
    # num_runs = 100
    # for _ in range(num_runs):
    #     flash_attn_with_kvcache(q, k_buffer, v_buffer)
    # torch.cuda.synchronize()
    # flash_time = (time.time() - start_time) / num_runs
    # print(f"Flash attention average time: {flash_time*1000:.4f} ms")
    # print(f"Speedup over Triton: {triton_time/flash_time:.2f}x")