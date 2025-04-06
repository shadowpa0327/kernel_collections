# xKV System
This repository contains the experimental system implementation for xKV.

## Installation
```bash
conda create -n xkv_system python=3.11
conda activate xkv_system
pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```

## Attention Implementations
+ `test_attention.py` contains the implementation of the attention mechanism.
    + `group_query_attention` is the eager implementation of the attention mechanism.
    + `group_query_attention_fa` is flash decoding
    + `group_query_attention_factorized` is a implementation of the attention mechanism with both K and V factorized.
    + `group_query_attention_factorized_v_only` is a implementation of the attention mechanism with only V factorized.

## Fused Key Reconstruction
+ `test_fused_key_reconstruction.py` contains the implementation of the fused key reconstruction.
    + `_qAB_fwd` is the forward pass of the fused key reconstruction.
