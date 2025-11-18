
```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip

uv pip install torch==2.6.0
uv pip install setuptools
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# flashinfer
uv pip install flashinfer-python==0.3.1

# cutlass
mkdir 3rdparty
git clone https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass

# build kernels for ShadowKV
python setup.py build_ext --inplace