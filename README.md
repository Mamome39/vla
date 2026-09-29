# vla-arch

## Setup

```bash
conda env create -f environment.yml
conda activate vla_arch
```

Torch is pinned to `2.11.0+cu128`: the driver supports CUDA 12.8, and newer torch on PyPI is built for cu130.

## Qwen3.5-2B

```python
from vla_arch.backbones import Qwen35_2B

bb = Qwen35_2B()                                        # downloads on first use
feats, mask = bb(images=[[img]], texts=["pick up the red cube"])   # (B, T, 2048), (B, T)
bb.chat(images=[[img]], texts=["What is in this image?"])          # text answers
```

Try it: `python scripts/smoke_backbone.py`
