## Ship It

A provisioning setup that lives in your head is not shipped. The deliverable is a script that any teammate can run to verify their environment before they start a training run or deploy an inference endpoint. This script becomes step one in every deployment runbook.

The following script checks four things in order: (1) is there an NVIDIA GPU visible to the OS, (2) can PyTorch see it through CUDA, (3) what is the VRAM headroom, and (4) do the driver and CUDA versions match. It exits 0 on success and 1 on any failure, so it can be wired into CI or a Makefile target.

```python
import subprocess
import sys

def check_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi