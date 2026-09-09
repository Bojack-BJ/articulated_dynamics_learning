#!/usr/bin/env bash
set -euo pipefail

root=/lumos-vePFS/suzhou/Users/lixiaotong/third_party
env_dir="$root/da3_venv"
src_dir="$root/Depth-Anything-3-main"
base_python=/root/Users/miniconda3/envs/particulate/bin/python

"$base_python" -m venv --system-site-packages "$env_dir"
export HTTP_PROXY=http://127.0.0.1:18888
export HTTPS_PROXY=http://127.0.0.1:18888
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=2
"$env_dir/bin/python" -c 'import hatchling' 2>/dev/null || \
  "$env_dir/bin/python" -m pip install hatchling hatch-vcs
"$env_dir/bin/python" -m pip install e3nn pycolmap pillow_heif 'moviepy==1.0.3'
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEPTH_ANYTHING_3=0.0.0 \
  "$env_dir/bin/python" -m pip install --no-deps -e "$src_dir"

"$env_dir/bin/python" - <<'PY'
import torch
from depth_anything_3.api import DepthAnything3

print("DA3_IMPORT_OK", torch.__version__, torch.cuda.is_available())
PY

HF_HUB_DOWNLOAD_TIMEOUT=300 "$env_dir/bin/python" - <<'PY'
from depth_anything_3.api import DepthAnything3

DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
print("DA3_METRIC_LARGE_CACHED")
PY
