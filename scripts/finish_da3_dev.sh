#!/usr/bin/env bash
set -euo pipefail

python=/lumos-vePFS/suzhou/Users/lixiaotong/third_party/da3_venv/bin/python
export HTTPS_PROXY=http://127.0.0.1:18888
export HTTP_PROXY=http://127.0.0.1:18888
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120

"$python" -m pip install opencv-python-headless evo
"$python" - <<'PY'
from depth_anything_3.api import DepthAnything3

print("DA3_IMPORT_OK", flush=True)
DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
print("DA3_WEIGHTS_OK", flush=True)
PY
