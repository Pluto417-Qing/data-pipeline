#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv-sam2"
sam2_root="${project_root}/models/sam2"
checkpoint="${sam2_root}/checkpoints/sam2.1_hiera_small.pt"
torch_index="${SAM2_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
pypi_index="${SAM2_PYPI_INDEX:-https://pypi.org/simple}"

if [[ -n "${SAM2_PROXY:-}" ]]; then
  export http_proxy="${SAM2_PROXY}"
  export https_proxy="${SAM2_PROXY}"
  export HTTP_PROXY="${SAM2_PROXY}"
  export HTTPS_PROXY="${SAM2_PROXY}"
fi

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install torch torchvision --index-url "${torch_index}"
"${venv_dir}/bin/python" -m pip install -e "${project_root}[mediapipe]" --index-url "${pypi_index}"
if [[ ! -f "${sam2_root}/pyproject.toml" ]]; then
  echo "Missing ${sam2_root}. Transfer the SAM 2 source into models/sam2 before running this script." >&2
  exit 2
fi
"${venv_dir}/bin/python" -m pip install -e "${sam2_root}" --index-url "${pypi_index}"
mkdir -p "$(dirname "${checkpoint}")"
if [[ ! -f "${checkpoint}" ]]; then
  curl -fL "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -o "${checkpoint}"
fi
"${venv_dir}/bin/python" - <<'PY'
import torch
from sam2.build_sam import build_sam2_video_predictor
print("SAM 2 ready; CUDA available:", torch.cuda.is_available())
PY
