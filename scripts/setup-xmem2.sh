#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv-sam2"
xmem_root="${project_root}/models/xmem2"
checkpoint="${xmem_root}/saves/XMem.pth"
cd "${project_root}"

# Some shared shells export the literal documentation placeholder
# ``proxy_ip:port``. Git/curl reject it before making a network request.
for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do
  if [[ "${!proxy_var:-}" == *"proxy_ip:port"* ]]; then
    unset "${proxy_var}"
  fi
done
git_proxy_args=()
curl_proxy_args=()
if [[ -n "${XMEM2_PROXY:-}" ]]; then
  git_proxy_args=(-c "http.proxy=${XMEM2_PROXY}" -c "https.proxy=${XMEM2_PROXY}")
  curl_proxy_args=(--proxy "${XMEM2_PROXY}")
else
  configured_git_proxy="$(git config --get http.proxy 2>/dev/null || true)"
  if [[ "${configured_git_proxy}" == *"proxy_ip:port"* ]]; then
    git_proxy_args=(-c http.proxy= -c https.proxy=)
    curl_proxy_args=(--noproxy '*')
  fi
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  echo "Missing ${venv_dir}. Create the current SAM 2 environment before installing XMem++." >&2
  exit 2
fi

# Install a CUDA-enabled PyTorch build. CUDA 12.1 is compatible with the
# installed NVIDIA driver (535) and H100 GPU.
if ! "${venv_dir}/bin/python" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  "${venv_dir}/bin/python" -m pip install \
    --upgrade --force-reinstall --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
fi

if [[ ! -f "${xmem_root}/process_video.py" ]]; then
  git "${git_proxy_args[@]}" clone --depth 1 https://github.com/mbzuai-metaverse/XMem2.git "${xmem_root}"
fi
# XMem++'s original command-line loader assumes CUDA checkpoints are restored
# on a CUDA host. Keep the upstream checkout usable in this project's CPU
# environment as well. These two replacements are idempotent.
"${venv_dir}/bin/python" - <<'PY'
from pathlib import Path
root = Path("models/xmem2")
replacements = {
    root / "model/network.py": (
        "torch.load(model_path, map_location=map_location)",
        "torch.load(model_path, map_location=map_location or ('cuda' if torch.cuda.is_available() else 'cpu'))",
    ),
    root / "inference/run_on_video.py": (
        "torch.load(model_path)",
        "torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')",
    ),
    root / "inference/inference_core.py": (
        "torch.zeros((1, 3, 480, 854), device='cuda:0')",
        "torch.zeros((1, 3, 480, 854), device='cuda:0' if torch.cuda.is_available() else 'cpu')",
    ),
}
for path, (old, new) in replacements.items():
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
PY
# The configured package mirror rejects the authenticated GitHub proxy. When a
# proxy is supplied, use PyPI proper; otherwise retain the machine's mirror.
if [[ -n "${XMEM2_PROXY:-}" ]]; then
  env http_proxy="${XMEM2_PROXY}" https_proxy="${XMEM2_PROXY}" HTTP_PROXY="${XMEM2_PROXY}" HTTPS_PROXY="${XMEM2_PROXY}" ALL_PROXY='' all_proxy='' PIP_INDEX_URL='https://pypi.org/simple' "${venv_dir}/bin/python" -m pip install progressbar2 pandas tqdm
else
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY "${venv_dir}/bin/python" -m pip install progressbar2 pandas tqdm
fi
mkdir -p "$(dirname "${checkpoint}")"
if [[ ! -f "${checkpoint}" ]] || ! "${venv_dir}/bin/python" -c "import torch; torch.load(r'${checkpoint}', map_location='cpu', weights_only=False)" >/dev/null 2>&1; then
  curl "${curl_proxy_args[@]}" --continue-at - --retry 5 --retry-all-errors --retry-delay 2 -fL https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth -o "${checkpoint}"
fi
"${venv_dir}/bin/python" - <<'PY'
import sys
from pathlib import Path
root = Path.cwd() / "models" / "xmem2"
sys.path.insert(0, str(root))
from inference.run_on_video import run_on_video
print("XMem++ ready:", root)
PY
