$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-run\Scripts\python.exe'
$sam2Root = Join-Path $projectRoot 'models\sam2'
$checkpoint = Join-Path $sam2Root 'checkpoints\sam2.1_hiera_small.pt'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.pip-cache'
& $python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if (-not (Test-Path $sam2Root)) { git clone https://github.com/facebookresearch/sam2.git $sam2Root }
Push-Location $sam2Root; try { $env:SAM2_BUILD_CUDA = '0'; & $python -m pip install -e . } finally { Pop-Location }
New-Item -ItemType Directory -Force (Split-Path -Parent $checkpoint) | Out-Null
if (-not (Test-Path $checkpoint)) { Invoke-WebRequest 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt' -OutFile $checkpoint }
& $python -c "import torch; from sam2.build_sam import build_sam2_video_predictor; print('SAM 2 ready:', torch.cuda.get_device_name(0))"
