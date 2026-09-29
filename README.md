# GhostHands

两条并行路线，共用一个 Python 环境 `.venv-run` 和同一批输入素材。

| 路线 | 实现 | 入口 | 输出 |
| --- | --- | --- | --- |
| 切割（手部分割 + 特效渲染） | skin / MediaPipe / SAM 2 / XMem++ → mask → 蓝色发光效果 | `scripts/segment.ps1` | `outputs/segmentation/<实验名>/<后端>/` |
| 直接生成（视频编辑模型） | DashScope / Runway 直接编辑视频 | `scripts/direct.ps1` | `outputs/direct/<模型>/<实验名>/` |

## 目录

```text
pipeline/
├── .venv-run/             唯一运行环境，两条路线共用
├── .env                  API 密钥（本地保存，不提交）
├── .env.example          配置示例
├── scripts/              两条路线的 PowerShell 入口
├── src/ghosthands/        共享 Python 包
│   ├── backends.py        手部分割后端
│   ├── pipeline.py        切割、渲染和数据集导出
│   ├── render.py          特效渲染
│   ├── report.py          切割结果质检报告
│   ├── direct.py          直接生成服务接入
│   └── cli.py             统一 CLI
├── input/                原始视频和当前输入（保留既有路径）
├── clips/                短视频及 smoke 测试素材
├── config/               模型候选清单
└── outputs/              所有新结果统一放在这里
    ├── segmentation/     切割实验：<实验名>/<后端>/
    ├── direct/           直接生成：<模型>/<实验名>/
    └── checks/           测试输出；保留 unified-env 最近验证
```

`input/` 会被递归扫描；需要固定对比素材时，直接指定一个视频文件。两条路线可以在两个终端同时运行，使用不同的输出目录。重复使用输出目录会覆盖对应结果，请为新实验使用新名称。

## 环境

现有环境可直接使用，无需激活。保留 `.venv-run` 名称是为了兼容已有任务和虚拟环境内的绝对路径。新机器上从项目根目录安装：

```powershell
py -3.11 -m venv .venv-run
.\.venv-run\Scripts\python.exe -m pip install -e ".[mediapipe,runway,dashscope]"
.\.venv-run\Scripts\python.exe -m pip check
```

切割路线还需要 `ffmpeg` 在 PATH 中。两条入口均从项目根目录执行，示例路径也相对于项目根目录。

## 路线一：切割

```powershell
# 原始分支：MediaPipe 关键点拟合 mask
.\scripts\segment.ps1 process .\input\test_00_05.mp4 .\outputs\segmentation\mediapipe-01\mediapipe --backend mediapipe --seed 7

# 精细分支：MediaPipe 定位 + GrabCut 像素边界细化
.\scripts\segment.ps1 process .\input\test_00_05.mp4 .\outputs\segmentation\refined-01 --branch refined --seed 7

# 使用相同素材和种子比较后端
.\scripts\segment.ps1 compare .\input\test_00_05.mp4 .\outputs\segmentation\comparison-02 --backends 'skin,mediapipe' --seed 7

# 重建质检报告
.\scripts\segment.ps1 report .\outputs\segmentation\mediapipe-01\mediapipe
```

每个样本输出 `source.mp4`、`target.mp4`、逐帧 PNG mask、`metadata.json` 和 `quality.json`，后端目录包含 `report.html`；compare 实验根目录包含 `comparison.json`。`skin` 是肤色阈值基线；MediaPipe 根据手部关键点构建 mask；SAM 2 仍是待接入占位后端。

SAM 2 是独立的时序分割分支：MediaPipe 自动提供手框和 21 个关节点正提示，SAM 2 在视频内推断精确轮廓并做时序追踪。默认每 12 帧重新提示一次，以便恢复被遮挡或后进入画面的手；在 `config/sam2.yaml` 设 `prompt_mode: initial` 可复现单提示基线，设 `prompt_type: mask` 可复现旧的关节点填充 mask 基线。运行 `scripts/setup-sam2.ps1` 完成本机安装；在 Linux 服务器使用 `scripts/setup-sam2.sh`。先用 2–5 秒视频验证，再在 GPU 节点批量处理。

### 处理分支开关

原代码保留为 `classic` 分支。新增的 `refined` 分支先运行相同的 MediaPipe 手部定位，再以原帧像素和颜色模型执行受限 GrabCut，使 mask 边界贴近手部；它不会扩张到原手形以外的大范围区域。

默认分支变量位于 `config/pipeline.yaml`：

```yaml
active_branch: classic  # 改为 refined 后，未显式给出 --backend/--branch 的 process 命令走精细分支
```

单次覆盖：`--branch classic` 或 `--branch refined`。对同一批视频比较两条分支：

```powershell
.\scripts\segment.ps1 compare-branches .\input\test_00_05.mp4 .\outputs\segmentation\branch-comparison-01 --branches 'classic,refined' --seed 7
```

每个样本的 `metadata.json` 会记录 `processing_branch`，方便训练时筛选。

### XMem++ 时序分割分支

`xmem2` 使用 [XMem++](https://github.com/mbzuai-metaverse/XMem2) 的永久记忆式视频分割：先由 MediaPipe 为每只手自动生成稀疏参考 mask，再由 XMem++ 在全片传播。默认每三分之一秒保留一帧参考（`config/xmem2.yaml` 的 `seed_stride: 10`），并会在新的手首次出现时额外保留一帧；这能更快纠正姿态变化和遮挡后的漂移。传播结果会在原始分辨率上执行受 XMem 掩码约束的局部 GrabCut 边缘贴合（`edge_refinement: true`），只在轮廓周围 `edge_margin` 像素内调整，避免误吸附远处的肤色背景。上游代码采用 GPL-3.0，源码和模型仅放在忽略提交的 `models/xmem2/`。

当前 Linux 环境复用已可运行的 `.venv-sam2`，首次执行：

```bash
bash scripts/setup-xmem2.sh
.venv-sam2/bin/python -m ghosthands.cli process clips/test-5s.mp4 outputs/segmentation/xmem2-01 --branch xmem2 --seed 7
```

安装脚本会获取 XMem++ 源码、命令行依赖与 `XMem.pth` 权重。当前环境为 CPU；短片可以验证流程，长视频应预留较长运行时间。运行完成后，正常输出逐帧 mask、发光视频、质检报告；`metadata.json` 的 `backend_metadata` 会记录实际种子帧和对象数。与其他路线比较：

如果网络需要代理，执行安装时设置 `XMEM2_PROXY=http://host:port`；脚本会忽略机器环境中误填的 `proxy_ip:port` 占位代理。

```powershell
.\scripts\segment.ps1 compare-branches .\input\test_00_05.mp4 .\outputs\segmentation\xmem2-comparison --branches 'sam2,xmem2' --seed 7
```

## 路线二：直接生成

在 `.env` 中通过 `DIRECT_ENGINE` 固定处理分支：`wan`、`ltx`、`runway`，或使用默认 `auto` 保持旧的 provider 分流。Wan / Runway 会提交真实 API 任务并可能产生费用；LTX 在本机 ComfyUI 中运行，不调用云端 API。

```powershell
.\scripts\direct.ps1 dashscope .\input\test_00_05.mp4 .\outputs\direct\wan27\run-02 --model wan2.7-videoedit --seed 7

.\scripts\direct.ps1 runway .\input\test_00_05.mp4 .\outputs\direct\runway\run-01 --model gemini_omni_flash --seed 7
```

LTX 使用单独的本地分支，不改动 Wan 代码。先在 ComfyUI 安装 LTX 节点和模型，做出一个可以读取视频、接收提示词并保存 MP4 的视频到视频工作流，然后用 ComfyUI 的 **Save (API Format)** 导出节点 JSON。将该工作流内以下值写成占位符：`__SOURCE_VIDEO__`、`__PROMPT__`、`__SEED__`、`__OUTPUT_PREFIX__`；若工作流使用参考图，再加入 `__REFERENCE_IMAGE__`。随后在 `.env` 配置 `DIRECT_ENGINE=ltx`、`LTX_COMFYUI_INPUT_DIR` 和 `LTX_COMFYUI_WORKFLOW`，并启动 ComfyUI：

```powershell
.\scripts\direct.ps1 ltx .\input\test_00_05.mp4 .\outputs\direct\ltx\run-01 --seed 7
```

LTX 分支会把输入视频复制到 ComfyUI 的 `input/` 目录，提交工作流、轮询完成状态并下载结果到 `target.mp4`。它只适合短片和低分辨率验证；工作流节点名称及编号由本机的 ComfyUI/LTX 版本决定，因此保留在独立 JSON 中，而非写死到原有代码里。

输出包含 `target.mp4`、`metadata.json` 和 `provider-task.json`。已有 Wan 结果位于 `outputs/direct/wan27/test_00_05_seed7/`。直接生成模型可能改变手部结构、动作或背景，结果需要人工比对。

查看帮助：`scripts/segment.ps1 --help`、`scripts/direct.ps1 --help`。完整 CLI：`.venv-run/Scripts/python.exe -m ghosthands.cli --help`。


## 实验与日志约定

每次运行使用独立实验名，日志 `run.stdout.log`、`run.stderr.log` 放在对应实验根目录。检查性运行只写入 `outputs/checks/`。CLI 参数与两个 PowerShell 入口保持不变，输出路径仍需显式传入。

本次重跑使用 `input/test.mp4`、`skin,mediapipe` 和 seed 7，输出至 `outputs/segmentation/experiment-01/`；不包含新增短片。

模型候选清单位于 `config/direct-model-candidates.json`，刷新命令：

```powershell
.\.venv-run\Scripts\python.exe -m ghosthands.cli direct-candidates .\config
```
