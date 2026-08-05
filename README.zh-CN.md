<div align="center">

# TempSamp-R1：面向视频大语言模型的高效时序采样与强化微调

**Yunheng Li**<sup>1</sup> · **Jing Cheng**<sup>1</sup> · **Shaoyong Jia**<sup>2</sup> · **Hangyi Kuang**<sup>1</sup> · **Shaohui Jiao**<sup>2</sup> · **Qibin Hou**<sup>1†</sup> · **Ming-Ming Cheng**<sup>1</sup>

<sup>1</sup> VCIP，南开大学 &nbsp;&nbsp; <sup>2</sup> 字节跳动

<sup>†</sup> 通讯作者

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.18056)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-1f6feb.svg)](https://arxiv.org/abs/2509.18056)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

> [!TIP]
> 如果你觉得这个项目有用，欢迎给仓库点个 ⭐ 并引用我们的论文，这能帮助项目成长并让更多人发现它。引用信息见文末。

> [!NOTE]
> 🏆 **AIC 高光剪辑赛道 / Highlight Video Re-framing competition**：可运行的基线工具包（极简居中裁剪 + Qwen-VL 两阶段）位于 [`baseline/`](baseline/)。任务定义、提交格式和运行方式见 [baseline/README.md](baseline/README.md)。

本仓库是 **TempSamp-R1** 的开源实现，面向**视频时序定位（video temporal grounding）**任务，构建在 [EasyR1](https://github.com/hiyouga/EasyR1) / [verl](https://github.com/volcengine/verl) 强化学习训练栈之上。相比原始 GRPO，它额外贡献了两点：

1. **GT 注入（GT injection）**：把每个 rollout 组中的一个槽位替换为真实答案（mix-policy GRPO）。
2. **非线性奖励整形（non-linear reward shaping）**：在计算 advantage 之前，对高奖励做对数压缩、对低奖励做指数放大。

GT 注入位于 [`verl/trainer/ray_trainer.py`](verl/trainer/ray_trainer.py)（`RayPPOTrainer._inject_gt_rollout_in_gen_output`）；奖励整形（`transform_rewards`）和 GT 回复构建（`build_gt_response`）位于 [`scripts/timelens/timelens_reward.py`](scripts/timelens/timelens_reward.py)。

## 结果（Results）

在 **TimeLens-Bench**（Charades / ActivityNet / QVHighlights-TimeLens）上的表现。**TempSamp-R1-4B** = Qwen3.5-4B + GT 注入 + 奖励整形。各列中，最高分用**粗体琥珀色**标出，第二名用 <u>下划线</u> 标出。

<p align="center">
  <img src="docs/assets/timelens_bench_full.svg" alt="TimeLens-Bench full results table" width="960"/>
</p>

<details style="margin: 1.5em 0;">
<summary><h2 style="display: inline-block; margin: 0 0 0.75em 0; padding: 0;">安装（Installation）</h2></summary>

下面的版本组合是我们使用 **8 × NVIDIA H20（sm_90，CUDA 12.6 runtime）** 和 Qwen3.5-4B 完整端到端验证过的版本。较新的工具链（CUDA 12.8 PyTorch wheel）与 CUDA 12.6 驱动向前兼容。

| 软件包 | 版本 |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0 + cu128 |
| Triton | 3.6.0（随 torch 一起安装） |
| vLLM | 0.19.1 |
| transformers | 5.5.4 |
| flash-attn | 2.8.1 |
| flash-linear-attention | 0.4.2 |
| ray | 2.54.0 |
| qwen-vl-utils | 0.0.14 |
| decord | 0.6.0 |

```bash
conda create -n tempsamp-verl python=3.12 -y
conda activate tempsamp-verl

# 1) torch + triton（Triton 3.6 随 cu128 wheel 一起安装）
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) 项目 + 固定依赖（使用仓库内可编辑安装的 verl）
pip install -e .

# 3) flash-attn：安装与 torch + cu + cp + ABI 匹配的预编译 wheel。
#    先检查你的 cxx11 ABI 标志：
#      python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
#    然后从
#      https://github.com/Dao-AILab/flash-attention/releases
#    下载匹配的 wheel
#    示例（torch 2.10 / cu12 / cp312 / ABI=True）：
pip install flash-attn==2.8.1 --no-build-isolation
```

环境自检：

```bash
python - <<'PY'
import torch, vllm, transformers, flash_attn
print('torch       :', torch.__version__, 'cuda', torch.version.cuda)
print('vllm        :', vllm.__version__)
print('transformers:', transformers.__version__)
print('flash_attn  :', flash_attn.__version__)
print('GPU         :', torch.cuda.get_device_name(0))
print('compute cap :', torch.cuda.get_device_capability(0))
PY
```

> **注意：** **不要**执行 `pip install opencv-python`，它会引入无头服务器上缺失的 `libGL.so.1` 依赖，并在导入阶段导致 vLLM 评测崩溃。`requirements.txt` 改而固定使用 API 兼容的 `opencv-python-headless`。

请根据你的硬件调整脚本中的 `NPROC_PER_NODE`、`CUDA_VISIBLE_DEVICES` 和 `worker.rollout.tensor_parallel_size`。

</details>

<details style="margin: 1.5em 0;">
<summary><h2 style="display: inline-block; margin: 0 0 0.75em 0; padding: 0;">数据（Data）</h2></summary>

使用 **TimeLens-100K** 训练，并在 **TimeLens-Bench** 的 3 个子任务上评测。推荐的目录结构如下：

```
datasets/
├── TimeLens-100K/
│   ├── videos/*.mp4
│   └── (可选) preprocessed_videos/*.pt
└── TimeLens-Bench/
    ├── charades-timelens.json
    ├── activitynet-timelens.json
    ├── qvhighlights-timelens.json
    └── video_shards/{charades,activitynet,qvhighlights}/*.mp4
```

流水线分两个阶段：**阶段 A 是必须的**；**阶段 B 可选但建议执行**（它避免每个 epoch 都重新解码视频，能显著节省训练时间）。

### 阶段 A：把原始 JSONL 转换为训练器 JSONL（填充绝对视频路径）

```bash
python data/convert_timelens_to_verl.py \
    --input      /path/to/timelens-100k.jsonl \
    --output     data/timelens_grpo_train.jsonl \
    --video_root /path/to/datasets/TimeLens-100K
```

`--video_root` 会被拼接到源数据中每一条相对视频路径前，保证输出的 JSONL 始终包含完整解析后的绝对路径。

我们提供了用于复现论文结果的**精确 2000 行训练子集**，位于 [`data/timelens_grpo_train.jsonl`](data/timelens_grpo_train.jsonl)（这是 TimeLens-100K 按难度分层采样的子集）。其中视频路径保持相对路径形式，即 `video_shards/<source>/<vid>.mp4`，因此训练前你需要二选一：(a) 用本地的 `--video_root` 重新运行阶段 A 将路径绝对化，或 (b) 把 `data.image_dir` / `--image_dir` 指向你的 TimeLens-100K 根目录。

每行输出示例：

```json
{
  "problem": "<video>To accurately pinpoint the event \"...\" in the video, ... <answer> 12.5 to 17.8 </answer>.",
  "answer":  "<answer> 7.0 to 11.0 </answer>",
  "problem_type": "temporal grounding",
  "data_type": "video",
  "videos": ["/abs/path/to/video.mp4"]
}
```

### 阶段 B：（可选）把视频离线解码成 `.pt` 缓存

训练器默认的 rollout 路径会在每个 epoch 解码一次视频。离线解码一次后，每个 step 的视频 I/O 就变成对单个 `.pt` 文件的一次 `torch.load`。这里的参数（fps / pixel caps / max_frames）必须与训练 YAML 保持一致，否则加载器会静默回退到实时解码。

```bash
INPUT_FILE=data/timelens_grpo_train.jsonl \
OUTPUT_DIR=data/preprocessed_videos \
OUTPUT_FILE=data/timelens_grpo_train.preprocessed.jsonl \
  bash scripts/preprocess_videos.sh
```

然后把训练 YAML 指向预处理缓存：

```yaml
data:
  train_files:             data/timelens_grpo_train.preprocessed.jsonl
  use_preprocessed_videos: true
  video_source_mode:       prefer_preprocessed   # 或 preprocessed_only
  preprocessed_video_dir:  data/preprocessed_videos
```

（这些配置已通过环境变量接入 `scripts/train/timelens_*.sh`。启动训练前设置 `TIMELENS_USE_PREPROCESSED=true`、`TIMELENS_VIDEO_SOURCE_MODE=prefer_preprocessed`、`TIMELENS_PREPROCESSED_VIDEO_DIR=...` 和 `TIMELENS_TRAIN_FILES=...preprocessed.jsonl` 即可。）

</details>

## 训练（Training）

```bash
# 原始 GRPO 基线
TIMELENS_MODEL_PATH=/path/to/Qwen3.5-4B \
TIMELENS_TRAIN_FILES=data/timelens_grpo_train.jsonl \
  bash scripts/train/timelens_grpo.sh

# TempSamp-R1（GT 注入 + 奖励整形）
TIMELENS_MODEL_PATH=/path/to/Qwen3.5-4B \
TIMELENS_TRAIN_FILES=data/timelens_grpo_train.jsonl \
  bash scripts/train/timelens_tempsamp.sh
```

## 评测（Evaluation）

```bash
# 使用 vLLM 在 TimeLens-Bench（全部 3 个子任务）上做单步评测
BASE_MODEL=/path/to/Qwen3.5-4B \
BENCH_DIR=/path/to/datasets/TimeLens-Bench \
  bash scripts/eval/eval_timelens_bench.sh /path/to/checkpoint/global_step_NNN/actor
```

也可以只评测单个数据集：

```bash
DATASETS=charades-timelens bash scripts/eval/eval_timelens_bench.sh /path/to/actor
```

## 引用（Citation）

如果你觉得我们的工作对研究有帮助，欢迎给仓库点个 ⭐ 并引用我们的论文。感谢支持！

```bibtex
@inproceedings{li2026tempsamp,
  title     = {TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs},
  author    = {Li, Yunheng and Cheng, Jing and Jia, Shaoyong and Kuang, Hangyi and Jiao, Shaohui and Hou, Qibin and Cheng, Ming-Ming},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  pages     = {40692--40716},
  year      = {2026}
}
```

## 致谢（Acknowledgements）

项目基于 [verl](https://github.com/volcengine/verl)、[EasyR1](https://github.com/hiyouga/EasyR1)、[vLLM](https://github.com/vllm-project/vllm) 和 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 构建，以 [Apache 2.0](LICENSE) 协议发布。
