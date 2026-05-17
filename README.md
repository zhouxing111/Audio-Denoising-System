# 智能音频降噪系统 (Intelligent Audio Denoising System)

基于 Python 的工业级智能音频降噪系统，集成传统信号处理算法与深度学习端到端模型，提供完整的训练流水线和交互式图形界面。

## 核心功能

- **双模式音源输入**：支持 WAV/FLAC/MP3/M4A 文件加载和麦克风在线录音（最长 30s，实时波形预览）
- **多算法支持**：频域维纳滤波 / 谱减法 / U-Net / Hybrid (U-Net+Wiener 混合) / 音频修复（时域插值 / 频域重建 / U-Net 迭代修复）
- **离线预混合加速**：训练前一次性生成 .npy 训练对，训练速度提升 10~20 倍
- **实时可视化**：时域波形对比、STFT 频谱图、梅尔谱，左右分屏同步展示
- **全面评估体系**：SNR / SegSNR / SI-SDR / STOI / PESQ / LSD / DNSMOS 共 7 项指标，得分板彩色编码
- **噪声类型诊断**：VAD 分离语音帧 → 频谱分析 → 自动识别白噪声/低频嗡嗡/背景人声/高频电子噪声
- **动态在线混合**：训练阶段实时合成带噪语音（随机 SNR∈[-5,+15]dB），无需预生成海量带噪数据集
- **音频回放**：带噪/降噪/纯净三段切换播放，支持进度拖拽
- **批量评估报告**：一键生成多算法的 CSV 对比报告，支持降噪和修复两种评估模式

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                  交互表现层 (ui/)                  │
│  PySide6 + pyqtgraph                              │
│  MainWindow → WaveformView / SpectrogramView      │
│            → MetricsPanel / DiagnosisPanel        │
│            → AudioPlayer / AudioRecorder          │
└─────────────────────┬────────────────────────────┘
                      │ 调用
┌─────────────────────▼────────────────────────────┐
│               算法引擎层 (models/)                 │
│  BaseDenoiser (ABC)                               │
│  ├── WienerFilter        (频域维纳滤波, 传统基准)   │
│  ├── SpectralSubtraction (谱减法, 传统对比)        │
│  ├── UNetDenoiser        (U-Net IRM 掩膜, 深度学习) │
│  ├── HybridDenoiser      (U-Net+Wiener 混合)       │
│  └── AudioInpainter      (时域/频域/U-Net 修复)     │
└──────┬──────────────────────────┬────────────────┘
       │ 训练靠                    │ 评估靠
┌──────▼──────────┐    ┌──────────▼────────────────┐
│  数据工程层      │    │     评估基准层              │
│  (data/)        │    │   (evaluation/)            │
│  动态在线混合    │    │   SNR / SegSNR / SI-SDR    │
│  数据增强        │    │   STOI / PESQ / LSD        │
│  噪声诊断        │    │   DNSMOS / 可视化          │
└─────────────────┘    └───────────────────────────┘
```

四层解耦架构，各层通过接口交互，可独立开发、测试和替换。

## 环境安装

### 前置要求

- Python 3.10+
- CUDA 11.8+ (可选，用于 GPU 训练 U-Net)

### 安装步骤

```bash
# 1. 创建 conda 虚拟环境 (推荐)
conda create -n audio-denoise python=3.10 -y
conda activate audio-denoise

# 2. 安装 PyTorch (根据 CUDA 版本选择)
# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
pip install -r requirements.txt
```

## 数据集准备

### 1. 训练数据（基础模型）

| 类型 | 数据集 | 大小 | 下载链接 |
|------|--------|------|----------|
| 纯净语音 | **LibriSpeech train-clean-100** | ~6GB | https://www.openslr.org/12 |
| 环境噪声 | **DEMAND** | ~500MB | https://zenodo.org/record/1227121 |

下载后请解压放入 `datasets/raw/`：

```
datasets/raw/
├── LibriSpeech/
│   └── train-clean-100/
│       ├── 19/  198/  19-198-0000.flac ...
│       └── 26/  ...
└── DEMAND/
    ├── ch01/  ch02/  ...  ch18/
```

### 2. 微调数据（跨域泛化）

| 类型 | 推荐来源 | 数量 | 获取方式 | 放入目录 |
|------|----------|------|----------|----------|
| 中文语音 | **ST-CMDS**(~8GB,任选其中部分音频) | 30~50 条, 5~15s/条 | [https://www.openslr.org/38/](https://www.openslr.org/38/) | `datasets/finetune/raw/zh/` |
| 英文对话 | **LibriSpeech test-clean** | ~300 条, 自动抽取 | 已有数据，脚本自动从 test-clean 抽 10 个 speaker | `en/` 留空即可 |
| 歌曲人声 | **MIR-1K** (~1GB) | 10~20 条片段 | [http://mirlab.org/dataset/public/MIR-1K.zip](http://mirlab.org/dataset/public/MIR-1K.zip) | `datasets/finetune/raw/sing/` |
| 多样化噪声 | **ESC-50** (200MB) | 2000 条 | `git clone https://github.com/karolpiczak/ESC-50` | `datasets/finetune/raw/noise/` |

#### 数据集介绍

**LibriSpeech (train-clean-100)** \
是源自 LibriVox 项目的大规模有声读物英文清晰语音语料库，在自动语音识别（ASR）及语音增强（SE）领域被广泛应用。该子集包含由 251 名朗读者（125 名男性，126 名女性）录制的约 100 小时高质量、低噪声英文朗读音频。所有音频文件均采用 16 kHz 采样率、16-bit 量化精度的无损 FLAC 格式存储，并提供了与之严格对齐的文本标注。因其较高的信噪比与发音标准度，该子集在智能音频处理算法中充当理想的干净语音基准源。

**DEMAND (Diverse Environments Multi-channel Acoustic Noise Database)** \
是广泛用于评估稳健语音增强及盲源分离算法的多通道环境噪声数据库。该数据集涵盖多种真实生活场景下的三维空间噪声，细分为室内（如办公室、餐厅）、室外（如广场、街区）以及交通工具（如地铁、汽车）等 6 大类共 18 种特定环境。录音采用 16 通道正方体麦克风阵列同步采集，提供 48 kHz 高采样率、24-bit 无损 WAV 格式的原始音轨，能够为深度学习降噪模型提供丰富且逼真的空间声学特征与非平稳噪声分布。

**ST-CMDS (Surfingtech Chinese Mandarin Corpus)** \
是由上海希尔贝壳（Surfingtech）开源的轻量级中文普通话语音数据集，常用于语音识别及面向中文场景的语音降噪开发。该语料库由 855 名发音标准的中文北方方言区发言人（涵盖不同年龄段与性别比例）在相对安静的室内环境下录制，共包含 102,600 条短语音频。每段音频时长约 3~5 秒，采用单通道、16 kHz 采样率及 16-bit 线性 PCM（WAV）格式存储，并配有对应汉字文本转写，是快速验证深度学习模型对中文语音泛化性能的标准基准之一。

**MIR-1K (Music Information Retrieval 1K)** \
是专为音高追踪、歌声提取及音乐信息检索任务设计的开放式多媒体数据集。该数据集由 19 名业余歌手（11 名女性，8 名男性）演唱的 1,000 段中文流行歌曲片段组成，总时长约 133 分钟。其核心声学特征在于采用双声道格式：左声道独立录制纯伴奏音乐，右声道独立录制干净的纯人声歌唱。音频采样率为 16 kHz，量化位深为 16-bit，为评估音频源分离与歌声降噪算法提供了天然且精确的成分标签。

**ESC-50 (Environmental Sound Classification)** \
是环境声分类与计算音频场景分析领域最核心的基准数据集之一。该数据集由 2,000 段经过严格筛选的短环境音组成，均匀划分为 50 个功能性类别（如动物声、自然环境音、人类非言语声音、家庭内部噪声以及城市环境噪声），每类包含 40 个样本。所有音频片段时长固定为 5 秒，采用单声道、44.1 kHz 采样率及 16-bit WAV 格式。该数据集的标准构造使得研究人员能够以统一的 5 折交叉验证协议科学评估深度神经网络对复杂多变噪声的分类与表征能力。

### 3. 运行预处理

```bash
# 训练数据预处理
python scripts/prepare_data.py --raw_dir datasets/raw --output_dir datasets/processed --val_ratio 0.1 --test_ratio 0.1 --chunk_duration 10.0

# 微调数据预处理 (en/ 留空时自动从 test-clean 抽取 300 条英文)
python scripts/prepare_finetune_data.py --raw_dir datasets/finetune/raw --output_dir datasets/finetune/processed --librispeech_test datasets/raw/LibriSpeech/test-clean
```

脚本自动：扫描原始文件 → 统一 16kHz 单声道 → DEMAND 切 10s 分块 → 按 speaker/类型组织 → 生成 train/val/test split。

### 4. 离线预混合（加速训练）

```bash
python scripts/premix_dataset.py --clean_dir datasets/processed/clean --noise_dir datasets/processed/noise --output_dir datasets/premixed --num_pairs 20000
```

生成 20000 对 `.npy` 训练对，训练 I/O 提升 10~20 倍。约占用 5GB 磁盘。

### 5. 输出结构

```
datasets/
├── raw/                      # 原始数据集 (用户放入)
├── processed/                # 训练用预处理后 (脚本生成)
│   ├── clean/                # 16kHz WAV, 按 speaker_id 分目录
│   └── noise/                # 16kHz WAV, 按噪声类型分目录 (18 种)
├── finetune/                 # 微调用
│   ├── raw/                  # 微调原始数据 (用户放入)
│   └── processed/            # 微调预处理后 (脚本生成)
├── premixed/                 # 离线预混合 .npy
├── test_generated/           # 评估时自动生成的测试数据
└── splits/                   # train/val/test 分割列表 JSON
```

## 快速开始

### 1. 训练 U-Net

动态混合模式：
```bash
python scripts/train.py --config config/unet.yaml --clean_dir datasets/processed/clean --noise_dir datasets/processed/noise
```

预混合模式（推荐）：
```bash
python scripts/train.py --config config/unet.yaml --use_premix --premix_dir datasets/premixed
```

#### U-Net 微调

原始 U-Net 在 LibriSpeech（英文朗读）上训练，对歌曲/中文对话等分布外数据效果差。微调用少量跨域数据适配模型。

**微调原理**：Encoder 前 4 层提取底层频谱纹理（通用特征，冻结）；Encoder 后 3 层 + Bottleneck + Decoder 提取语音/噪声判别（领域相关，可训练）。冻结 ~40% 参数，仅训练 ~60%，lr=5e-5（比从头训练小 20 倍），10~20 epoch 防过拟合。权重保存至 `checkpoints/unet_finetuned/`，不覆盖原始权重。

```bash
# 1. 预处理微调数据 (需先按上面步骤下载并放入 datasets/finetune/raw/)
python scripts/prepare_finetune_data.py

# 2. 微调
python scripts/finetune.py --clean_dir datasets/finetune/processed/clean --noise_dir datasets/finetune/processed/noise --pretrained checkpoints/unet/best_model.pt --output_dir checkpoints/unet_finetuned --epochs 15

# 3. 使用微调权重
python scripts/inference.py song.wav --algo unet --ckpt checkpoints/unet_finetuned/best_model.pt --output clean.wav
```

### 2. 批量评估

降噪评估（原始+微调 U-Net 对比）：
```bash
python scripts/evaluate.py --clean_source datasets/processed/clean --noise_source datasets/processed/noise --split_json datasets/splits/test_clean.json --algorithms wiener spectral_sub unet unet_ft hybrid --ckpt checkpoints/unet/best_model.pt --ft_ckpt checkpoints/unet_finetuned/best_model.pt --num_test 30 --output evaluation_report.csv
```

音频修复评估（自动损坏→修复→对比）：
```bash
python scripts/evaluate.py --mode inpainting --clean_source datasets/processed/clean --split_json datasets/splits/test_clean.json --methods spline spectral unet --ckpt checkpoints/unet/best_model.pt --num_test 20 --output evaluation_report_inpainting.csv
```

### 3. 生成实验图表

训练和微调都完成后，一键生成全部科研图表：

```bash
python scripts/plot_results.py --eval_csv evaluation_report.csv --train_csv logs/training_history.csv --ft_train_csv logs/finetune_history.csv --model_ckpt checkpoints/unet/best_model.pt --output_dir results/figures
```

| 图表文件 | 生成条件 | 内容 |
|----------|----------|------|
| `training_curves.png` | 训练完成 | 原始 U-Net Loss + LR 曲线 |
| `finetune_curves.png` | 微调完成 | 微调 Loss 曲线 (finetune.py 自动生成) |
| `training_comparison.png` | 提供 `--ft_train_csv` | 原始 vs 微调 Loss 并排对比 |
| `algorithm_comparison.png` | 提供 `--eval_csv` | 全部算法柱状图对比 |
| `quality_vs_speed.png` | 提供 `--eval_csv` | PESQ vs 推理时间散点图 |
| `irm_mask_example.png` | `--model_ckpt` + `--test_audio` | IRM 掩膜三行热力图 |
| `activation_map.png` | `--model_ckpt` + `--test_audio` | 激活图叠加 |
| `feature_tsne.png` | `--model_ckpt` + `--tsne` | t-SNE 特征降维散点图 |

### 4. 命令行推理

```bash
# 维纳滤波
python scripts/inference.py input.wav --algo wiener --output clean.wav

# 谱减法
python scripts/inference.py input.wav --algo spectral_sub --output clean.wav --plot comparison.png

# U-Net
python scripts/inference.py input.m4a --algo unet --ckpt checkpoints/unet/best_model.pt --output clean.wav

# Hybrid (不需微调，跨域鲁棒)
python scripts/inference.py song.wav --algo hybrid --ckpt checkpoints/unet/best_model.pt --output clean.wav

# 音频修复
python scripts/inference.py damaged.wav --algo inpaint --inpaint_method unet --ckpt checkpoints/unet/best_model.pt --output repaired.wav
```

所有格式 (.wav .flac .mp3 .m4a .aac) 均可作为输入。

### 5. 启动 GUI

```bash
# 传统算法 (无需 --ckpt)
python ui/main_window.py

# 含 U-Net 原始权重
python ui/main_window.py --ckpt checkpoints/unet/best_model.pt

# 同时加载原始 + 微调权重 (方法列表多出 "U-Net (Fine-tuned)"，Hybrid 自动使用微调权重)
python ui/main_window.py --ckpt checkpoints/unet/best_model.pt --ft_ckpt checkpoints/unet_finetuned/best_model.pt
```

状态栏会显示当前加载的模型路径。GUI 操作流程：

**降噪对比模式**：
1. 点击 **加载音频** → 选择文件 (支持 `.wav/.mp3/.flac/.m4a/.aac`)
2. 模式选 **降噪**，勾选 **对比全部方法**
3. 点击 **▶ 执行** → 串行跑 Wiener + SpectralSub + U-Net + Hybrid
4. 对比表格第一行为 **"Noisy (Original)"**（带噪原始基准），下拉可切换查看各方法波形/频谱/播放

**修复对比模式**：
1. 加载损坏音频，模式选 **修复**，勾选 **对比全部方法**
2. 点击 **▶ 执行** → 串行跑 Spline + Spectral + U-Net 修复
3. 对比表格直接展示三种修复策略 + Noisy (Original) 基准

**在线录音模式**：点击 **录制音频** → 开始录音 → 停止 → 选模式/方法 → **▶ 执行**

**单选模式**：取消勾选"对比全部方法"，下拉选择具体方法，只跑一种。

## 项目目录结构

```
audio-denoising/
├── config/                     # YAML 超参数集中管理
│   ├── default.yaml            # 全局默认配置 (采样率/STFT/增强/评估/日志)
│   ├── unet.yaml               # U-Net 训练配置 (模型/训练/损失)
│   └── wiener.yaml             # 传统算法参数 (维纳滤波/谱减法)
│
├── data/                       # 数据工程层
│   ├── __init__.py             # 模块导出 (load_audio, normalize_rms, ...)
│   ├── dataset.py              # DenoisingDataset — 动态在线混合
│   ├── augment.py              # SpecAugment + 音量/速度扰动 + 静音插入
│   ├── preprocess.py           # 音频加载/重采样/RMS 归一化
│   └── noise_diagnosis.py      # VAD + 频谱分析 → 噪声类型自动分类
│
├── models/                     # 算法引擎层
│   ├── __init__.py             # 模块导出 (BaseDenoiser, WienerFilter, ...)
│   ├── base.py                 # BaseDenoiser 抽象基类 (强制 forward + denoise_audio)
│   ├── wiener.py               # WienerFilter — 频域维纳滤波
│   ├── spectral_sub.py         # SpectralSubtraction — 谱减法
│   ├── unet.py                 # UNetDenoiser — 7层 Encoder-Decoder IRM 掩膜模型
│   ├── hybrid.py               # HybridDenoiser — U-Net+Wiener 混合降噪
│   └── audio_inpainter.py      # AudioInpainter — 音频修复 (spline/spectral/unet)
│
├── evaluation/                 # 评估基准层
│   ├── __init__.py             # 模块导出 (compute_all_metrics, plot_*, ...)
│   ├── metrics.py              # SNR/SegSNR/SI-SDR/STOI/PESQ/LSD/DNSMOS
│   └── visualizer.py           # 波形图/STFT 频谱图/梅尔谱/对比图绘制
│
├── ui/                         # 交互表现层
│   ├── __init__.py             # 模块导出 (MainWindow)
│   ├── main_window.py          # 主窗口 — 双模式工作流 (文件/录音) + 降噪调度
│   ├── audio_player.py         # AudioPlayer — 带噪/降噪/纯净三段播放控制
│   ├── audio_recorder.py       # AudioRecorder — 麦克风录音 + 实时波形预览
│   └── widgets/                # 自定义 UI 组件
│       ├── waveform_view.py    # pyqtgraph 双通道波形对比图
│       ├── spectrogram_view.py # 左右并排 STFT 频谱图 (inferno/viridis)
│       ├── metrics_panel.py    # 指标得分板表格 (绿/黄/红 自动着色)
│       └── diagnosis_panel.py  # 噪声诊断结论 + 频谱曲线 + 频段标注
│
├── scripts/                    # 入口脚本
│   ├── prepare_data.py         # 数据集自动化预处理 (LibriSpeech + DEMAND)
│   ├── premix_dataset.py       # 离线预混合 → .npy 训练对 (加速训练)
│   ├── train.py                # U-Net 训练 (--use_premix 启用预混合模式)
│   ├── finetune.py             # U-Net 微调 (冻结 Encoder 前 4 层)
│   ├── prepare_finetune_data.py # 微调数据预处理
│   ├── inference.py            # 单文件推理 (--algo wiener|spectral_sub|unet|hybrid|inpaint)
│   ├── evaluate.py             # 批量评估 → CSV 对比报告
│   └── plot_results.py         # 一键生成全部实验图表
│
├── tests/                      # 单元测试 (pytest)
│   ├── test_dataset.py         # 数据集 shape/SNR 范围验证
│   ├── test_metrics.py         # 指标极端场景验证 (相同信号/零信号)
│   └── test_models.py          # 模型推理 shape/值域验证
│
├── checkpoints/                # 模型权重存放
│   ├── unet/                   # 原始训练权重
│   └── unet_finetuned/         # 微调权重
├── logs/                       # 训练日志和 loss CSV (gitignore)
├── results/                    # 实验图表输出 (gitignore)
├── requirements.txt            # Python 完整依赖列表
└── README.md                   # 项目文档
```

## 各模块详细说明

### 数据工程层 (`data/`)

| 文件 | 功能 |
|------|------|
| `preprocess.py` | `load_audio()` 加载+重采样+转单声道；`normalize_rms()` RMS 电平归一化；`resample_if_needed()` 按需重采样 |
| `dataset.py` | `DenoisingDataset` — PyTorch Dataset，`__getitem__` 中随机截取 4s 片段 → 按 SNR∈[-5,+15]dB 在线混合 → 返回 (noisy, clean) 对；支持 speaker_id 分割防泄露 |
| `augment.py` | `spec_augment()` 频谱掩蔽 (频率宽度≤16, 时间宽度≤20)；`volume_perturb()` 音量扰动；`speed_perturb()` 速度扰动；`insert_mute_segment()` 随机静音段 |
| `noise_diagnosis.py` | `detect_speech_frames()` 基于能量+过零率的 VAD；`analyze_noise_spectrum()` 非语音帧频谱分析；`classify_noise_type()` 四级分类；`diagnose_noise()` 一键诊断入口 |

### 算法引擎层 (`models/`)

| 文件 | 功能 |
|------|------|
| `base.py` | `BaseDenoiser` 抽象基类，定义 `forward()` (训练) 和 `denoise_audio()` (推理) 接口，提供 `_compute_stft()` / `_compute_istft()` 工具方法 |
| `wiener.py` | `WienerFilter` — 前 N 帧估计噪声功率谱 → 逐帧维纳增益衰减 → 重叠相加还原。参数：帧长 32ms、噪声窗口 500ms |
| `spectral_sub.py` | `SpectralSubtraction` — 过减因子 α=2.0 + 频谱底板 β=0.01，前 N 帧估计噪声 → 逐帧谱减 → IFFT 重建 |
| `unet.py` | `UNetDenoiser` — 7 层 Encoder-Decoder (Conv2d+BN+ReLU) + Skip Connections → Sigmoid 输出 IRM 掩膜；训练损失 = MSE(掩膜) + L1(幅度谱)；推理 `denoise_audio()` (STFT→掩膜→iSTFT)；GUI/inference/evaluate 全链路集成 |
| `hybrid.py` | `HybridDenoiser` — U-Net 预测 IRM 掩膜 → 反推动态噪声能量谱 `(1-mask)²×|Y|²` → 驱动 Wiener 保守降噪。U-Net 高置信度区域信任原版效果，低置信度区域自动切换 Wiener 保守策略，对歌曲/中文等分布外数据有效，不重新训练 |
| `audio_inpainter.py` | `AudioInpainter` — 损坏检测 + 三种修复方法 (spline/spectral/unet)；`generate_damaged()` 自动生成静音/削波/频率缺失损坏音频用于评估 |

### 评估基准层 (`evaluation/`)

| 文件 | 功能 |
|------|------|
| `metrics.py` | `compute_all_metrics()` — 计算 8 项客观指标 (SNR/SegSNR/SI-SDR/STOI/PESQ/LSD/DNSMOS) |
| `visualizer.py` | 包含基础展示图 (波形/频谱/梅尔谱) 和科研图表 (训练曲线/算法对比柱状图/t-SNE/IRM 掩膜/激活图)，共计 11 个绘图函数 |

| 指标 | 类别 | 范围 | 说明 |
|------|------|------|------|
| SNR | 物理 | dB | 全局信号功率与噪声功率之比 |
| SegSNR | 物理 | dB | 30ms 帧长分段 SNR 均值，对局部失真更敏感 |
| SI-SDR | 物理 | dB | 尺度不变信号失真比，消除增益差异影响 |
| STOI | 可懂度 | 0~1 | 短时客观可懂度，越高越清晰 |
| PESQ_WB | 感知 | 1.0~4.5 | 宽带感知语音质量评估 (ITU-T P.862) |
| PESQ_NB | 感知 | -0.5~4.5 | 窄带感知语音质量评估 |
| LSD | 物理 | dB | 对数谱距离，值越低频域保真度越好 |
| DNSMOS | 无参考 | 1~5 | MIT 预训练模型预测 MOS 分，无需参考信号 |

### 交互表现层 (`ui/`)

| 文件 | 功能 |
|------|------|
| `main_window.py` | `MainWindow` — 双模式工作流。模式切换(降噪/修复) + 对比全部方法开关 + `BatchWorker` 批量执行。支持单选一种方法或一键对比全部。启动参数 `--ckpt` 指定 U-Net 权重路径 |
| `audio_player.py` | `AudioPlayer` — 带噪/降噪/纯净三段音源切换，播放/暂停/停止，进度条 seek，`sounddevice.OutputStream` 流式回放 |
| `audio_recorder.py` | `AudioRecorder` — `sounddevice.InputStream` 麦克风采集，`RingBuffer` 线程安全缓冲，QTimer 50ms 实时波形预览，最长 30s 自动停止 |
| `widgets/waveform_view.py` | `WaveformView` — pyqtgraph 双通道波形对比 (Noisy 红 / Denoised 绿 / Clean 蓝虚线) |
| `widgets/spectrogram_view.py` | `SpectrogramView` — 左右并排 STFT 频谱图 (inferno/viridis colormap)，带 dB 色标 |
| `widgets/metrics_panel.py` | `MetricsPanel` — QTableWidget 指标得分板，绿色(优秀) / 黄色(一般) / 红色(较差) 自动着色 |
| `widgets/diagnosis_panel.py` | `DiagnosisPanel` — 左侧文本诊断结论 + 频段占比，右侧噪声频谱曲线 + 频段标注线 |

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 单独测试
pytest tests/test_metrics.py -v
pytest tests/test_models.py -v
pytest tests/test_dataset.py -v
```

## 开发说明

### 文档规范 (强制)
- 每个 `.py` 文件头部需包含模块级 docstring，说明文件在系统中的角色
- 每个函数/方法需包含 docstring，说明功能、参数 (Args) 和返回值 (Returns)
- 公共 API 使用完整格式，私有/内部辅助函数至少一行摘要

### 代码风格
- 格式化：black + isort + ruff
- 配置管理：所有超参数写入 `config/*.yaml`，代码中禁止硬编码
- 日志系统：使用 `logging` 模块，控制台 + 文件双输出
- 数据安全：关键路径添加维度断言和 NaN/Inf 梯度检测

### 版本控制
- Git 托管至 GitHub
- 分支策略：`main` (稳定版) / `feature/unet` / `feature/gui` / `feature/traditional`
