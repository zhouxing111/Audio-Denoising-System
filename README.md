# 智能音频降噪系统 (Intelligent Audio Denoising System)

基于 Python 的工业级智能音频降噪系统，集成传统信号处理算法与深度学习端到端模型，提供完整的训练流水线和交互式图形界面。

## 核心功能

- **双模式音源输入**：支持 WAV/FLAC/MP3/M4A 文件加载和麦克风在线录音（最长 30s，实时波形预览）
- **多算法支持**：频域维纳滤波 / 谱减法 / U-Net 深度学习模型（训练完可全链路调用）/ 音频修复（时域插值 / 频域重建 / U-Net 迭代修复）
- **离线预混合加速**：训练前一次性生成 .npy 训练对，训练速度提升 10~20 倍
- **实时可视化**：时域波形对比、STFT 频谱图、梅尔谱，左右分屏同步展示
- **全面评估体系**：SNR / SegSNR / SI-SDR / STOI / PESQ / LSD / DNSMOS 共 7 项指标，得分板彩色编码
- **噪声类型诊断**：VAD 分离语音帧 → 频谱分析 → 自动识别白噪声/低频嗡嗡/背景人声/高频电子噪声
- **动态在线混合**：训练阶段实时合成带噪语音（随机 SNR∈[-5,+15]dB），无需预生成海量带噪数据集
- **音频回放**：带噪/降噪/纯净三段切换播放，支持进度拖拽
- **批量评估报告**：一键生成多算法的 CSV 对比报告

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
│  └── AudioInpainter      (插值修复, 待实现)        │
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

### 1. 下载原始数据集

| 类型 | 数据集 | 大小 | 获取方式 |
|------|--------|------|----------|
| 纯净语音 | LibriSpeech train-clean-100 | ~6GB | https://www.openslr.org/12 |
| 环境噪声 | DEMAND | ~500MB | https://zenodo.org/record/1227121 |

### 2. 放入原始数据

将下载的数据集原封不动放入 `datasets/raw/` 目录：

```
datasets/raw/
├── LibriSpeech/
│   └── train-clean-100/      # 解压后整个文件夹放入
│       ├── 19/
│       ├── 26/
│       └── ...
└── DEMAND/
    ├── ch01/                  # 解压后 18 个通道文件夹放入
    ├── ch02/
    └── ...
```

### 3. 运行自动化预处理

```bash
python scripts/prepare_data.py --raw_dir datasets/raw --output_dir datasets/processed --val_ratio 0.1 --test_ratio 0.1 --chunk_duration 10.0
```

脚本自动完成：
- 扫描 LibriSpeech 全部 `.flac` 文件 → 转换为 16kHz 单声道 WAV
- 扫描 DEMAND 全部噪声 `.wav` → 重采样至 16kHz，**自动切分为 10 秒片段**（避免训练时加载 5 分钟整文件）
- 按 speaker_id / 噪声类型组织输出目录
- 生成说话人级 train/val/test 分割列表（同一个说话人不跨集合，防泄露）

### 4. 输出结构

```
datasets/
├── raw/                      # 原始数据集 (用户放入)
├── processed/                # 预处理后 (脚本自动生成)
│   ├── clean/                # 16kHz WAV, 按 speaker_id 分目录
│   │   ├── speaker_19/
│   │   └── ...
│   └── noise/                # 16kHz WAV, 按噪声类型分目录
│       ├── kitchen/
│       ├── traffic/
│       └── ...               # 共 18 种噪声场景
├── premixed/                  # 离线预混合 .npy 文件 (premix_dataset.py 生成)
└── splits/                   # 分割列表 JSON
    ├── train_clean.json
    ├── val_clean.json
    └── test_clean.json
```

脚本执行完毕后，终端会输出下一步命令，直接复制运行即可。

### 4. 离线预混合（加速训练，推荐）

```bash
python scripts/premix_dataset.py --clean_dir datasets/processed/clean --noise_dir datasets/processed/noise --output_dir datasets/premixed --num_pairs 20000
```

生成 20000 对预混合 .npy 文件，训练时加载速度提升 10~20 倍。预混合数据约占用 5GB 磁盘空间。

## 快速开始

### 1. 命令行推理 (单文件)

```bash
# 维纳滤波
python scripts/inference.py input.wav --algo wiener --output clean.wav

# 谱减法 + 对比图
python scripts/inference.py input.wav --algo spectral_sub --output clean.wav --plot comparison.png

# U-Net 深度学习
python scripts/inference.py input.m4a --algo unet --ckpt checkpoints/unet/best_model.pt --output clean.wav

# 音频修复 (三次样条插值)
python scripts/inference.py damaged.wav --algo inpaint --inpaint_method spline --output repaired.wav

# 音频修复 (U-Net 迭代修复)
python scripts/inference.py damaged.wav --algo inpaint --inpaint_method unet --ckpt checkpoints/unet/best_model.pt --output repaired.wav
```

所有格式 (.wav .flac .mp3 .m4a .aac) 均可作为输入。

### 2. 批量评估

自动生成测试对并评估（无需预先准备测试文件）：

```bash
python scripts/evaluate.py --clean_source datasets/processed/clean --noise_source datasets/processed/noise --algorithms wiener spectral_sub unet --ckpt checkpoints/unet/best_model.pt --num_test 30 --output evaluation_report.csv
```

如果已有测试文件，也可直接指定目录：

```bash
python scripts/evaluate.py --noisy_dir datasets/test_noisy --clean_dir datasets/test_clean --algorithms wiener spectral_sub unet --ckpt checkpoints/unet/best_model.pt --output evaluation_report.csv
```

音频修复批量评估（自动生成损坏 → 修复 → 对比）：
```bash
python scripts/evaluate.py --mode inpainting --clean_source datasets/processed/clean --methods spline spectral unet --ckpt checkpoints/unet/best_model.pt --num_test 20 --output evaluation_report_inpainting.csv
```

### 3. 训练 U-Net 模型

动态混合模式（在线合成带噪数据）：
```bash
python scripts/train.py --config config/unet.yaml --clean_dir datasets/processed/clean --noise_dir datasets/processed/noise
```

预混合模式（先运行 premix_dataset.py，再训练，速度提升 10~20 倍）：
```bash
python scripts/train.py --config config/unet.yaml --use_premix --premix_dir datasets/premixed
```

训练完成后，checkpoint 保存至 `checkpoints/unet/best_model.pt`。

### 4. 生成实验图表

训练和评估完成后，一键生成全部科研报告所需图表：

```bash
python scripts/plot_results.py --eval_csv evaluation_report.csv --train_csv logs/training_history.csv --model_ckpt checkpoints/unet/best_model.pt --output_dir results/figures
```

自动生成的图表：
```
results/figures/
├── training_curves.png         # Loss + LR 训练曲线
├── algorithm_comparison.png    # 多算法柱状图对比
├── quality_vs_speed.png        # 质量-推理速度散点图
├── irm_mask_example.png        # IRM 掩膜三行热力图
├── feature_tsne.png            # t-SNE 特征降维散点图
└── activation_map.png          # 激活图叠加
```

### 5. 启动 GUI

```bash
# 传统算法
python ui/main_window.py

# 含 U-Net 深度学习模型
python ui/main_window.py --ckpt checkpoints/unet/best_model.pt
```

GUI 操作流程：

**模式 A — 文件加载**：
1. 点击 **加载音频** → 选择本地 `.wav`/`.mp3`/`.flac`/`.m4a`/`.aac` 文件
2. 下拉选择算法 (Wiener Filter / Spectral Subtraction / U-Net / Audio Inpainting)
3. 点击 **一键降噪/修复** → 后台线程异步执行
4. 查看结果：波形对比 / 频谱图 / 评估指标 / 噪声诊断 / 音频回放

**模式 B — 在线录音**：
1. 点击 **录制音频** → 展开录音面板
2. 点击 **开始录音** → 实时波形预览 (最长 30s 自动停止)
3. 点击 **停止录音** → 录制完成，波形存入系统
4. 选择算法 → 点击 **一键降噪** → 全部展示面板同步更新
5. 使用音频播放器切换试听带噪/降噪结果

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
│   └── unet.py                 # UNetDenoiser — 7层 Encoder-Decoder IRM 掩膜模型
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
│   ├── inference.py            # 单文件推理 (--algo wiener|spectral_sub|unet)
│   ├── evaluate.py             # 批量评估 → CSV 对比报告
│   └── plot_results.py         # 一键生成全部实验图表
│
├── tests/                      # 单元测试 (pytest)
│   ├── test_dataset.py         # 数据集 shape/SNR 范围验证
│   ├── test_metrics.py         # 指标极端场景验证 (相同信号/零信号)
│   └── test_models.py          # 模型推理 shape/值域验证
│
├── checkpoints/                # 模型权重存放 (gitignore)
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
| `main_window.py` | `MainWindow` — 整合全部组件的主窗口。控制面板提供加载/录制/算法选择/降噪/导出。`DenoiseWorker` 支持 Wiener/SpectralSub/U-Net/AudioInpainting 四种算法。启动参数 `--ckpt` 指定 U-Net 权重路径 |
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
