"""
evaluation/visualizer.py — 音频可视化工具

提供时域波形、STFT 频谱图、梅尔谱的绘制函数。
接受 numpy 波形数组，返回 matplotlib Figure 对象，可直接嵌入 GUI 或保存为 PNG。
"""

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 非交互后端，可在无 GUI 环境运行
import matplotlib.pyplot as plt

# 统一配色方案: noisy=红色系, clean=蓝色系, denoised=绿色系
COLORS = {
    "noisy": "#E74C3C",
    "clean": "#2980B9",
    "denoised": "#27AE60",
}


def plot_waveform(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray | None = None,
    sr: int = 16000,
    title: str = "Waveform Comparison",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """绘制时域波形对比图。

    Args:
        noisy: 带噪信号.
        denoised: 降噪后信号.
        clean: 纯净参考信号 (可选).
        sr: 采样率.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(noisy)) / sr

    ax.plot(t, noisy, color=COLORS["noisy"], alpha=0.6, linewidth=0.5, label="Noisy")
    ax.plot(t, denoised, color=COLORS["denoised"], alpha=0.8, linewidth=0.5, label="Denoised")
    if clean is not None:
        min_len = min(len(t), len(clean))
        ax.plot(
            t[:min_len], clean[:min_len], color=COLORS["clean"],
            alpha=0.5, linewidth=0.5, label="Clean"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, t[-1])
    plt.tight_layout()
    return fig


def plot_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    title: str = "STFT Spectrogram",
    figsize: tuple[int, int] = (6, 4),
    color: str = "viridis",
) -> plt.Figure:
    """绘制 STFT 频谱图 (dB 尺度)。

    Args:
        waveform: 输入波形.
        sr: 采样率.
        n_fft: FFT 点数.
        hop_length: 帧移.
        title: 图表标题.
        figsize: 图表尺寸.
        color: matplotlib colormap 名称.

    Returns:
        matplotlib Figure 对象.
    """
    import librosa
    import librosa.display

    stft = librosa.stft(waveform.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mag_db, sr=sr, hop_length=hop_length, x_axis="time",
        y_axis="hz", ax=ax, cmap=color,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 128,
    title: str = "Mel Spectrogram",
    figsize: tuple[int, int] = (6, 4),
    color: str = "magma",
) -> plt.Figure:
    """绘制梅尔频谱图。

    Args:
        waveform: 输入波形.
        sr: 采样率.
        n_fft: FFT 点数.
        hop_length: 帧移.
        n_mels: 梅尔滤波器组数量.
        title: 图表标题.
        figsize: 图表尺寸.
        color: matplotlib colormap 名称.

    Returns:
        matplotlib Figure 对象.
    """
    import librosa
    import librosa.display

    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32), sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length, x_axis="time",
        y_axis="mel", ax=ax, cmap=color,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_comparison(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray | None = None,
    sr: int = 16000,
) -> plt.Figure:
    """绘制完整对比图 — 上方波形，下方频谱。

    Args:
        noisy: 带噪信号.
        denoised: 降噪后信号.
        clean: 纯净参考信号 (可选).
        sr: 采样率.

    Returns:
        组合 matplotlib Figure 对象.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 波形
    t = np.arange(len(noisy)) / sr
    ax_wf = axes[0, 0]
    ax_wf.plot(t, noisy, color=COLORS["noisy"], alpha=0.6, linewidth=0.4, label="Noisy")
    ax_wf.plot(t, denoised, color=COLORS["denoised"], alpha=0.8, linewidth=0.4, label="Denoised")
    ax_wf.set_title("Waveform")
    ax_wf.set_xlabel("Time (s)")
    ax_wf.legend(fontsize=7)
    ax_wf.set_xlim(0, t[-1])

    # 带噪频谱
    import librosa
    import librosa.display

    ax_spec_n = axes[1, 0]
    stft_n = librosa.stft(noisy.astype(np.float32), n_fft=512, hop_length=256)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(stft_n), ref=np.max),
        sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec_n,
        cmap="inferno",
    )
    ax_spec_n.set_title("Noisy Spectrogram")

    # 降噪频谱
    ax_spec_d = axes[1, 1]
    stft_d = librosa.stft(denoised.astype(np.float32), n_fft=512, hop_length=256)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(stft_d), ref=np.max),
        sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec_d,
        cmap="viridis",
    )
    ax_spec_d.set_title("Denoised Spectrogram")

    # 纯净频谱 (如有)
    if clean is not None:
        ax_spec_c = axes[0, 1]
        stft_c = librosa.stft(clean[:len(noisy)].astype(np.float32), n_fft=512, hop_length=256)
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(stft_c), ref=np.max),
            sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec_c,
            cmap="Blues",
        )
        ax_spec_c.set_title("Clean Spectrogram")
    else:
        axes[0, 1].axis("off")

    plt.tight_layout()
    return fig


# ============================================================
#  训练动态图
# ============================================================


def plot_training_curves(
    csv_path: str,
    title: str = "Training Curves",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """从训练 CSV 读取 epoch/loss/lr，绘制 Loss 下降曲线和 LR 衰减副轴。

    CSV 格式: epoch, train_loss, lr

    Args:
        csv_path: 训练历史 CSV 文件路径.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    import csv

    epochs, losses, lrs = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            lrs.append(float(row["lr"]))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(epochs, losses, color="#E74C3C", linewidth=1.5, marker="o", markersize=3, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#E74C3C")
    ax1.tick_params(axis="y", labelcolor="#E74C3C")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="#2980B9", linewidth=1.2, linestyle="--", label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="#2980B9")
    ax2.tick_params(axis="y", labelcolor="#2980B9")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(title)
    plt.tight_layout()
    return fig


# ============================================================
#  定量分析图
# ============================================================


def plot_algorithm_comparison(
    metrics_dict: dict[str, dict[str, float]],
    metric_names: list[str] | None = None,
    title: str = "Algorithm Comparison",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """多算法柱状图对比，每组指标一组柱，不同算法不同颜色。

    Args:
        metrics_dict: {"Wiener": {"SNR (dB)": 12.3, "PESQ_WB": 3.1}, "U-Net": {...}}.
        metric_names: 需要展示的指标名称列表，None 则自动从数据中提取.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    algorithms = list(metrics_dict.keys())
    if metric_names is None:
        metric_names = list(next(iter(metrics_dict.values())).keys())

    n_metrics = len(metric_names)
    n_algos = len(algorithms)
    x = np.arange(n_metrics)
    width = 0.8 / n_algos

    colors = ["#E74C3C", "#2980B9", "#27AE60", "#F39C12", "#8E44AD"]
    fig, ax = plt.subplots(figsize=figsize)

    for i, algo in enumerate(algorithms):
        values = [metrics_dict[algo].get(m, 0) for m in metric_names]
        offset = (i - n_algos / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=algo, color=colors[i % len(colors)])
        # 数值标注
        for bar, val in zip(bars, values):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scatter_quality_speed(
    results: list[dict],
    title: str = "Quality vs. Inference Speed",
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """质量-速度散点图: X=推理时间(ms), Y=PESQ, 每个算法一个点。

    Args:
        results: [{"algorithm": "Wiener", "pesq": 3.1, "time_ms": 12.5}, ...].
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#E74C3C", "#2980B9", "#27AE60", "#F39C12"]
    for i, r in enumerate(results):
        ax.scatter(r["time_ms"], r.get("pesq", r.get("PESQ_WB", 0)),
                   color=colors[i % len(colors)], s=120, label=r["algorithm"], zorder=3)
        ax.annotate(r["algorithm"], (r["time_ms"], r.get("pesq", r.get("PESQ_WB", 0))),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)

    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("PESQ")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix_noise_type(
    y_true: list[str],
    y_pred: list[str],
    title: str = "Noise Type Classification Confusion Matrix",
    figsize: tuple[int, int] = (8, 7),
) -> plt.Figure:
    """噪声类型诊断的混淆矩阵。

    Args:
        y_true: 真实噪声类型标签列表.
        y_pred: 预测噪声类型标签列表.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm_norm[i, j] > 0.5:
                color = "white"
            else:
                color = "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Normalized Accuracy")
    plt.tight_layout()
    return fig


# ============================================================
#  特征可视化图
# ============================================================


def plot_feature_tsne(
    features: np.ndarray,
    labels: list[str],
    title: str = "t-SNE Feature Visualization",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """对高维特征向量做 t-SNE 降维到 2D 后绘制散点图。

    Args:
        features: 特征矩阵, shape (n_samples, feature_dim).
        labels: 每个样本的类别标签.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embedded = tsne.fit_transform(features)

    unique_labels = sorted(set(labels))
    cmap = plt.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   color=cmap(i % 10), label=lbl, alpha=0.7, s=30)

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ============================================================
#  模型输出可视化图
# ============================================================


def plot_irm_mask(
    noisy_mag: np.ndarray,
    predicted_mask: np.ndarray,
    clean_mag: np.ndarray | None = None,
    sr: int = 16000,
    hop_length: int = 256,
    title: str = "IRM Mask Visualization",
    figsize: tuple[int, int] = (16, 6),
) -> plt.Figure:
    """三行子图展示 IRM 掩膜效果。

    Row 1: 带噪幅度谱 (hot colormap).
    Row 2: 预测 IRM 掩膜 (coolwarm, 蓝=抑制/红=保留).
    Row 3: 掩蔽后幅度谱 = mask * noisy_mag (viridis).

    Args:
        noisy_mag: 带噪幅度谱 (n_freqs, n_frames).
        predicted_mask: 预测 IRM (n_freqs, n_frames), 值域 [0,1].
        clean_mag: 纯净幅度谱 (n_freqs, n_frames), 可选.
        sr: 采样率.
        hop_length: STFT 帧移.
        title: 图表总标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    import librosa
    import librosa.display

    fig, axes = plt.subplots(3 if clean_mag is None else 4, 1, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    n_freqs = noisy_mag.shape[0]

    row = 0
    librosa.display.specshow(
        librosa.amplitude_to_db(noisy_mag, ref=np.max),
        sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz",
        ax=axes[row], cmap="hot",
    )
    axes[row].set_title("Noisy Magnitude Spectrum")
    axes[row].set_xlabel("")

    row += 1
    im = axes[row].imshow(predicted_mask, aspect="auto", origin="lower",
                          cmap="coolwarm", vmin=0, vmax=1,
                          extent=[0, predicted_mask.shape[1] * hop_length / sr,
                                  0, sr / 2])
    axes[row].set_title("Predicted IRM Mask (blue=suppress / red=preserve)")
    axes[row].set_ylabel("Frequency (Hz)")
    axes[row].set_xlabel("")
    fig.colorbar(im, ax=axes[row], label="Mask Value")

    row += 1
    restored_mag = noisy_mag * predicted_mask
    librosa.display.specshow(
        librosa.amplitude_to_db(restored_mag, ref=np.max),
        sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz",
        ax=axes[row], cmap="viridis",
    )
    axes[row].set_title("Restored Magnitude (mask * noisy)")

    if clean_mag is not None:
        row += 1
        librosa.display.specshow(
            librosa.amplitude_to_db(clean_mag, ref=np.max),
            sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz",
            ax=axes[row], cmap="viridis",
        )
        axes[row].set_title("Clean Magnitude Spectrum (reference)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_activation_map(
    activation: np.ndarray,
    input_mag: np.ndarray,
    sr: int = 16000,
    hop_length: int = 256,
    title: str = "Activation Map Overlay",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """将某层激活图叠加到输入幅度谱上，展示模型关注的时频区域。

    对 feature map 沿通道维取均值 → resize 到输入尺寸 → 叠加显示。

    Args:
        activation: 某层输出特征, shape (C, H, W) 或 (1, C, H, W).
        input_mag: 输入幅度谱, shape (n_freqs, n_frames).
        sr: 采样率.
        hop_length: 帧移.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    import librosa
    import librosa.display

    # 统一形状
    if activation.ndim == 4:
        activation = activation.squeeze(0)
    # (C, H, W) → 通道平均 → (H, W)
    if activation.ndim == 3:
        act_map = np.mean(activation, axis=0)
    else:
        act_map = activation

    # resize 到输入尺寸
    from scipy.ndimage import zoom
    zoom_h = input_mag.shape[0] / act_map.shape[0]
    zoom_w = input_mag.shape[1] / act_map.shape[1]
    act_resized = zoom(act_map, (zoom_h, zoom_w), order=1)
    act_norm = (act_resized - act_resized.min()) / (act_resized.max() - act_resized.min() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax0 = axes[0]
    librosa.display.specshow(
        librosa.amplitude_to_db(input_mag, ref=np.max),
        sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz",
        ax=ax0, cmap="viridis",
    )
    ax0.set_title("Input Magnitude Spectrum")

    ax1 = axes[1]
    db = librosa.amplitude_to_db(input_mag, ref=np.max)
    ax1.imshow(db, aspect="auto", origin="lower", cmap="gray",
               extent=[0, input_mag.shape[1] * hop_length / sr, 0, sr / 2])
    ax1.imshow(act_norm, aspect="auto", origin="lower", cmap="hot", alpha=0.5,
               extent=[0, input_mag.shape[1] * hop_length / sr, 0, sr / 2])
    ax1.set_title("Activation Overlay (hot=high activation)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")

    fig.suptitle(title)
    plt.tight_layout()
    return fig
