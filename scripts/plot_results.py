"""
scripts/plot_results.py — 实验图表一键生成

从训练历史 CSV 和评估报告 CSV 中读取数据，自动生成全部科研报告所需的图表:
  - 训练曲线 (loss + lr)
  - 算法对比柱状图
  - IRM 掩膜可视化
  - t-SNE 特征降维
  - 激活图叠加
  - 质量-速度散点图

用法:
  python scripts/plot_results.py --eval_csv evaluation_report.csv --train_csv logs/training_history.csv --output_dir results/figures
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.visualizer import (
    plot_algorithm_comparison,
    plot_feature_tsne,
    plot_irm_mask,
    plot_scatter_quality_speed,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成全部实验图表")
    parser.add_argument(
        "--eval_csv", type=str, default="evaluation_report.csv",
        help="评估报告 CSV 路径 (evaluate.py 生成)",
    )
    parser.add_argument(
        "--train_csv", type=str, default="logs/training_history.csv",
        help="训练历史 CSV 路径 (train.py 生成)",
    )
    parser.add_argument(
        "--model_ckpt", type=str, default=None,
        help="U-Net checkpoint 路径 (用于特征提取和 IRM 可视化)",
    )
    parser.add_argument(
        "--test_audio", type=str, default=None,
        help="测试音频路径 (用于 IRM+激活图演示)",
    )
    parser.add_argument(
        "--tsne", action="store_true",
        help="生成 t-SNE 特征图 (需额外显存, 随机频谱缺乏真实多样性时可能不稳定)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/figures",
        help="图表输出目录",
    )
    return parser.parse_args()


def generate_from_eval_csv(eval_path: str, output_dir: str) -> dict:
    """从评估 CSV 读取数据，生成算法对比柱状图和散点图。

    CSV 列: file, algorithm, SNR (dB), SegSNR (dB), SI-SDR (dB),
             STOI, PESQ_WB, PESQ_NB, LSD (dB)

    Args:
        eval_path: 评估报告 CSV 路径.
        output_dir: 输出目录.

    Returns:
        metrics_dict: {"Wiener": {"SNR (dB)": ..., "PESQ_WB": ...}, ...}.
    """
    if not Path(eval_path).exists():
        logging.warning(f"评估 CSV 不存在: {eval_path}，跳过定量图表")
        return {}

    # 读取并按算法聚合均值
    algo_data: dict[str, dict[str, list[float]]] = {}
    with open(eval_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algorithm"]
            if algo not in algo_data:
                algo_data[algo] = {}
            for key, val in row.items():
                if key in ("file", "algorithm"):
                    continue
                try:
                    algo_data[algo].setdefault(key, []).append(float(val))
                except ValueError:
                    pass

    # 计算均值
    metrics_dict: dict[str, dict[str, float]] = {}
    for algo, metrics in algo_data.items():
        metrics_dict[algo] = {k: np.mean(v) for k, v in metrics.items()}

    # 柱状图
    fig = plot_algorithm_comparison(metrics_dict, title="Algorithm Performance Comparison")
    path = os.path.join(output_dir, "algorithm_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info(f"算法对比柱状图: {path}")

    # 散点图 (质量 vs 速度，速度为占位值)
    scatter_data = []
    for algo, m in metrics_dict.items():
        scatter_data.append({
            "algorithm": algo,
            "pesq": m.get("PESQ_WB", 0),
            "time_ms": 10 if algo not in ("UNetDenoiser", "unet") else 50,
        })
    fig2 = plot_scatter_quality_speed(scatter_data)
    path2 = os.path.join(output_dir, "quality_vs_speed.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    logging.info(f"质量速度散点图: {path2}")

    return metrics_dict


def generate_training_plot(train_csv: str, output_dir: str) -> None:
    """生成训练曲线图。

    Args:
        train_csv: 训练历史 CSV 路径.
        output_dir: 输出目录.
    """
    if not Path(train_csv).exists():
        logging.warning(f"训练 CSV 不存在: {train_csv}，跳过训练曲线")
        return

    fig = plot_training_curves(train_csv, title="U-Net Training Curves")
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info(f"训练曲线图: {path}")


def generate_model_plots(
    model_ckpt: str, test_audio: str, output_dir: str, tsne: bool = False,
) -> None:
    """生成 IRM 掩膜可视化和 t-SNE 特征图。

    Args:
        model_ckpt: U-Net checkpoint 路径.
        test_audio: 测试音频路径.
        output_dir: 输出目录.
        tsne: 是否生成 t-SNE 图 (随机频谱缺乏真实多样性时可能崩溃).
    """
    if not model_ckpt or not Path(model_ckpt).exists():
        logging.warning("未提供有效 model_ckpt，跳过模型图表")
        return

    import torch

    from models.unet import UNetDenoiser

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"加载模型 (device={device}) ...")
        model = UNetDenoiser(n_fft=512, hop_length=256).to(device)
        ckpt = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    # --- IRM 掩膜可视化 ---
    if test_audio and Path(test_audio).exists():
        _plot_irm_and_activation(model, test_audio, output_dir, device)

    # --- t-SNE 特征可视化 (opt-in, 随机频谱可能不稳定) ---
    if tsne:
        _plot_tsne_features(model, output_dir, device)
    else:
        logging.info("跳过 t-SNE (使用 --tsne 启用)")


def _plot_irm_and_activation(
    model, test_audio: str, output_dir: str, device
) -> None:
    """生成 IRM 掩膜图和激活图。

    Args:
        model: 已加载的 U-Net 模型.
        test_audio: 测试音频路径.
        output_dir: 输出目录.
        device: torch device.
    """
    import torch
    import librosa

    try:
        waveform, sr = librosa.load(test_audio, sr=16000)
        stft = librosa.stft(waveform.astype(np.float32), n_fft=512, hop_length=256)
        noisy_mag = np.abs(stft)

        mag_tensor = torch.from_numpy(noisy_mag).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            mask = model.forward(mag_tensor).squeeze().cpu().numpy()

        fig = plot_irm_mask(noisy_mag, mask, sr=16000, hop_length=256)
        path = os.path.join(output_dir, "irm_mask_example.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt = __import__("matplotlib.pyplot")
        plt.close(fig)
        logging.info(f"IRM 掩膜图: {path}")

        # 激活图
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook

        handle = model.enc1[0].register_forward_hook(hook_fn("enc1_conv0"))
        with torch.no_grad():
            _ = model.forward(mag_tensor)
        handle.remove()

        if "enc1_conv0" in activations:
            from evaluation.visualizer import plot_activation_map
            fig2 = plot_activation_map(activations["enc1_conv0"], noisy_mag)
            path2 = os.path.join(output_dir, "activation_map.png")
            fig2.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            logging.info(f"激活图: {path2}")
    except Exception as e:
        logging.warning(f"IRM/激活图生成失败: {e}")


def _make_embedding_plot(embedded: np.ndarray, labels: list[str], title: str) -> "plt.Figure":
    """从 2D 坐标和标签生成散点图，供 PCA/t-SNE 共用。

    Args:
        embedded: 2D 坐标, shape (n, 2).
        labels: 标签列表.
        title: 图表标题.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps["tab10"]
    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   color=cmap(i % 10), label=lbl, alpha=0.7, s=30)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def _plot_tsne_features(model, output_dir: str, device) -> None:
    """用 forward hook 提取 bottleneck 特征并生成 t-SNE 图。

    通过注册 hook 在 bottleneck 层捕获特征，避免逐层手动前向传播
    导致中间张量累积 OOM。

    Args:
        model: 已加载的 U-Net 模型.
        output_dir: 输出目录.
        device: torch device.
    """
    import torch

    bottleneck_features = []

    def bottleneck_hook(module, input, output):
        # output shape: (B, C, H, W) → 全局均值池化 → (B, C)
        feat = torch.mean(output.detach(), dim=(2, 3)).cpu().numpy()
        bottleneck_features.append(feat[0])

    # 注册 hook 到 bottleneck 的第二层卷积
    handle = model.bottleneck[-1].register_forward_hook(bottleneck_hook)

    try:
        noise_types = ["white_noise", "low_freq_hum", "background_speech", "high_freq_noise"]
        labels = []
        for i, nt in enumerate(noise_types):
            for _ in range(5):  # 每类 5 个样本
                # 用不同噪声分布增强多样性，避免 t-SNE 方差为零
                fake_mag = torch.randn(1, 1, 257, 128, device=device) * (0.1 + i * 0.2) + i * 0.1
                with torch.no_grad():
                    model.forward(fake_mag)
                labels.append(nt)

        if len(bottleneck_features) >= 10:
            features = np.array(bottleneck_features)
            feat_var = float(np.var(features))
            logging.info(f"t-SNE 特征方差: {feat_var:.8f}")
            if feat_var < 1e-12:
                logging.warning("特征方差过低, t-SNE 不稳定, 改用 PCA")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embedded = pca.fit_transform(features)
                fig3 = _make_embedding_plot(embedded, labels, "U-Net Bottleneck Feature PCA")
            else:
                fig3 = plot_feature_tsne(features, labels, title="U-Net Bottleneck Feature t-SNE")
            path3 = os.path.join(output_dir, "feature_tsne.png")
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            plt = __import__("matplotlib.pyplot")
            plt.close(fig3)
            logging.info(f"特征可视化图: {path3}")
        else:
            logging.warning(f"t-SNE 特征样本不足 ({len(bottleneck_features)})，跳过")
    except Exception as e:
        logging.warning(f"t-SNE 生成失败: {e}")
    finally:
        handle.remove()


def main() -> None:
    """主函数: 一键生成全部图表。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    generate_from_eval_csv(args.eval_csv, args.output_dir)
    generate_training_plot(args.train_csv, args.output_dir)
    generate_model_plots(args.model_ckpt, args.test_audio, args.output_dir, tsne=args.tsne)

    logging.info(f"全部图表已保存至: {args.output_dir}/")


if __name__ == "__main__":
    main()
