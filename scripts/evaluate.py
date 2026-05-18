"""
scripts/evaluate.py — 批量评估脚本

支持两种模式:
  模式 A (已有文件): 对 --noisy_dir 和 --clean_dir 中的配对文件评估
  模式 B (自动生成): 从 --clean_source 和 --noise_source 动态生成测试对后评估

用法:
  python scripts/evaluate.py --clean_source datasets/processed/clean --noise_source datasets/processed/noise --algorithms wiener spectral_sub unet --ckpt checkpoints/unet/best_model.pt --num_test 20 --output evaluation_report.csv
"""

import argparse
import csv
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio, normalize_rms, rms_energy
from evaluation.metrics import compute_all_metrics


def _get_device():
    """返回最佳可用设备: CUDA > MPS > CPU。"""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="批量评估降噪/修复算法")
    parser.add_argument(
        "--mode", type=str, default="denoising",
        choices=["denoising", "inpainting"],
        help="评估模式: denoising (降噪) 或 inpainting (修复)",
    )
    parser.add_argument(
        "--noisy_dir", type=str, default=None,
        help="带噪音频目录 (模式 A)",
    )
    parser.add_argument(
        "--clean_dir", type=str, default=None,
        help="纯净参考目录 (模式 A, 文件名需对应)",
    )
    parser.add_argument(
        "--clean_source", type=str, default=None,
        help="纯净语音源目录 (模式 B, 自动生成测试对)",
    )
    parser.add_argument(
        "--noise_source", type=str, default=None,
        help="噪声源目录 (模式 B, 自动生成测试对)",
    )
    parser.add_argument(
        "--num_test", type=int, default=30,
        help="生成测试对数量 (模式 B)",
    )
    parser.add_argument(
        "--test_snr", type=str, default="-5,0,5,10",
        help="测试 SNR 列表逗号分隔 (模式 B)",
    )
    parser.add_argument(
        "--test_output_dir", type=str, default="datasets/test_generated",
        help="测试文件输出目录 (模式 B)",
    )
    parser.add_argument(
        "--output", type=str, default="evaluations/evaluation_report.csv", help="输出 CSV 路径"
    )
    parser.add_argument(
        "--algorithms", type=str, nargs="+",
        default=["wiener", "spectral_sub"],
        help="评估的算法列表 (wiener spectral_sub unet)",
    )
    parser.add_argument(
        "--ckpt", type=str, default="checkpoints/unet/best_model.pt",
        help="U-Net checkpoint 路径 (unet/inpainting 时需要)",
    )
    parser.add_argument(
        "--ft_ckpt", type=str, default="checkpoints/unet_finetuned/best_model.pt",
        help="微调 U-Net checkpoint 路径 (仅 unet_ft 算法需要)",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["spline", "spectral", "unet"],
        help="修复方法列表 (仅 --mode inpainting)",
    )
    parser.add_argument(
        "--split_json", type=str, default=None,
        help="测试集 split JSON 路径 (如 datasets/splits/test_clean.json)，确保只评估未见过的说话人",
    )
    return parser.parse_args()


def collect_files(root: Path, split_json: str | None = None) -> list[Path]:
    """递归扫描目录收集音频文件，可选按 test split 过滤。

    Args:
        root: 扫描根目录.
        split_json: test split JSON 路径 (如 test_clean.json),
                    提供时只返回 split 中列出的文件，保证不包含训练集说话人.

    Returns:
        音频文件路径列表.
    """
    if split_json and Path(split_json).exists():
        import json
        with open(split_json, "r") as f:
            allowed = set(json.load(f))
        # allowed 格式: "speaker_19/19-198-0000.wav"
        files = []
        for rel in allowed:
            p = root / rel
            if p.exists():
                files.append(p)
        logging.getLogger(__name__).info(
            f"使用 test split: {len(files)}/{len(allowed)} 文件来自 {split_json}"
        )
        return sorted(files)

    exts = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def generate_test_pairs(
    clean_source: Path,
    noise_source: Path,
    output_dir: Path,
    num_pairs: int,
    snr_list: list[float],
    duration: float = 4.0,
    target_sr: int = 16000,
    split_json: str | None = None,
) -> None:
    """从纯净语音和噪声源动态生成测试对。

    每个 SNR 等量分配样本，文件命名含 SNR 信息便于追溯。

    Args:
        clean_source: 纯净语音源目录.
        noise_source: 噪声源目录.
        output_dir: 测试文件输出目录.
        num_pairs: 生成测试对总数.
        snr_list: SNR 值列表 (dB).
        duration: 每段时长 (秒).
        target_sr: 目标采样率.
    """
    noisy_dir = output_dir / "noisy"
    clean_dir = output_dir / "clean"

    # 清空旧数据，避免残留文件干扰
    if noisy_dir.exists():
        shutil.rmtree(noisy_dir)
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean_files = collect_files(clean_source, split_json=split_json)
    noise_files = collect_files(noise_source)
    if not clean_files:
        raise FileNotFoundError(f"未在 {clean_source} 找到音频文件 (split_json={split_json})")
    if not noise_files:
        raise FileNotFoundError(f"未在 {noise_source} 找到音频文件")

    num_samples = int(duration * target_sr)
    samples_per_snr = max(1, num_pairs // len(snr_list))

    pair_idx = 0
    for snr_db in snr_list:
        for _ in range(samples_per_snr):
            # 随机选纯净语音片段
            cf = random.choice(clean_files)
            clean_full, _ = load_audio(str(cf), target_sr=target_sr)
            if len(clean_full) < num_samples:
                repeats = num_samples // len(clean_full) + 1
                clean_full = np.tile(clean_full, repeats)
            start_c = random.randint(0, len(clean_full) - num_samples)
            clean_seg = clean_full[start_c : start_c + num_samples].astype(np.float32)

            # 随机选噪声片段
            nf = random.choice(noise_files)
            noise_full, _ = load_audio(str(nf), target_sr=target_sr)
            if len(noise_full) < num_samples:
                repeats = num_samples // len(noise_full) + 1
                noise_full = np.tile(noise_full, repeats)
            start_n = random.randint(0, len(noise_full) - num_samples)
            noise_seg = noise_full[start_n : start_n + num_samples].astype(np.float32)

            # 混合
            clean_rms = rms_energy(clean_seg)
            noise_rms = rms_energy(noise_seg)
            target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
            noise_seg = noise_seg * (target_noise_rms / (noise_rms + 1e-12))
            noisy_seg = clean_seg + noise_seg

            # 保存
            name = f"test_{pair_idx:04d}_snr{snr_db:+.0f}dB.wav"
            sf.write(str(clean_dir / name), clean_seg, target_sr, subtype="PCM_16")
            sf.write(str(noisy_dir / name), noisy_seg, target_sr, subtype="PCM_16")
            pair_idx += 1

    logging.info(f"已生成 {pair_idx} 对测试样本至 {output_dir}/")


def run_inpainting_eval(args) -> None:
    """音频修复评估模式。

    从纯净语音生成损坏音频 → 各方法修复 → 与原始对比评估。
    """
    logger = logging.getLogger(__name__)
    if not args.clean_source:
        logger.error("--clean_source 必须指定 (用于生成损坏音频的纯净语音源)")
        sys.exit(1)

    from models.audio_inpainter import AudioInpainter

    clean_files = collect_files(Path(args.clean_source), split_json=args.split_json)
    if not clean_files:
        logger.error(f"未在 {args.clean_source} 找到音频文件 (split_json={args.split_json})")
        return

    out_dir = Path(args.test_output_dir) / "inpainting"
    damaged_dir = out_dir / "damaged"
    repaired_dir = out_dir / "repaired"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    damaged_dir.mkdir(parents=True)
    repaired_dir.mkdir(parents=True)

    inpainter = AudioInpainter()
    samples = min(args.num_test, len(clean_files))

    # 生成损坏音频 + 修复
    fieldnames = ["file", "method", "SNR (dB)", "SegSNR (dB)", "SI-SDR (dB)", "STOI", "PESQ_WB", "LSD (dB)"]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(samples):
            cf = clean_files[idx]
            clean, sr = load_audio(str(cf))

            # 截取 4s
            num_samples = int(4.0 * sr)
            if len(clean) < num_samples:
                repeats = num_samples // len(clean) + 1
                clean = np.tile(clean, repeats)
            start = random.randint(0, len(clean) - num_samples)
            clean_seg = clean[start : start + num_samples].astype(np.float32)

            # 生成损坏
            damaged, gt_regions = AudioInpainter.generate_damaged(
                clean_seg, sr, num_silence=3, seed=idx,
            )
            damaged_name = f"damaged_{idx:04d}.wav"
            sf.write(str(damaged_dir / damaged_name), damaged, sr, subtype="PCM_16")

            # 各方法修复
            for method in args.methods:
                logger.info(f"修复: {damaged_name} / {method}")
                ckpt = args.ckpt if method == "unet" else None
                repaired = inpainter.inpaint(damaged, sr, method=method, model_ckpt=ckpt)
                metrics = compute_all_metrics(clean_seg[:len(repaired)], repaired[:len(clean_seg)], sr)

                row = {"file": damaged_name, "method": method}
                for key in fieldnames[2:]:
                    val = metrics.get(key, float("nan"))
                    row[key] = f"{val:.4f}" if not np.isnan(val) else "N/A"
                writer.writerow(row)

                # 保存修复后的音频
                sf.write(str(repaired_dir / f"repaired_{idx:04d}_{method}.wav"),
                         repaired, sr, subtype="PCM_16")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    logger.info(f"修复评估报告已保存: {args.output}")


def main() -> None:
    """批量评估主函数。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- 模式 C: 音频修复评估 ---
    if args.mode == "inpainting":
        run_inpainting_eval(args)
        return

    # --- 模式 B: 自动生成测试对 ---
    if args.clean_source and args.noise_source:
        snr_list = [float(s.strip()) for s in args.test_snr.split(",")]
        logger.info(f"自动生成 {args.num_test} 对测试样本 (SNR={snr_list}) ...")
        generate_test_pairs(
            Path(args.clean_source), Path(args.noise_source),
            Path(args.test_output_dir), args.num_test, snr_list,
            split_json=args.split_json,
        )
        # 生成后切换到模式 A 路径
        noisy_dir = Path(args.test_output_dir) / "noisy"
        clean_dir = Path(args.test_output_dir) / "clean"
    else:
        if not args.noisy_dir or not args.clean_dir:
            logger.error("请指定 --noisy_dir 和 --clean_dir，或使用 --clean_source 和 --noise_source 自动生成")
            sys.exit(1)
        noisy_dir = Path(args.noisy_dir)
        clean_dir = Path(args.clean_dir)

    noisy_files = sorted(noisy_dir.rglob("*.wav"))
    if not noisy_files:
        logger.error(f"未在 {noisy_dir} 中找到 WAV 文件")
        return

    # 准备去噪器
    denoisers = {}
    unet_model = None
    unet_device = None
    unet_ft_model = None
    for algo in args.algorithms:
        if algo == "wiener":
            from models.wiener import WienerFilter
            denoisers["wiener"] = WienerFilter()
        elif algo == "spectral_sub":
            from models.spectral_sub import SpectralSubtraction
            denoisers["spectral_sub"] = SpectralSubtraction()
        elif algo == "hybrid":
            from models.hybrid import HybridDenoiser
            hybrid_denoiser = HybridDenoiser()
            denoisers["hybrid"] = hybrid_denoiser
            logger.info("Hybrid 降噪器已就绪")
        elif algo == "unet_ft":
            import torch
            from models.unet import UNetDenoiser
            ft_device = _get_device()
            unet_ft_model = UNetDenoiser(n_fft=512, hop_length=256).to(ft_device)
            ft_ckpt = torch.load(args.ft_ckpt, map_location=ft_device)
            unet_ft_model.load_state_dict(ft_ckpt["model_state_dict"])
            unet_ft_model.eval()
            denoisers["unet_ft"] = None
            logger.info(f"微调 U-Net 已加载: {args.ft_ckpt}")
        elif algo == "unet":
            import torch
            from models.unet import UNetDenoiser
            unet_device = _get_device()
            unet_model = UNetDenoiser(n_fft=512, hop_length=256).to(unet_device)
            ckpt = torch.load(args.ckpt, map_location=unet_device)
            unet_model.load_state_dict(ckpt["model_state_dict"])
            unet_model.eval()
            denoisers["unet"] = None
            logger.info(f"U-Net 模型已加载: {args.ckpt}")

    # 写入 CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = [
        "file", "algorithm",
        "SNR (dB)", "SegSNR (dB)", "SI-SDR (dB)",
        "STOI", "PESQ_WB", "PESQ_NB", "LSD (dB)",
    ]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for nf in noisy_files:
            clean_path = clean_dir / nf.name
            if not clean_path.exists():
                logger.warning(f"跳过 (无纯净参考): {nf.name}")
                continue

            noisy, sr = load_audio(str(nf))
            clean, _ = load_audio(str(clean_path))

            for algo_name in args.algorithms:
                logger.info(f"处理: {nf.name} / {algo_name}")
                if algo_name == "unet":
                    denoised = unet_model.denoise_audio(noisy, sr)
                elif algo_name == "unet_ft":
                    denoised = unet_ft_model.denoise_audio(noisy, sr)
                elif algo_name == "hybrid":
                    denoised = denoisers["hybrid"].denoise_audio(noisy, sr, model_ckpt=args.ckpt)
                else:
                    denoised = denoisers[algo_name].denoise_audio(noisy, sr)
                metrics = compute_all_metrics(clean[:len(denoised)], denoised, sr)

                row = {"file": nf.name, "algorithm": algo_name}
                for key in fieldnames[2:]:
                    val = metrics.get(key, float("nan"))
                    row[key] = f"{val:.4f}" if not np.isnan(val) else "N/A"
                writer.writerow(row)

    logger.info(f"评估报告已保存: {args.output}")


if __name__ == "__main__":
    main()
