"""
scripts/premix_dataset.py — 离线预混合数据集

在训练前运行一次，预先生成 (noisy, clean) 训练对并保存为 .npy 文件。
大幅减少训练时的磁盘 I/O 开销，预期 10~20 倍加速。

用法:
  python scripts/premix_dataset.py --clean_dir datasets/processed/clean --noise_dir datasets/processed/noise --output_dir datasets/premixed --num_pairs 20000
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio, normalize_rms, rms_energy


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="离线预混合训练数据集")
    parser.add_argument("--clean_dir", type=str, required=True, help="纯净语音目录")
    parser.add_argument("--noise_dir", type=str, required=True, help="噪声目录")
    parser.add_argument("--output_dir", type=str, default="datasets/premixed", help="输出目录")
    parser.add_argument("--num_pairs", type=int, default=20000, help="生成的训练对数量")
    parser.add_argument("--duration", type=float, default=4.0, help="每段时长 (秒)")
    parser.add_argument("--snr_low", type=float, default=-5.0, help="最低 SNR (dB)")
    parser.add_argument("--snr_high", type=float, default=15.0, help="最高 SNR (dB)")
    parser.add_argument("--target_sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def collect_files(root: Path) -> list[Path]:
    """递归扫描目录收集所有音频文件。

    Args:
        root: 扫描根目录.

    Returns:
        音频文件路径列表.
    """
    exts = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按指定 SNR 混合纯净语音和噪声。

    Args:
        clean: 纯净语音.
        noise: 噪声 (等长).
        snr_db: 目标 SNR (dB).

    Returns:
        混合后的带噪语音.
    """
    clean_rms = rms_energy(clean)
    noise_rms = rms_energy(noise)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / (noise_rms + 1e-12))
    return clean + noise


def main() -> None:
    """离线预混合主函数。"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_dir = Path(args.clean_dir)
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(args.duration * args.target_sr)

    # 收集文件
    clean_files = collect_files(clean_dir)
    noise_files = collect_files(noise_dir)
    if not clean_files:
        logger.error(f"未在 {clean_dir} 找到音频文件")
        sys.exit(1)
    if not noise_files:
        logger.error(f"未在 {noise_dir} 找到音频文件")
        sys.exit(1)
    logger.info(f"纯净语音: {len(clean_files)} files, 噪声: {len(noise_files)} files")

    # 预加载所有噪声到内存 (噪声文件通常较少，可全部加载)
    noise_cache: list[np.ndarray] = []
    for nf in tqdm(noise_files, desc="Loading noise files"):
        try:
            wf, _ = load_audio(str(nf), target_sr=args.target_sr)
            if len(wf) >= num_samples:
                noise_cache.append(wf)
        except Exception as e:
            logger.warning(f"跳过 {nf.name}: {e}")
    logger.info(f"已加载 {len(noise_cache)} 个噪声文件到内存")

    # 生成预混合对
    logger.info(f"开始生成 {args.num_pairs} 个训练对...")
    for i in tqdm(range(args.num_pairs)):
        # 随机选纯净语音
        cf = random.choice(clean_files)
        try:
            clean_full, _ = load_audio(str(cf), target_sr=args.target_sr)
        except Exception:
            continue

        # 截取固定长度片段
        if len(clean_full) < num_samples:
            repeats = num_samples // len(clean_full) + 1
            clean_full = np.tile(clean_full, repeats)
        start_c = random.randint(0, len(clean_full) - num_samples)
        clean_seg = clean_full[start_c : start_c + num_samples].astype(np.float32)

        # 随机选噪声
        noise_full = random.choice(noise_cache)
        start_n = random.randint(0, len(noise_full) - num_samples)
        noise_seg = noise_full[start_n : start_n + num_samples].astype(np.float32)

        # 混合
        snr = random.uniform(args.snr_low, args.snr_high)
        noisy = mix_at_snr(clean_seg, noise_seg, snr)

        # RMS 归一化
        noisy = normalize_rms(noisy, target_db=-25.0)
        clean_seg = normalize_rms(clean_seg, target_db=-25.0)

        # 保存 .npy 对
        np.save(output_dir / f"noisy_{i:06d}.npy", noisy)
        np.save(output_dir / f"clean_{i:06d}.npy", clean_seg)

    logger.info(f"预混合完成！{args.num_pairs} 对已保存至 {output_dir}/")
    logger.info("下一步训练命令:")
    logger.info(f"  python scripts/train.py --use_premix --premix_dir {output_dir}")


if __name__ == "__main__":
    main()
