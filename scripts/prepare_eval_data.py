"""
scripts/prepare_eval_data.py — 三域评估测试数据生成

从 EN (英文)、ZH (中文)、Sing (歌曲) 三个领域的纯净语音中各随机抽取 10 条，
生成降噪测试对 (noisy+clean) 和修复测试对 (damaged+clean)。

输出: evaluations/test_data/{denoising,inpainting}/{en,zh,sing}/{noisy|damaged,clean}/

用法:
  python scripts/prepare_eval_data.py
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio, normalize_rms, rms_energy

# 配置
NUM_SAMPLES = 10
SNR_RANGE = (0, 10)          # 降噪 SNR 范围 (dB)
DURATION = 4.0                # 片段时长 (秒)
TARGET_SR = 16000
NOISE_DIR = "datasets/processed/noise"
CLEAN_DIR = "datasets/processed/clean"
ZH_DIR = "datasets/finetune/processed/clean/zh"
SING_DIR = "datasets/finetune/processed/clean/sing"
SPLIT_JSON = "datasets/splits/test_clean.json"
OUTPUT_BASE = "evaluations/test_data"


def collect_files(root: Path, n: int, seed: int = 42) -> list[Path]:
    """从目录中随机抽取 n 个音频文件。

    Args:
        root: 扫描根目录.
        n: 抽取数量.
        seed: 随机种子.

    Returns:
        文件路径列表.
    """
    exts = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
    all_files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if len(all_files) < n:
        n = len(all_files)
    random.seed(seed)
    return random.sample(all_files, n)


def mix_noisy(clean_wav: np.ndarray, noise_dir: Path, snr_low: float, snr_high: float, sr: int) -> np.ndarray:
    """给纯净语音叠加随机噪声。

    Args:
        clean_wav: 纯净语音.
        noise_dir: 噪声源目录.
        snr_low/snr_high: SNR 范围.
        sr: 采样率.

    Returns:
        带噪语音.
    """
    noise_files = collect_files(noise_dir, 100, seed=random.randint(0, 9999))
    if not noise_files:
        return clean_wav
    nf = random.choice(noise_files)
    noise_full, _ = load_audio(str(nf), target_sr=sr)
    num_samples = len(clean_wav)
    if len(noise_full) < num_samples:
        reps = num_samples // len(noise_full) + 1
        noise_full = np.tile(noise_full, reps)
    start = random.randint(0, len(noise_full) - num_samples)
    noise_seg = noise_full[start : start + num_samples].astype(np.float32)

    snr = random.uniform(snr_low, snr_high)
    c_rms = rms_energy(clean_wav)
    n_rms = rms_energy(noise_seg)
    target_n_rms = c_rms / (10.0 ** (snr / 20.0))
    noise_seg = noise_seg * (target_n_rms / (n_rms + 1e-12))
    return (clean_wav + noise_seg).astype(np.float32)


def apply_damage(clean_wav: np.ndarray, sr: int) -> np.ndarray:
    """施加随机静音段损坏。

    Args:
        clean_wav: 纯净语音.
        sr: 采样率.

    Returns:
        损坏后的语音.
    """
    from models.audio_inpainter import AudioInpainter
    damaged, _ = AudioInpainter.generate_damaged(
        clean_wav, sr, num_silence=3, silence_min_ms=50, silence_max_ms=300, seed=random.randint(0, 9999),
    )
    return damaged.astype(np.float32)


def extract_segment(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    """从波形中截取固定长度片段。

    Args:
        waveform: 输入波形.
        num_samples: 目标采样点数.

    Returns:
        截取后的片段.
    """
    if len(waveform) < num_samples:
        reps = num_samples // len(waveform) + 1
        waveform = np.tile(waveform, reps)
    start = random.randint(0, len(waveform) - num_samples)
    return waveform[start : start + num_samples].astype(np.float32)


def main() -> None:
    """主函数。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    num_samples = int(DURATION * TARGET_SR)

    # ---- 收集三域文件 ----
    # EN: 从 test split 中过滤
    en_files: list[Path] = []
    split_path = Path(SPLIT_JSON)
    if split_path.exists():
        with open(split_path) as f:
            allowed = set(json.load(f))
        clean_root = Path(CLEAN_DIR)
        for rel in allowed:
            p = clean_root / rel
            if p.exists():
                en_files.append(p)
    en_files = random.sample(en_files, min(NUM_SAMPLES, len(en_files)))
    logger.info(f"EN: {len(en_files)} files from test split")

    # ZH / Sing
    zh_files = collect_files(Path(ZH_DIR), NUM_SAMPLES, seed=123)
    sing_files = collect_files(Path(SING_DIR), NUM_SAMPLES, seed=456)
    logger.info(f"ZH: {len(zh_files)}, Sing: {len(sing_files)}")

    noise_dir = Path(NOISE_DIR)

    # ---- 生成降噪测试对 ----
    for domain, files in [("en", en_files), ("zh", zh_files), ("sing", sing_files)]:
        noisy_dir = Path(OUTPUT_BASE) / "denoising" / domain / "noisy"
        clean_dir = Path(OUTPUT_BASE) / "denoising" / domain / "clean"
        noisy_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)

        for i, fp in enumerate(files):
            wav, _ = load_audio(str(fp), target_sr=TARGET_SR)
            seg = extract_segment(wav, num_samples)
            seg = normalize_rms(seg, target_db=-25.0)
            noisy = mix_noisy(seg, noise_dir, SNR_RANGE[0], SNR_RANGE[1], TARGET_SR)
            noisy = normalize_rms(noisy, target_db=-25.0)

            name = f"{domain}_{i:04d}.wav"
            sf.write(str(clean_dir / name), seg, TARGET_SR, subtype="PCM_16")
            sf.write(str(noisy_dir / name), noisy, TARGET_SR, subtype="PCM_16")
        logger.info(f"[denoising] {domain}: {len(files)} pairs saved")

    # ---- 生成修复测试对 ----
    for domain, files in [("en", en_files), ("zh", zh_files), ("sing", sing_files)]:
        damaged_dir = Path(OUTPUT_BASE) / "inpainting" / domain / "damaged"
        clean_dir = Path(OUTPUT_BASE) / "inpainting" / domain / "clean"
        damaged_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)

        for i, fp in enumerate(files):
            wav, _ = load_audio(str(fp), target_sr=TARGET_SR)
            seg = extract_segment(wav, num_samples)
            seg = normalize_rms(seg, target_db=-25.0)
            damaged = apply_damage(seg, TARGET_SR)

            name = f"{domain}_{i:04d}.wav"
            sf.write(str(clean_dir / name), seg, TARGET_SR, subtype="PCM_16")
            sf.write(str(damaged_dir / name), damaged, TARGET_SR, subtype="PCM_16")
        logger.info(f"[inpainting] {domain}: {len(files)} pairs saved")

    logger.info("全部完成！后续评估命令:")
    logger.info("  for d in en zh sing; do")
    logger.info("    python scripts/evaluate.py --noisy_dir evaluations/test_data/denoising/$d/noisy --clean_dir evaluations/test_data/denoising/$d/clean --algorithms wiener spectral_sub unet unet_ft hybrid --ckpt checkpoints/unet/best_model.pt --ft_ckpt checkpoints/unet_finetuned/best_model.pt --output evaluations/exp_denoise_$d.csv")
    logger.info("  done")


if __name__ == "__main__":
    main()
