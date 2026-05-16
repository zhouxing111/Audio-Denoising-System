"""
scripts/prepare_finetune_data.py — 微调数据预处理

扫描 datasets/finetune/raw/ 下的中文/英文/歌曲/噪声原始文件，
统一转换为 16kHz 单声道 WAV 并切分为 4s 片段。

用法:
  python scripts/prepare_finetune_data.py --raw_dir datasets/finetune/raw --output_dir datasets/finetune/processed --duration 4.0
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.preprocess import load_audio


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="预处理微调数据集")
    parser.add_argument("--raw_dir", type=str, default="datasets/finetune/raw", help="原始数据根目录")
    parser.add_argument("--output_dir", type=str, default="datasets/finetune/processed", help="输出目录")
    parser.add_argument("--duration", type=float, default=4.0, help="切分片段时长 (秒)")
    parser.add_argument("--target_sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--librispeech_test", type=str, default=None,
                        help="LibriSpeech test-clean 路径, en/ 为空时自动从中抽取英文对话")
    return parser.parse_args()


def process_category(raw_subdir: Path, out_subdir: Path, args) -> int:
    """处理一个数据类别的所有音频文件。

    Args:
        raw_subdir: 原始数据子目录 (如 raw/zh/).
        out_subdir: 输出子目录.
        args: 命令行参数.

    Returns:
        生成的文件数量.
    """
    exts = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
    files = sorted([p for p in raw_subdir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        logging.warning(f"未在 {raw_subdir} 找到音频文件，跳过")
        return 0

    out_subdir.mkdir(parents=True, exist_ok=True)
    num_samples = int(args.duration * args.target_sr)
    cat_name = raw_subdir.name
    total = 0

    for fpath in tqdm(files, desc=f"Processing {cat_name}"):
        try:
            waveform, sr = load_audio(str(fpath), target_sr=args.target_sr)
        except Exception as e:
            logging.warning(f"跳过 {fpath.name}: {e}")
            continue

        if len(waveform) < num_samples:
            # 短文件循环填充
            repeats = num_samples // len(waveform) + 1
            waveform = np.tile(waveform, repeats)[:num_samples]
            out_name = f"{cat_name}_{total:04d}.wav"
            sf.write(str(out_subdir / out_name), waveform.astype(np.float32), args.target_sr, subtype="PCM_16")
            total += 1
        else:
            # 长文件切分为多个片段
            n_chunks = len(waveform) // num_samples
            for i in range(n_chunks):
                chunk = waveform[i * num_samples : (i + 1) * num_samples]
                out_name = f"{cat_name}_{total:04d}.wav"
                sf.write(str(out_subdir / out_name), chunk.astype(np.float32), args.target_sr, subtype="PCM_16")
                total += 1

    return total


def main() -> None:
    """预处理主函数。"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    raw_dir = Path(args.raw_dir)
    out_clean = Path(args.output_dir) / "clean"

    # 英文对话：如果 en/ 为空且提供了 test-clean 路径，自动从中抽取
    en_dir = raw_dir / "en"
    if (not en_dir.exists() or not any(en_dir.iterdir())) and args.librispeech_test:
        libri_test = Path(args.librispeech_test)
        if libri_test.exists():
            # 选前 10 个 speaker
            import shutil
            speakers = sorted([d for d in libri_test.iterdir() if d.is_dir()])[:10]
            en_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for spk in speakers:
                for flac in spk.rglob("*.flac"):
                    shutil.copy2(flac, en_dir / flac.name)
                    count += 1
                    if count >= 300:
                        break
                if count >= 300:
                    break
            logger.info(f"从 LibriSpeech test-clean 自动抽取 {count} 条英文对话到 {en_dir}")

    # 处理各类型纯净语音
    clean_categories = ["zh", "en", "sing"]
    grand_total = 0
    for cat in clean_categories:
        raw_sub = raw_dir / cat
        if not raw_sub.exists():
            logger.warning(f"目录不存在: {raw_sub}，跳过")
            continue
        n = process_category(raw_sub, out_clean / cat, args)
        logger.info(f"  {cat}: {n} 个 {args.duration}s 片段")
        grand_total += n

    # 处理噪声
    raw_noise = raw_dir / "noise"
    if raw_noise.exists():
        out_noise = Path(args.output_dir) / "noise"
        n_noise = process_category(raw_noise, out_noise / "esc50", args)
        logger.info(f"  noise: {n_noise} 个片段")
    else:
        logger.warning(f"噪声目录不存在: {raw_noise}，请放入 ESC-50 数据")

    logger.info(f"完成！共 {grand_total} 个纯净语音片段")
    logger.info(f"下一步: python scripts/finetune.py --clean_dir {out_clean} --noise_dir {Path(args.output_dir) / 'noise'} --pretrained checkpoints/unet/best_model.pt --output_dir checkpoints/unet_finetuned")


if __name__ == "__main__":
    main()
