"""
scripts/prepare_data.py — 数据集自动化预处理

扫描原始 LibriSpeech (.flac) 和 DEMAND (.wav) 文件，
统一转换为 16kHz 单声道 WAV，按 speaker_id / 噪声类型组织输出，
并生成说话人级 train/val/test 分割列表 (防泄露)。

用法:
  python scripts/prepare_data.py \
    --raw_dir datasets/raw \
    --output_dir datasets/processed \
    --val_ratio 0.1 --test_ratio 0.1
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# DEMAND 通道名 → 场景名 映射表
DEMAND_CHANNEL_MAP = {
    "ch01": "kitchen",
    "ch02": "traffic",
    "ch03": "cafeteria",
    "ch04": "restaurant",
    "ch05": "station",
    "ch06": "airport",
    "ch07": "park",
    "ch08": "library",
    "ch09": "meeting",
    "ch10": "office",
    "ch11": "bedroom",
    "ch12": "bathroom",
    "ch13": "livingroom",
    "ch14": "laundry",
    "ch15": "workshop",
    "ch16": "construction",
    "ch17": "nature",
    "ch18": "square",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="预处理 LibriSpeech 和 DEMAND 数据集为统一 16kHz WAV 格式"
    )
    parser.add_argument(
        "--raw_dir", type=str, default="datasets/raw",
        help="原始数据集根目录 (含 LibriSpeech/ 和 DEMAND/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/processed",
        help="预处理输出根目录",
    )
    parser.add_argument(
        "--splits_dir", type=str, default="datasets/splits",
        help="train/val/test 分割列表输出目录",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="验证集说话人比例",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="测试集说话人比例",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (保证分割可复现)",
    )
    parser.add_argument(
        "--target_sr", type=int, default=16000,
        help="目标采样率 (Hz)",
    )
    parser.add_argument(
        "--chunk_duration", type=float, default=10.0,
        help="噪声切分时长 (秒), 将长文件切成等长片段以降低训练 I/O",
    )
    return parser.parse_args()


def find_librispeech_files(raw_dir: Path) -> list[dict]:
    """扫描 LibriSpeech 目录，返回 (.flac 路径, speaker_id) 列表。

    LibriSpeech 目录结构: train-clean-100/{speaker_id}/{chapter_id}/{utterance}.flac

    Args:
        raw_dir: LibriSpeech 根目录路径.

    Returns:
        字典列表 [{"path": Path, "speaker_id": str, "filename": str}, ...].
    """
    libri_root = raw_dir / "LibriSpeech"
    if not libri_root.exists():
        logging.warning(f"LibriSpeech 目录不存在: {libri_root}")
        return []

    files = []
    for flac_path in libri_root.rglob("*.flac"):
        # 路径格式: .../train-clean-100/19/198/19-198-0000.flac
        parts = flac_path.parts
        speaker_id = None
        for i, p in enumerate(parts):
            if p.startswith("train-clean") and i + 1 < len(parts):
                speaker_id = parts[i + 1]
                break
        if speaker_id is None:
            speaker_id = parts[-3]  # fallback
        files.append({
            "path": flac_path,
            "speaker_id": speaker_id,
            "filename": flac_path.stem + ".wav",
        })
    return files


def find_demand_files(raw_dir: Path) -> list[dict]:
    """扫描 DEMAND 目录，返回 (.wav 路径, 噪声类型) 列表。

    DEMAND 目录结构: DEMAND/{channel}/{channel}_16k.wav

    Args:
        raw_dir: DEMAND 根目录路径.

    Returns:
        字典列表 [{"path": Path, "noise_type": str, "filename": str}, ...].
    """
    demand_root = raw_dir / "DEMAND"
    if not demand_root.exists():
        logging.warning(f"DEMAND 目录不存在: {demand_root}")
        return []

    files = []
    for wav_path in demand_root.rglob("*.wav"):
        # 从父目录名提取通道号
        channel = wav_path.parent.name.lower()  # e.g. "ch01"
        noise_type = DEMAND_CHANNEL_MAP.get(channel, channel)
        files.append({
            "path": wav_path,
            "noise_type": noise_type,
            "filename": noise_type + "_" + wav_path.name,
        })
    return files


def convert_to_mono_16k(
    src_path: Path, target_sr: int = 16000
) -> np.ndarray:
    """读取任意格式音频，转换为 16kHz 单声道 float32 numpy 数组。

    Args:
        src_path: 源文件路径.
        target_sr: 目标采样率 (Hz).

    Returns:
        转换后的音频波形, shape (n_samples,).
    """
    import librosa

    waveform, orig_sr = sf.read(src_path, dtype="float32")
    # 多声道 → 单声道
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    # 重采样
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform.astype(np.float32)


def process_clean_speech(
    files: list[dict], output_dir: Path, target_sr: int
) -> dict[str, list[str]]:
    """转换纯净语音文件并按 speaker_id 组织输出。

    Args:
        files: find_librispeech_files 返回的文件列表.
        output_dir: 输出根目录 (clean/ 将创建在此下).
        target_sr: 目标采样率.

    Returns:
        speaker_files: {speaker_id: [relative_wav_path, ...]}.
    """
    clean_dir = output_dir / "clean"
    speaker_files: dict[str, list[str]] = {}

    for item in tqdm(files, desc="Processing LibriSpeech"):
        sid = item["speaker_id"]
        out_subdir = clean_dir / f"speaker_{sid}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / item["filename"]

        if not out_path.exists():
            waveform = convert_to_mono_16k(item["path"], target_sr)
            sf.write(str(out_path), waveform, target_sr, subtype="PCM_16")

        rel_path = f"speaker_{sid}/{item['filename']}"
        speaker_files.setdefault(sid, []).append(rel_path)

    return speaker_files


def process_noise(
    files: list[dict], output_dir: Path, target_sr: int,
    chunk_duration: float = 10.0,
) -> int:
    """转换噪声文件并按噪声类型组织输出。

    长噪声文件 (如 DEMAND 的 5 分钟) 将被切分为等长片段，
    防止训练时每次加载 19MB 文件只取 4s 的 I/O 浪费。

    Args:
        files: find_demand_files 返回的文件列表.
        output_dir: 输出根目录 (noise/ 将创建在此下).
        target_sr: 目标采样率.
        chunk_duration: 切分片段时长 (秒), 0 表示不切分.

    Returns:
        已处理的噪声 chunk 总数.
    """
    noise_dir = output_dir / "noise"
    chunk_samples = int(chunk_duration * target_sr) if chunk_duration > 0 else 0
    total = 0

    for item in tqdm(files, desc="Processing DEMAND"):
        out_subdir = noise_dir / item["noise_type"]
        out_subdir.mkdir(parents=True, exist_ok=True)

        waveform = convert_to_mono_16k(item["path"], target_sr)
        n_total = len(waveform)

        if chunk_samples > 0 and n_total > chunk_samples:
            # 切分为多个等长片段
            n_chunks = n_total // chunk_samples
            for i in range(n_chunks):
                chunk = waveform[i * chunk_samples : (i + 1) * chunk_samples]
                out_name = f"{item['noise_type']}_{i:04d}.wav"
                out_path = out_subdir / out_name
                if not out_path.exists():
                    sf.write(str(out_path), chunk, target_sr, subtype="PCM_16")
                total += 1
            # 尾部不足一个 chunk 的部分丢弃
        else:
            out_path = out_subdir / item["filename"]
            if not out_path.exists():
                sf.write(str(out_path), waveform, target_sr, subtype="PCM_16")
            total += 1
    return total


def split_speakers(
    speaker_files: dict[str, list[str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """按 speaker_id 随机分割 train/val/test 集。

    保证同一个说话人的所有文件只在其中一个集合中，防止泄露。

    Args:
        speaker_files: {speaker_id: [relative_paths]}.
        val_ratio: 验证集说话人比例.
        test_ratio: 测试集说话人比例.
        seed: 随机种子.

    Returns:
        (train_files, val_files, test_files): 三个文件相对路径列表.
    """
    random.seed(seed)
    speaker_ids = list(speaker_files.keys())
    random.shuffle(speaker_ids)

    n_total = len(speaker_ids)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    test_sids = set(speaker_ids[:n_test])
    val_sids = set(speaker_ids[n_test : n_test + n_val])
    train_sids = set(speaker_ids[n_test + n_val:])

    def _collect(sids: set[str]) -> list[str]:
        files = []
        for sid in sids:
            files.extend(speaker_files[sid])
        return sorted(files)

    train_files = _collect(train_sids)
    val_files = _collect(val_sids)
    test_files = _collect(test_sids)

    logging.info(
        f"说话人分割: train={len(train_sids)} spk / {len(train_files)} files, "
        f"val={len(val_sids)} spk / {len(val_files)} files, "
        f"test={len(test_sids)} spk / {len(test_files)} files"
    )
    return train_files, val_files, test_files


def save_splits(
    train: list[str],
    val: list[str],
    test: list[str],
    splits_dir: Path,
) -> None:
    """将分割列表保存为 JSON 文件。

    Args:
        train/val/test: 文件相对路径列表.
        splits_dir: 输出目录.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, files in [("train", train), ("val", val), ("test", test)]:
        out_path = splits_dir / f"{name}_clean.json"
        with open(out_path, "w") as f:
            json.dump(files, f, indent=2, ensure_ascii=False)
        logging.info(f"写入 {out_path} ({len(files)} files)")


def main() -> None:
    """预处理主函数。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)

    # ---- 1. 扫描源文件 ----
    clean_files = find_librispeech_files(raw_dir)
    noise_files = find_demand_files(raw_dir)

    if not clean_files:
        logging.error(
            f"未找到 LibriSpeech .flac 文件，请确保 {raw_dir / 'LibriSpeech'} 目录存在"
        )
        sys.exit(1)
    if not noise_files:
        logging.error(
            f"未找到 DEMAND .wav 文件，请确保 {raw_dir / 'DEMAND'} 目录存在"
        )
        sys.exit(1)

    logging.info(
        f"扫描完成: {len(clean_files)} 纯净语音, {len(noise_files)} 噪声文件"
    )

    # ---- 2. 转换纯净语音 ----
    speaker_files = process_clean_speech(clean_files, output_dir, args.target_sr)

    # ---- 3. 转换噪声 (自动切分为 chunk 以降低训练 I/O) ----
    noise_total = process_noise(
        noise_files, output_dir, args.target_sr, chunk_duration=args.chunk_duration,
    )

    # ---- 4. 说话人级分割 ----
    train, val, test = split_speakers(
        speaker_files,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    save_splits(train, val, test, splits_dir)

    # ---- 5. 统计 ----
    n_speakers = len(speaker_files)
    n_clean_files = sum(len(v) for v in speaker_files.values())
    logging.info("=" * 50)
    logging.info(f"预处理完成:")
    logging.info(f"  纯净语音: {n_clean_files} files, {n_speakers} speakers")
    logging.info(f"  噪声:     {noise_total} files, {len(DEMAND_CHANNEL_MAP)} types")
    logging.info(f"  输出路径: {output_dir}")
    logging.info(f"  分割列表: {splits_dir}")
    logging.info("=" * 50)
    logging.info("下一步训练命令:")
    logging.info(
        f"  python scripts/train.py "
        f"--clean_dir {output_dir / 'clean'} "
        f"--noise_dir {output_dir / 'noise'}"
    )


if __name__ == "__main__":
    main()
