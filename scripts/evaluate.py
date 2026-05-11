"""
scripts/evaluate.py — 批量评估脚本

对指定目录下的音频对 (带噪+纯净) 批量执行所有算法，生成 CSV 对比报告。
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio
from evaluation.metrics import compute_all_metrics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="批量评估降噪算法")
    parser.add_argument(
        "--noisy_dir", type=str, required=True, help="带噪音频目录"
    )
    parser.add_argument(
        "--clean_dir", type=str, required=True, help="纯净音频目录 (文件名需对应)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_report.csv", help="输出 CSV 路径"
    )
    parser.add_argument(
        "--algorithms", type=str, nargs="+",
        default=["wiener", "spectral_sub"],
        help="评估的算法列表",
    )
    return parser.parse_args()


def main() -> None:
    """批量评估主函数。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    noisy_dir = Path(args.noisy_dir)
    clean_dir = Path(args.clean_dir)
    noisy_files = sorted(noisy_dir.glob("*.wav"))

    if not noisy_files:
        logger.error(f"未在 {noisy_dir} 中找到 WAV 文件")
        return

    # 准备去噪器
    denoisers = {}
    for algo in args.algorithms:
        if algo == "wiener":
            from models.wiener import WienerFilter
            denoisers["wiener"] = WienerFilter()
        elif algo == "spectral_sub":
            from models.spectral_sub import SpectralSubtraction
            denoisers["spectral_sub"] = SpectralSubtraction()

    # 写入 CSV
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

            for algo_name, denoiser in denoisers.items():
                logger.info(f"处理: {nf.name} / {algo_name}")
                denoised = denoiser.denoise_audio(noisy, sr)
                metrics = compute_all_metrics(clean[:len(denoised)], denoised, sr)

                row = {"file": nf.name, "algorithm": algo_name}
                for key in fieldnames[2:]:
                    val = metrics.get(key, float("nan"))
                    row[key] = f"{val:.4f}" if not np.isnan(val) else "N/A"
                writer.writerow(row)

    logger.info(f"评估报告已保存: {args.output}")


if __name__ == "__main__":
    main()
