"""
scripts/inference.py — 单文件推理脚本

命令行接口，对单个音频文件执行指定算法的降噪，并输出评估指标。
支持 --algo (wiener|spectral_sub) 和 --plot 保存对比图。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio
from evaluation.metrics import compute_all_metrics
from evaluation.visualizer import plot_comparison


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对单个音频文件执行降噪推理"
    )
    parser.add_argument("input", type=str, help="输入带噪音频文件路径")
    parser.add_argument(
        "--clean", type=str, default=None, help="纯净参考文件 (可选, 用于计算指标)"
    )
    parser.add_argument(
        "--output", type=str, default="denoised.wav", help="降噪输出文件路径"
    )
    parser.add_argument(
        "--algo", type=str, default="wiener",
        choices=["wiener", "spectral_sub", "unet", "hybrid", "inpaint"],
        help="降噪/修复算法选择",
    )
    parser.add_argument(
        "--ckpt", type=str, default="checkpoints/unet/best_model.pt",
        help="U-Net checkpoint 路径 (unet/inpaint 时需要)",
    )
    parser.add_argument(
        "--inpaint_method", type=str, default="spline",
        choices=["spline", "spectral", "unet"],
        help="音频修复方法 (仅 --algo inpaint 时有效)",
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="保存对比图路径 (可选)"
    )
    return parser.parse_args()


def main() -> None:
    """推理主函数。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # 加载音频
    waveform, sr = load_audio(args.input)
    logger.info(f"加载音频: {args.input} ({len(waveform)/sr:.1f}s, {sr}Hz)")

    # 选择算法
    if args.algo == "wiener":
        from models.wiener import WienerFilter
        denoiser = WienerFilter()
    elif args.algo == "spectral_sub":
        from models.spectral_sub import SpectralSubtraction
        denoiser = SpectralSubtraction()
    elif args.algo == "hybrid":
        from models.hybrid import HybridDenoiser
        h = HybridDenoiser()
        denoised = h.denoise_audio(waveform, sr, model_ckpt=args.ckpt)
        sf.write(args.output, denoised.astype(np.float32), sr)
        logger.info(f"Hybrid 降噪音频已保存: {args.output}")
        if args.plot:
            fig = plot_comparison(waveform, denoised, sr=sr)
            fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        return
    elif args.algo == "unet":
        import torch
        from models.unet import UNetDenoiser
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetDenoiser(n_fft=512, hop_length=256).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        denoised = model.denoise_audio(waveform, sr)
        sf.write(args.output, denoised.astype(np.float32), sr)
        logger.info(f"降噪音频已保存: {args.output}")
        if args.plot:
            fig = plot_comparison(waveform, denoised, sr=sr)
            fig.savefig(args.plot, dpi=150, bbox_inches="tight")
            logger.info(f"对比图已保存: {args.plot}")
        return
    else:  # inpaint
        from models.audio_inpainter import AudioInpainter
        inpainter = AudioInpainter()
        repaired = inpainter.inpaint(
            waveform, sr, method=args.inpaint_method, model_ckpt=args.ckpt,
        )
        sf.write(args.output, repaired.astype(np.float32), sr)
        logger.info(f"修复音频已保存: {args.output}")
        if args.plot:
            fig = plot_comparison(waveform, repaired, sr=sr)
            fig.savefig(args.plot, dpi=150, bbox_inches="tight")
            logger.info(f"对比图已保存: {args.plot}")
        return

    # 降噪
    logger.info(f"执行降噪算法: {args.algo}")
    denoised = denoiser.denoise_audio(waveform, sr)
    sf.write(args.output, denoised.astype(np.float32), sr)
    logger.info(f"降噪音频已保存: {args.output}")

    # 评估
    if args.clean:
        clean_waveform, _ = load_audio(args.clean)
        metrics = compute_all_metrics(clean_waveform, denoised, sr)
        logger.info("评估指标:")
        for name, val in metrics.items():
            if not np.isnan(val):
                logger.info(f"  {name}: {val:.4f}")

    # 可视化
    if args.plot:
        fig = plot_comparison(waveform, denoised, sr=sr)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        logger.info(f"对比图已保存: {args.plot}")


if __name__ == "__main__":
    main()
