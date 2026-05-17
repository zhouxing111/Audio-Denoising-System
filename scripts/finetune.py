"""
scripts/finetune.py — U-Net 轻量微调

基于预训练权重，冻结 Encoder 前 4 层，仅训练高层特征 + Decoder。
适配中文/歌曲/英文对话等新领域，降低过拟合风险。

用法:
  python scripts/finetune.py --clean_dir datasets/finetune/processed/clean --noise_dir datasets/finetune/processed/noise --pretrained checkpoints/unet/best_model.pt --output_dir checkpoints/unet_finetuned --epochs 15
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import DenoisingDataset
from models.unet import UNetDenoiser
from scripts.train import compute_irm, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="微调 U-Net 降噪模型")
    parser.add_argument("--clean_dir", type=str, required=True, help="微调纯净语音目录")
    parser.add_argument("--noise_dir", type=str, required=True, help="微调噪声目录")
    parser.add_argument("--pretrained", type=str, required=True, help="预训练权重路径")
    parser.add_argument("--output_dir", type=str, default="checkpoints/unet_finetuned", help="输出目录")
    parser.add_argument("--config", type=str, default="config/unet.yaml", help="模型配置文件")
    parser.add_argument("--epochs", type=int, default=15, help="微调 epoch 数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率 (比从头训练小 20 倍)")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    return parser.parse_args()


def freeze_encoder_layers(model: nn.Module, freeze_up_to: int = 4) -> None:
    """冻结 Encoder 前 N 层的参数。

    Args:
        model: U-Net 模型.
        freeze_up_to: 冻结到第几层 (enc1 ~ encN), 默认 4.
    """
    layers_to_freeze = []
    for i in range(1, freeze_up_to + 1):
        layers_to_freeze.append(getattr(model, f"enc{i}"))
        if i < freeze_up_to:
            layers_to_freeze.append(model.pool)  # 只冻结对应的 pool

    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False

    # 统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    logging.getLogger(__name__).info(
        f"参数: 总 {total:,} | 可训练 {trainable:,} ({trainable/total*100:.1f}%) | 冻结 {frozen:,}"
    )


def train_epoch_finetune(
    model, dataloader, optimizer, device, n_fft=512, hop_length=256,
    mask_weight=1.0, mag_weight=0.5, gradient_clip=1.0,
) -> float:
    """微调训练一个 epoch。

    Args:
        model: U-Net 模型.
        dataloader: 数据加载器.
        optimizer: 优化器.
        device: 设备.
        n_fft/hop_length: STFT 参数.
        mask_weight/mag_weight: 损失权重.
        gradient_clip: 梯度裁剪.

    Returns:
        平均 loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    window = torch.hann_window(n_fft, device=device)

    pbar = tqdm(dataloader, desc="Fine-tuning")
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        noisy_stft = torch.stft(noisy, n_fft=n_fft, hop_length=hop_length,
                                win_length=n_fft, window=window, return_complex=True, onesided=True)
        clean_stft = torch.stft(clean, n_fft=n_fft, hop_length=hop_length,
                                win_length=n_fft, window=window, return_complex=True, onesided=True)
        noisy_mag = noisy_stft.abs().unsqueeze(1)
        clean_mag = clean_stft.abs().unsqueeze(1)

        target_irm = compute_irm(clean_mag, noisy_mag)
        pred_mask = model(noisy_mag)

        loss_mask = nn.functional.mse_loss(pred_mask, target_irm)
        denoised_mag = pred_mask * noisy_mag
        loss_mag = nn.functional.l1_loss(denoised_mag, clean_mag)
        loss = mask_weight * loss_mask + mag_weight * loss_mag

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n_batches


def main() -> None:
    """微调主函数。"""
    args = parse_args()
    setup_logging("logs")
    logger = logging.getLogger(__name__)
    cfg = load_config(args.config, "config/default.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 数据集
    dataset = DenoisingDataset(
        clean_dir=args.clean_dir, noise_dir=args.noise_dir,
        sample_rate=cfg["audio"]["sample_rate"], duration=cfg["audio"]["duration"],
        snr_low=cfg["mixing"]["snr_low"], snr_high=cfg["mixing"]["snr_high"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=cfg["training"]["num_workers"], pin_memory=True)
    logger.info(f"微调数据集: {len(dataset)} steps/epoch")

    # 加载预训练模型
    model = UNetDenoiser(n_fft=cfg["model"]["n_fft"], hop_length=cfg["model"]["hop_length"])
    ckpt = torch.load(args.pretrained, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    freeze_encoder_layers(model, freeze_up_to=4)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    finetune_cfg = {"pretrained": args.pretrained, "epochs": args.epochs, "lr": args.lr,
                    "clean_dir": args.clean_dir, "noise_dir": args.noise_dir}
    with open(os.path.join(args.output_dir, "finetune_config.yaml"), "w") as f:
        yaml.dump(finetune_cfg, f)

    # CSV 记录
    csv_path = os.path.join("logs", "finetune_history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "lr"])

    best_loss = float("inf")
    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_epoch_finetune(
            model, loader, optimizer, device,
            n_fft=cfg["model"]["n_fft"], hop_length=cfg["model"]["hop_length"],
        )
        logger.info(f"Epoch {epoch + 1} | Loss: {train_loss:.4f}")
        csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{args.lr:.8f}"])
        csv_file.flush()

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
            }, os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"保存最佳模型 (loss={best_loss:.4f})")

    csv_file.close()
    logger.info(f"微调完成！模型保存至 {args.output_dir}/best_model.pt")

    # 自动生成微调训练曲线
    from evaluation.visualizer import plot_training_curves
    fig_path = os.path.join("logs", "finetune_curves.png")
    fig = plot_training_curves(csv_path, title="U-Net Fine-tuning Curves")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"微调训练曲线图: {fig_path}")


if __name__ == "__main__":
    main()
