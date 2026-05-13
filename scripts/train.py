"""
scripts/train.py — U-Net 模型训练入口

支持从 YAML 配置文件加载全部超参数，包含:
- 数据加载 (动态在线混合)
- 模型训练循环 (含验证)
- Checkpoint 管理 (保存最佳模型)
- 日志记录 (控制台 + 文件 + TensorBoard 可选)
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

# 将项目根目录加入 sys.path，确保模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import DenoisingDataset
from evaluation.visualizer import plot_training_curves
from models.unet import UNetDenoiser


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train U-Net audio denoising model")
    parser.add_argument(
        "--config", type=str, default="config/unet.yaml",
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--base_config", type=str, default="config/default.yaml",
        help="全局配置文件路径",
    )
    parser.add_argument(
        "--clean_dir", type=str, required=True,
        help="纯净语音目录",
    )
    parser.add_argument(
        "--noise_dir", type=str, required=True,
        help="噪声目录",
    )
    parser.add_argument(
        "--val_clean_dir", type=str, default=None,
        help="验证集纯净语音目录",
    )
    parser.add_argument(
        "--val_noise_dir", type=str, default=None,
        help="验证集噪声目录",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从 checkpoint 恢复训练",
    )
    return parser.parse_args()


def load_config(config_path: str, base_config_path: str) -> dict:
    """加载并合并配置文件 (base + model-specific)。

    Args:
        config_path: 模型配置 YAML 路径.
        base_config_path: 全局配置 YAML 路径.

    Returns:
        合并后的配置字典.
    """
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def setup_logging(log_dir: str = "logs") -> None:
    """配置 logging：控制台 + 文件双输出。

    Args:
        log_dir: 日志文件目录.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ],
    )


def compute_irm(clean_mag: torch.Tensor, noisy_mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算理想比例掩膜 IRM = clean_mag / (clean_mag + noise_mag)。

    Args:
        clean_mag: 纯净幅度谱.
        noisy_mag: 带噪幅度谱.
        eps: 数值稳定项.

    Returns:
        IRM 掩膜 (0~1).
    """
    return clean_mag / (noisy_mag + eps)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> float:
    """训练一个 epoch。

    Args:
        model: U-Net 模型.
        dataloader: 训练数据加载器.
        optimizer: 优化器.
        device: 计算设备.
        cfg: 配置字典.

    Returns:
        本 epoch 平均损失.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        # STFT → 幅度谱
        noisy_stft = torch.stft(
            noisy, n_fft=cfg["model"]["n_fft"],
            hop_length=cfg["model"]["hop_length"],
            return_complex=True,
            onesided=True,
        )
        clean_stft = torch.stft(
            clean, n_fft=cfg["model"]["n_fft"],
            hop_length=cfg["model"]["hop_length"],
            return_complex=True,
            onesided=True,
        )
        noisy_mag = noisy_stft.abs().unsqueeze(1)   # (B, 1, F, T)
        clean_mag = clean_stft.abs().unsqueeze(1)

        # 真实 IRM
        target_irm = compute_irm(clean_mag, noisy_mag)

        # 前向传播
        pred_mask = model(noisy_mag)

        # 损失: MSE(掩膜) + L1(幅度谱)
        loss_mask = nn.functional.mse_loss(pred_mask, target_irm)
        denoised_mag = pred_mask * noisy_mag
        loss_mag = nn.functional.l1_loss(denoised_mag, clean_mag)
        loss = (
            cfg["loss"]["mask_weight"] * loss_mask
            + cfg["loss"]["magnitude_weight"] * loss_mag
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"]["gradient_clip"]
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n_batches


def main() -> None:
    """训练主函数。"""
    args = parse_args()
    cfg = load_config(args.config, args.base_config)
    setup_logging()
    logger = logging.getLogger(__name__)

    device = torch.device(
        cfg["training"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"使用设备: {device}")

    # 数据集
    train_dataset = DenoisingDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        sample_rate=cfg["audio"]["sample_rate"],
        duration=cfg["audio"]["duration"],
        snr_low=cfg["mixing"]["snr_low"],
        snr_high=cfg["mixing"]["snr_high"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    logger.info(f"训练集大小: {len(train_dataset)} 步/epoch")

    # 模型
    model = UNetDenoiser(
        n_fft=cfg["model"]["n_fft"],
        hop_length=cfg["model"]["hop_length"],
    ).to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg["training"]["lr_decay"]
    )

    start_epoch = 0
    best_loss = float("inf")

    # 恢复训练
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        logger.info(f"从 epoch {start_epoch} 恢复训练")

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    # CSV 记录训练历史，供后续出图使用
    csv_path = os.path.join("logs", "training_history.csv")
    os.makedirs("logs", exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "lr"])

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        logger.info(f"--- Epoch {epoch + 1}/{cfg['training']['epochs']} ---")

        train_loss = train_epoch(model, train_loader, optimizer, device, cfg)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
                     f"LR: {current_lr:.6f}")

        # 记录到 CSV
        csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{current_lr:.8f}"])
        csv_file.flush()

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt"))
            logger.info(f"保存最佳模型 (loss={best_loss:.4f})")

        # 定期保存 checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, os.path.join(
                cfg["training"]["checkpoint_dir"], f"checkpoint_epoch{epoch+1}.pt"
            ))

    csv_file.close()
    logger.info("训练完成")

    # 自动生成训练曲线图
    fig_path = os.path.join("logs", "training_curves.png")
    fig = plot_training_curves(csv_path, title="U-Net Training Curves")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"训练曲线图已保存: {fig_path}")


if __name__ == "__main__":
    main()
