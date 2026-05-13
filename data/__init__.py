"""
data/ — 数据工程层

负责音频数据的预处理、动态在线混合、数据增强和噪声诊断。
所有模块围绕 16kHz 单声道 WAV 格式设计。
"""

from .preprocess import load_audio, normalize_rms, resample_if_needed
from .dataset import DenoisingDataset, PremixedDataset
from .augment import (
    spec_augment,
    volume_perturb,
    speed_perturb,
    insert_mute_segment,
    apply_augmentations,
)
from .noise_diagnosis import diagnose_noise, classify_noise_type
