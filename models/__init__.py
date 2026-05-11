"""
models/ — 算法引擎层

提供统一的音频降噪接口 BaseDenoiser，以及传统和深度学习实现。
所有模型必须实现 forward() (训练) 和 denoise_audio() (推理).
"""

from .base import BaseDenoiser
from .wiener import WienerFilter
from .spectral_sub import SpectralSubtraction
