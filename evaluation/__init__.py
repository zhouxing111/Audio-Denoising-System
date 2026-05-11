"""
evaluation/ — 评估基准层

封装全部客观声学指标和可视化工具。
提供统一的评分函数接口，输入 (clean, denoised, sr) 返回指标字典。
"""

from .metrics import compute_all_metrics
from .visualizer import plot_waveform, plot_spectrogram, plot_mel_spectrogram
