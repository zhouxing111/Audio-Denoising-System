"""
ui/ — 交互表现层

基于 PySide6 + pyqtgraph 的图形用户界面。
提供三步工作流: 加载音频 → 选择算法 → 降噪展示，
集成波形图、频谱图、指标得分板、噪声诊断和音频回放。
"""

from .main_window import MainWindow
