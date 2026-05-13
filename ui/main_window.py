"""
ui/main_window.py — 主窗口 (完整版)

双模式工作流:
  模式A (文件加载): 加载音频 → 选择算法 → 一键降噪 → 全部面板同步更新
  模式B (在线录音): 录制音频 → 自动存入 → 选择算法 → 降噪展示 (复用后端)

集成全部 UI 组件:
  - WaveformView: 时域波形对比
  - SpectrogramView: 左右频谱图对比
  - MetricsPanel: 评估指标得分板
  - DiagnosisPanel: 噪声类型诊断
  - AudioPlayer: 音频回放控制
  - AudioRecorder: 麦克风在线录音
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio
from data.noise_diagnosis import diagnose_noise
from evaluation.metrics import compute_all_metrics

from .widgets.metrics_panel import MetricsPanel
from .widgets.waveform_view import WaveformView
from .widgets.spectrogram_view import SpectrogramView
from .widgets.diagnosis_panel import DiagnosisPanel
from .audio_player import AudioPlayer
from .audio_recorder import AudioRecorder

logger = logging.getLogger(__name__)


class DenoiseWorker(QThread):
    """后台线程执行降噪计算，避免阻塞 UI。

    支持三种算法: Wiener Filter, Spectral Subtraction, U-Net.
    U-Net 需要预先训练好 checkpoint (via scripts/train.py).
    """

    finished = Signal(np.ndarray, int, str)
    error = Signal(str)
    progress = Signal(int)

    def __init__(
        self, waveform: np.ndarray, sr: int, algorithm: str,
        model_ckpt: str | None = None, device: str = "cpu",
    ):
        """初始化降噪工作线程。

        Args:
            waveform: 带噪音频波形.
            sr: 采样率.
            algorithm: 算法名称.
            model_ckpt: U-Net checkpoint 路径 (仅 U-Net 需要).
            device: 推理设备 (cpu/cuda).
        """
        super().__init__()
        self.waveform = waveform
        self.sr = sr
        self.algorithm = algorithm
        self.model_ckpt = model_ckpt
        self.device = device

    def run(self) -> None:
        """线程主函数：执行降噪计算。"""
        try:
            self.progress.emit(10)
            if self.algorithm == "Wiener Filter":
                from models.wiener import WienerFilter
                denoiser = WienerFilter()
            elif self.algorithm == "Spectral Subtraction":
                from models.spectral_sub import SpectralSubtraction
                denoiser = SpectralSubtraction()
            elif self.algorithm == "U-Net":
                import torch
                from models.unet import UNetDenoiser
                if not self.model_ckpt or not Path(self.model_ckpt).exists():
                    raise FileNotFoundError(
                        f"U-Net checkpoint 未找到: {self.model_ckpt}。"
                        f"请先运行 scripts/train.py 训练模型，"
                        f"或通过 --ckpt 参数指定 checkpoint 路径。"
                    )
                self.progress.emit(30)
                device = torch.device(self.device if torch.cuda.is_available() else "cpu")
                model = UNetDenoiser(n_fft=512, hop_length=256).to(device)
                ckpt = torch.load(self.model_ckpt, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self.progress.emit(50)
                denoised = model.denoise_audio(self.waveform, self.sr)
                self.progress.emit(90)
                self.finished.emit(denoised.astype(np.float32), self.sr, self.algorithm)
                self.progress.emit(100)
                return
            else:
                raise ValueError(f"未知算法: {self.algorithm}")
            self.progress.emit(50)
            denoised = denoiser.denoise_audio(self.waveform, self.sr)
            self.progress.emit(90)
            self.finished.emit(denoised.astype(np.float32), self.sr, self.algorithm)
            self.progress.emit(100)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """智能音频降噪系统主窗口 (完整版)。

    布局结构:
      [控制面板: 加载 | 录制 | 算法 | 降噪 | 导出]
      [录音面板 (可折叠)]
      [Tab: 波形对比 | 频谱图]
      [Tab: 评估指标 | 噪声诊断]
      [音频回放控制]
      [状态栏 + 进度条]
    """

    def __init__(self, model_ckpt: str | None = None):
        super().__init__()
        self.setWindowTitle("智能音频降噪系统")
        self.setMinimumSize(1100, 800)
        self._waveform = None
        self._denoised = None
        self._sr = None
        self._current_algo = None
        self._model_ckpt = model_ckpt
        self._setup_ui()

    def _setup_ui(self) -> None:
        """构建完整 UI 布局。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ========== 控制面板 ==========
        ctrl_group = QGroupBox("控制面板")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self._btn_load = QPushButton("加载音频")
        self._btn_load.clicked.connect(self._on_load)
        ctrl_layout.addWidget(self._btn_load)

        self._btn_record = QPushButton("录制音频")
        self._btn_record.clicked.connect(self._on_record_toggle)
        ctrl_layout.addWidget(self._btn_record)

        self._lbl_source = QLabel("未选择音频源")
        self._lbl_source.setStyleSheet("color: gray;")
        ctrl_layout.addWidget(self._lbl_source)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QLabel("算法:"))
        self._combo_algo = QComboBox()
        self._combo_algo.addItems(["Wiener Filter", "Spectral Subtraction", "U-Net"])
        ctrl_layout.addWidget(self._combo_algo)

        self._btn_denoise = QPushButton("一键降噪")
        self._btn_denoise.setEnabled(False)
        self._btn_denoise.clicked.connect(self._on_denoise)
        self._btn_denoise.setStyleSheet(
            "QPushButton { background-color: #27AE60; color: white; font-weight: bold; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        ctrl_layout.addWidget(self._btn_denoise)

        self._btn_export = QPushButton("导出")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        ctrl_layout.addWidget(self._btn_export)

        main_layout.addWidget(ctrl_group)

        # ========== 录音面板 (默认隐藏) ==========
        self._recorder = AudioRecorder()
        self._recorder.recording_finished.connect(self._on_recording_finished)
        self._recorder.recording_error.connect(self._on_recording_error)
        self._recorder.setVisible(False)
        main_layout.addWidget(self._recorder)

        # ========== 可视化 Tab ==========
        viz_tab = QTabWidget()

        self._waveform_view = WaveformView()
        viz_tab.addTab(self._waveform_view, "时域波形")

        self._spectrogram_view = SpectrogramView()
        viz_tab.addTab(self._spectrogram_view, "频谱图")

        main_layout.addWidget(viz_tab, stretch=3)

        # ========== 评估与诊断 Tab ==========
        bottom_tab = QTabWidget()

        self._metrics_panel = MetricsPanel()
        bottom_tab.addTab(self._metrics_panel, "评估指标")

        self._diagnosis_panel = DiagnosisPanel()
        bottom_tab.addTab(self._diagnosis_panel, "噪声诊断")

        main_layout.addWidget(bottom_tab, stretch=2)

        # ========== 音频回放 ==========
        self._audio_player = AudioPlayer()
        main_layout.addWidget(self._audio_player)

        # ========== 状态栏 ==========
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        main_layout.addWidget(self._progress)

        self._status = QLabel("就绪 — 请加载音频文件或使用麦克风录制")
        main_layout.addWidget(self._status)

    # ---------- 事件处理 ----------

    def _on_load(self) -> None:
        """从文件加载音频，关闭录音面板。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择带噪音频", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.aac);;All Files (*)"
        )
        if not path:
            return
        try:
            # 关闭录音面板
            self._recorder.setVisible(False)
            self._btn_record.setEnabled(True)

            self._waveform, self._sr = load_audio(path)
            self._lbl_source.setText(f"文件: {os.path.basename(path)}")
            self._lbl_source.setStyleSheet("color: #2980B9;")
            self._btn_denoise.setEnabled(True)
            dur = len(self._waveform) / self._sr
            self._status.setText(
                f"已加载: {os.path.basename(path)} ({dur:.1f}s, {self._sr}Hz)"
            )
            self._clear_all_panels()
            logger.info(f"加载音频: {path}")
        except Exception as e:
            self._status.setText(f"加载失败: {e}")

    def _on_record_toggle(self) -> None:
        """切换录音面板的显示/隐藏。"""
        if self._recorder.isVisible():
            self._recorder.setVisible(False)
            self._btn_record.setText("录制音频")
            self._status.setText("就绪")
        else:
            self._recorder.setVisible(True)
            self._btn_record.setText("收起录音")
            self._status.setText("正在使用录音面板 — 点击 [开始录音] 采集音频")

    def _on_recording_finished(self, waveform: np.ndarray, sr: int) -> None:
        """录音完成回调：存储波形，启用降噪，更新面板。

        Args:
            waveform: 录制完成的音频波形.
            sr: 采样率.
        """
        self._waveform = waveform
        self._sr = sr
        dur = len(waveform) / sr
        self._lbl_source.setText(f"录音: {dur:.1f}s")
        self._lbl_source.setStyleSheet("color: #E74C3C;")
        self._btn_denoise.setEnabled(True)
        self._clear_all_panels()

        # 在波形预览中显示原始录音
        self._waveform_view.set_data(waveform, waveform, sr=sr)

        self._status.setText(
            f"录制完成 ({dur:.1f}s, {sr}Hz) — 选择算法后点击 [一键降噪]"
        )
        logger.info(f"录音完成: {dur:.1f}s")

    def _on_recording_error(self, msg: str) -> None:
        """录音出错回调。

        Args:
            msg: 错误消息.
        """
        self._status.setText(f"录音错误: {msg}")
        logger.error(f"录音错误: {msg}")

    def _on_denoise(self) -> None:
        """启动后台降噪线程。"""
        if self._waveform is None:
            return
        self._current_algo = self._combo_algo.currentText()
        # 降噪时隐藏录音面板
        self._recorder.setVisible(False)
        self._btn_record.setText("录制音频")
        self._btn_record.setEnabled(True)

        self._btn_denoise.setEnabled(False)
        self._btn_load.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText(f"正在执行 {self._current_algo} ...")

        self._worker = DenoiseWorker(
            self._waveform, self._sr, self._current_algo,
            model_ckpt=self._model_ckpt,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, val: int) -> None:
        """更新进度条。

        Args:
            val: 进度百分比 (0~100).
        """
        self._progress.setValue(val)

    def _on_finished(
        self, denoised: np.ndarray, sr: int, algorithm: str
    ) -> None:
        """降噪完成：更新全部面板。

        Args:
            denoised: 降噪后波形.
            sr: 采样率.
            algorithm: 使用的算法名称.
        """
        self._denoised = denoised
        self._btn_denoise.setEnabled(True)
        self._btn_load.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._progress.setVisible(False)

        min_len = min(len(self._waveform), len(denoised))
        noisy_seg = self._waveform[:min_len]
        denoised_seg = denoised[:min_len]

        # 1. 波形图
        self._waveform_view.set_data(noisy_seg, denoised_seg, sr=sr)

        # 2. 频谱图
        self._spectrogram_view.set_data(noisy_seg, denoised_seg, sr=sr)

        # 3. 评估指标
        metrics = compute_all_metrics(noisy_seg, denoised_seg, sr)
        self._metrics_panel.set_metrics(metrics)

        # 4. 噪声诊断
        noise_type, noise_label, details, profile = diagnose_noise(
            noisy_seg, sr
        )
        self._diagnosis_panel.set_diagnosis(
            noise_type, noise_label, details, profile
        )

        # 5. 音频播放器
        self._audio_player.set_audio(noisy_seg, denoised_seg, clean=None, sr=sr)

        self._status.setText(
            f"降噪完成 ({algorithm}) | 噪声类型: {noise_label} | 就绪"
        )
        logger.info(f"降噪完成: {algorithm}, noise={noise_type}")

    def _on_error(self, msg: str) -> None:
        """降噪出错回调。

        Args:
            msg: 错误消息.
        """
        self._btn_denoise.setEnabled(True)
        self._btn_load.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText(f"错误: {msg}")
        logger.error(f"降噪失败: {msg}")

    def _on_export(self) -> None:
        """导出降噪后的音频文件。"""
        if self._denoised is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出降噪音频", "denoised.wav", "WAV Files (*.wav)"
        )
        if not path:
            return
        sf.write(path, self._denoised, self._sr)
        self._status.setText(f"已导出: {os.path.basename(path)}")

    def _clear_all_panels(self) -> None:
        """清空所有展示面板。"""
        self._waveform_view.clear()
        self._spectrogram_view.clear()
        self._metrics_panel.clear()
        self._diagnosis_panel.clear()


def main():
    """启动 GUI 应用程序。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # 解析命令行参数 (--ckpt 用于 U-Net)
    ckpt = None
    for i, arg in enumerate(sys.argv):
        if arg == "--ckpt" and i + 1 < len(sys.argv):
            ckpt = sys.argv[i + 1]
            break

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(model_ckpt=ckpt)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
