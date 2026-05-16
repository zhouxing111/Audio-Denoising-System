"""
ui/main_window.py — 主窗口 (双模式 + 多方法对比)

两种工作模式:
  降噪模式: Wiener Filter / Spectral Subtraction / U-Net
  修复模式: Spline Interpolation / Spectral Inpainting / U-Net Inpainting

每种模式支持"单选一种方法"或"对比全部方法"。
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
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_audio
from data.noise_diagnosis import diagnose_noise
from evaluation.metrics import compute_all_metrics

from ui.widgets.metrics_panel import MetricsPanel
from ui.widgets.waveform_view import WaveformView
from ui.widgets.spectrogram_view import SpectrogramView
from ui.widgets.diagnosis_panel import DiagnosisPanel
from ui.audio_player import AudioPlayer
from ui.audio_recorder import AudioRecorder

logger = logging.getLogger(__name__)

# 模式 → 方法列表
DENOISING_METHODS = ["Wiener Filter", "Spectral Subtraction", "U-Net", "Hybrid (U-Net + Wiener)"]
INPAINTING_METHODS = ["Spline Interpolation", "Spectral Inpainting", "U-Net Inpainting"]


class BatchWorker(QThread):
    """后台线程批量执行降噪/修复，支持对比模式。

    串行执行每个方法，通过 progress 信号报告进度 (0~100)。
    """

    finished = Signal(dict, int)  # {method_name: waveform}, sr
    error = Signal(str)
    progress = Signal(int)

    def __init__(
        self, waveform: np.ndarray, sr: int, methods: list[str],
        mode: str, model_ckpt: str | None = None,
    ):
        """初始化批量工作线程。

        Args:
            waveform: 带噪/损坏音频波形.
            sr: 采样率.
            methods: 要执行的方法名列表.
            mode: "denoising" 或 "inpainting".
            model_ckpt: U-Net checkpoint 路径.
        """
        super().__init__()
        self.waveform = waveform
        self.sr = sr
        self.methods = methods
        self.mode = mode
        self.model_ckpt = model_ckpt

    def run(self) -> None:
        """线程主函数：逐个执行方法。"""
        results: dict[str, np.ndarray] = {}
        n = len(self.methods)
        try:
            for i, method in enumerate(self.methods):
                self.progress.emit(int(i / n * 100))
                if self.mode == "denoising":
                    results[method] = self._run_denoiser(method)
                else:
                    results[method] = self._run_inpainter(method)
                self.progress.emit(int((i + 1) / n * 100))
            self.finished.emit(results, self.sr)
        except Exception as e:
            self.error.emit(str(e))

    def _run_denoiser(self, method: str) -> np.ndarray:
        """执行单个降噪方法。

        Args:
            method: 方法名.

        Returns:
            降噪后波形.
        """
        if method == "Wiener Filter":
            from models.wiener import WienerFilter
            return WienerFilter().denoise_audio(self.waveform, self.sr).astype(np.float32)
        elif method == "Spectral Subtraction":
            from models.spectral_sub import SpectralSubtraction
            return SpectralSubtraction().denoise_audio(self.waveform, self.sr).astype(np.float32)
        elif method == "Hybrid (U-Net + Wiener)":
            from models.hybrid import HybridDenoiser
            h = HybridDenoiser()
            return h.denoise_audio(self.waveform, self.sr, model_ckpt=self.model_ckpt).astype(np.float32)
        else:  # U-Net
            return self._run_unet_denoise()

    def _run_inpainter(self, method: str) -> np.ndarray:
        """执行单个修复方法。

        Args:
            method: 方法名.

        Returns:
            修复后波形.
        """
        from models.audio_inpainter import AudioInpainter
        inpainter = AudioInpainter()
        if method == "Spline Interpolation":
            return inpainter.inpaint(self.waveform, self.sr, method="spline")
        elif method == "Spectral Inpainting":
            return inpainter.inpaint(self.waveform, self.sr, method="spectral")
        else:  # U-Net Inpainting
            return inpainter.inpaint(self.waveform, self.sr, method="unet", model_ckpt=self.model_ckpt)

    def _run_unet_denoise(self) -> np.ndarray:
        """U-Net 降噪推理。

        Returns:
            降噪后波形.
        """
        import torch
        from models.unet import UNetDenoiser
        if not self.model_ckpt or not Path(self.model_ckpt).exists():
            raise FileNotFoundError(f"U-Net checkpoint 未找到: {self.model_ckpt}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetDenoiser(n_fft=512, hop_length=256).to(device)
        ckpt = torch.load(self.model_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model.denoise_audio(self.waveform, self.sr).astype(np.float32)


class MainWindow(QMainWindow):
    """智能音频降噪系统主窗口。

    双模式 + 多方法对比:
      [控制面板: 加载/录制 | 模式 | 对比 | 方法 | 执行/导出]
      [Tab: 波形 | 频谱]
      [Tab: 评估指标 | 噪声诊断]
      [音频回放]
    """

    def __init__(self, model_ckpt: str | None = None):
        super().__init__()
        self.setWindowTitle("智能音频降噪系统")
        self.setMinimumSize(1100, 800)
        self._waveform = None
        self._results: dict[str, np.ndarray] = {}
        self._sr = None
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

        # 模式切换
        ctrl_layout.addWidget(QLabel("模式:"))
        self._radio_denoise = QRadioButton("降噪")
        self._radio_denoise.setChecked(True)
        self._radio_denoise.toggled.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self._radio_denoise)

        self._radio_inpaint = QRadioButton("修复")
        self._radio_inpaint.toggled.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self._radio_inpaint)

        # 对比开关
        self._chk_compare = QCheckBox("对比全部方法")
        self._chk_compare.toggled.connect(self._on_compare_toggled)
        ctrl_layout.addWidget(self._chk_compare)

        # 方法选择
        ctrl_layout.addWidget(QLabel("方法:"))
        self._combo_method = QComboBox()
        self._combo_method.addItems(DENOISING_METHODS)
        ctrl_layout.addWidget(self._combo_method)

        # 执行 / 导出
        self._btn_execute = QPushButton("▶ 执行")
        self._btn_execute.setEnabled(False)
        self._btn_execute.clicked.connect(self._on_execute)
        self._btn_execute.setStyleSheet(
            "QPushButton { background-color: #27AE60; color: white; font-weight: bold; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        ctrl_layout.addWidget(self._btn_execute)

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

        # ========== 结果方法切换器 (对比模式可见) ==========
        result_row = QHBoxLayout()
        self._lbl_result_selector = QLabel("对比结果 — 查看方法:")
        self._lbl_result_selector.setVisible(False)
        result_row.addWidget(self._lbl_result_selector)
        self._combo_result = QComboBox()
        self._combo_result.setVisible(False)
        self._combo_result.currentIndexChanged.connect(self._on_result_method_changed)
        result_row.addWidget(self._combo_result)
        result_row.addStretch()
        main_layout.addLayout(result_row)

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
        ckpt_info = f" | U-Net: {self._model_ckpt}" if self._model_ckpt else ""
        self._status = QLabel(f"就绪 — 请加载音频文件或使用麦克风录制{ckpt_info}")
        main_layout.addWidget(self._status)

    # ---------- 模式/对比切换 ----------

    def _on_mode_changed(self) -> None:
        """模式切换：更新方法下拉内容。"""
        methods = INPAINTING_METHODS if self._radio_inpaint.isChecked() else DENOISING_METHODS
        self._combo_method.clear()
        self._combo_method.addItems(methods)

    def _on_compare_toggled(self, checked: bool) -> None:
        """对比开关：勾选时禁用方法下拉。"""
        self._combo_method.setEnabled(not checked)

    def _get_selected_methods(self) -> list[str]:
        """获取当前应执行的方法列表。

        Returns:
            方法名列表.
        """
        if self._chk_compare.isChecked():
            return INPAINTING_METHODS if self._radio_inpaint.isChecked() else DENOISING_METHODS
        return [self._combo_method.currentText()]

    # ---------- 文件/录制 ----------

    def _on_load(self) -> None:
        """加载音频文件。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.aac);;All Files (*)",
        )
        if not path:
            return
        try:
            self._recorder.setVisible(False)
            self._btn_record.setEnabled(True)
            self._waveform, self._sr = load_audio(path)
            self._lbl_source.setText(f"文件: {os.path.basename(path)}")
            self._lbl_source.setStyleSheet("color: #2980B9;")
            self._btn_execute.setEnabled(True)
            self._results = {}
            dur = len(self._waveform) / self._sr
            self._status.setText(f"已加载: {os.path.basename(path)} ({dur:.1f}s, {self._sr}Hz)")
            self._clear_all_panels()
        except Exception as e:
            self._status.setText(f"加载失败: {e}")

    def _on_record_toggle(self) -> None:
        """切换录音面板。"""
        if self._recorder.isVisible():
            self._recorder.setVisible(False)
            self._btn_record.setText("录制音频")
        else:
            self._recorder.setVisible(True)
            self._btn_record.setText("收起录音")

    def _on_recording_finished(self, waveform: np.ndarray, sr: int) -> None:
        """录音完成。"""
        self._waveform = waveform
        self._sr = sr
        dur = len(waveform) / sr
        self._lbl_source.setText(f"录音: {dur:.1f}s")
        self._lbl_source.setStyleSheet("color: #E74C3C;")
        self._btn_execute.setEnabled(True)
        self._results = {}
        self._clear_all_panels()
        self._waveform_view.set_data(waveform, waveform, sr=sr)
        self._status.setText(f"录制完成 ({dur:.1f}s) — 点击 ▶ 执行")

    def _on_recording_error(self, msg: str) -> None:
        """录音出错。"""
        self._status.setText(f"录音错误: {msg}")

    # ---------- 执行 ----------

    def _on_execute(self) -> None:
        """启动后台批量执行。"""
        if self._waveform is None:
            return
        methods = self._get_selected_methods()
        mode = "inpainting" if self._radio_inpaint.isChecked() else "denoising"

        self._recorder.setVisible(False)
        self._btn_record.setText("录制音频")
        self._btn_record.setEnabled(True)
        self._btn_execute.setEnabled(False)
        self._btn_load.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText(f"正在执行 {mode} ({', '.join(methods)}) ...")

        self._worker = BatchWorker(
            self._waveform, self._sr, methods, mode, model_ckpt=self._model_ckpt,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, val: int) -> None:
        """更新进度条。"""
        self._progress.setValue(val)

    def _on_batch_finished(self, results: dict[str, np.ndarray], sr: int) -> None:
        """批量执行完成：更新全部面板。

        对比模式下自动注入"Noisy (Original)"作为基准行。

        Args:
            results: {method_name: waveform}.
            sr: 采样率.
        """
        self._results = results
        self._btn_execute.setEnabled(True)
        self._btn_load.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._progress.setVisible(False)

        method_names = list(results.keys())
        min_len = min(len(self._waveform), min(len(w) for w in results.values()))
        self._noisy_seg = self._waveform[:min_len]
        self._result_sr = sr
        is_compare = len(method_names) > 1

        # 对比模式下注入带噪原始音频作为基准行
        if is_compare:
            results["Noisy (Original)"] = self._noisy_seg.copy()
            all_names = ["Noisy (Original)"] + method_names
        else:
            all_names = method_names

        self._method_names = all_names
        self._results = results

        # 预计算所有指标
        self._all_metrics = {}
        for name in all_names:
            m = compute_all_metrics(self._noisy_seg, results[name][:min_len], sr)
            self._all_metrics[name] = {k: v for k, v in m.items() if not np.isnan(v)}

        # 结果方法切换器
        self._lbl_result_selector.setVisible(is_compare)
        self._combo_result.setVisible(is_compare)
        if is_compare:
            self._combo_result.blockSignals(True)
            self._combo_result.clear()
            self._combo_result.addItems(all_names)
            self._combo_result.blockSignals(False)

        # 指标表格
        if is_compare:
            self._metrics_panel.set_comparison(self._all_metrics)
        else:
            self._metrics_panel.set_metrics(self._all_metrics[method_names[0]])

        # 噪声诊断
        if not is_compare and self._radio_denoise.isChecked():
            noise_type, noise_label, details, profile = diagnose_noise(self._noisy_seg, sr)
            self._diagnosis_panel.set_diagnosis(noise_type, noise_label, details, profile)

        # 默认显示第一个方法
        self._show_result_method(all_names[0])
        self._status.setText(f"执行完成 ({len(method_names)} 个方法) | 就绪")

    def _on_result_method_changed(self, idx: int) -> None:
        """对比模式下切换查看的方法。"""
        if idx < 0 or not hasattr(self, '_method_names'):
            return
        self._show_result_method(self._method_names[idx])

    def _show_result_method(self, name: str) -> None:
        """将波形/频谱/播放器切换到指定方法。

        波形图只展示当前选中方法 + 原始带噪信号，不混杂其他方法。

        Args:
            name: 方法名.
        """
        wf = self._results[name][:len(self._noisy_seg)]
        sr = self._result_sr
        self._waveform_view.set_data(self._noisy_seg, wf, sr=sr)
        self._spectrogram_view.set_data(self._noisy_seg, wf, sr=sr)
        self._audio_player.set_audio(self._noisy_seg, wf, clean=None, sr=sr)

    def _on_error(self, msg: str) -> None:
        """执行出错。"""
        self._btn_execute.setEnabled(True)
        self._btn_load.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText(f"错误: {msg}")

    def _on_export(self) -> None:
        """导出结果音频。"""
        if not self._results:
            return
        for name, wf in self._results.items():
            safe_name = name.lower().replace(" ", "_")
            path, _ = QFileDialog.getSaveFileName(
                self, f"导出 - {name}", f"{safe_name}.wav", "WAV Files (*.wav)",
            )
            if not path:
                continue
            sf.write(path, wf, self._sr)
        self._status.setText("导出完成")

    def _clear_all_panels(self) -> None:
        """清空所有展示面板。"""
        self._waveform_view.clear()
        self._spectrogram_view.clear()
        self._metrics_panel.clear()
        self._diagnosis_panel.clear()


def main():
    """启动 GUI。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
