import datetime
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from automation.runner import BatchAutomationRunner


class RunnerThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal(dict)
    failed_signal = Signal(str)

    def __init__(self, config: Dict[str, Any], root_dir: str):
        super().__init__()
        self.config = config
        self.root_dir = root_dir
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            runner = BatchAutomationRunner(
                config=self.config,
                log=self.log_signal.emit,
                progress=self.progress_signal.emit,
                should_stop=lambda: self._stop,
            )
            result = runner.run(self.root_dir)
            self.finished_signal.emit(result)
        except Exception:
            self.failed_signal.emit(traceback.format_exc())


class SemanticChip(QWidget):
    def __init__(self, label_id: int, label_name: str, on_remove):
        super().__init__()
        self.label_id = label_id
        self.on_remove = on_remove
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.btn = QPushButton(label_name)
        self.btn.clicked.connect(lambda: self.on_remove(self.label_id))
        self.btn.setMinimumHeight(30)
        self.btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.btn.setStyleSheet(
            "QPushButton {"
            "  background: #eef5ff;"
            "  border: 1px solid #8fb2ff;"
            "  border-radius: 12px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:hover { background: #dceaff; }"
        )
        layout.addWidget(self.btn)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)


class SemanticGroupEditor(QWidget):
    def __init__(self, label_options: List[Dict[str, Any]], on_remove_group):
        super().__init__()
        self.label_options = label_options
        self.on_remove_group = on_remove_group
        self._chips: Dict[int, SemanticChip] = {}
        self.selected_semantic_ids: List[int] = []

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { border: 1px solid #d8d8d8; border-radius: 6px; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("组别名称"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：管道相关")
        row_top.addWidget(self.name_edit, 1)
        self.delete_group_btn = QPushButton("删除组别")
        self.delete_group_btn.clicked.connect(lambda: self.on_remove_group(self))
        row_top.addWidget(self.delete_group_btn)
        layout.addLayout(row_top)

        row_chips = QHBoxLayout()
        self.chips_wrap = QWidget()
        self.chips_layout = QGridLayout(self.chips_wrap)
        self.chips_layout.setContentsMargins(0, 0, 0, 0)
        self.chips_layout.setHorizontalSpacing(8)
        self.chips_layout.setVerticalSpacing(8)
        row_chips.addWidget(self.chips_wrap, 1)
        self.add_chip_btn = QPushButton("+")
        self.add_chip_btn.setFixedWidth(30)
        self.add_chip_btn.clicked.connect(self._show_add_menu)
        row_chips.addWidget(self.add_chip_btn)
        layout.addLayout(row_chips)

    def _show_add_menu(self):
        menu = QMenu(self)
        existing = set(self.selected_semantic_ids)
        for item in self.label_options:
            label_id = int(item["id"])
            if label_id in existing:
                continue
            action = QAction(str(item["name"]), menu)
            action.triggered.connect(lambda _=False, sid=label_id: self.add_semantic_id(sid))
            menu.addAction(action)
        if menu.isEmpty():
            action = QAction("没有可添加的语义", menu)
            action.setEnabled(False)
            menu.addAction(action)
        menu.exec(self.add_chip_btn.mapToGlobal(self.add_chip_btn.rect().bottomLeft()))

    def add_semantic_id(self, label_id: int):
        if label_id in self.selected_semantic_ids:
            return
        opt = next((x for x in self.label_options if int(x["id"]) == int(label_id)), None)
        if opt is None:
            return
        chip = SemanticChip(label_id, str(opt["name"]), self.remove_semantic_id)
        self._chips[label_id] = chip
        self.selected_semantic_ids.append(label_id)
        self.selected_semantic_ids.sort()
        self._reflow_chips()

    def remove_semantic_id(self, label_id: int):
        if label_id not in self._chips:
            return
        chip = self._chips.pop(label_id)
        self.selected_semantic_ids = [x for x in self.selected_semantic_ids if x != label_id]
        self.chips_layout.removeWidget(chip)
        chip.deleteLater()
        self._reflow_chips()

    def _reflow_chips(self):
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            # Detach only, do not delete here.
            if item.widget() is not None:
                item.widget().setParent(self.chips_wrap)

        max_cols = 4
        for i, sid in enumerate(self.selected_semantic_ids):
            chip = self._chips.get(sid)
            if chip is None:
                continue
            row = i // max_cols
            col = i % max_cols
            self.chips_layout.addWidget(chip, row, col)

        rows = max(1, (len(self.selected_semantic_ids) + max_cols - 1) // max_cols)
        wrap_min_h = rows * 38 - 8
        self.chips_wrap.setMinimumHeight(max(34, wrap_min_h))
        self.setMinimumHeight(92 + max(34, wrap_min_h))

    def set_name(self, name: str):
        self.name_edit.setText(name)

    def get_name(self) -> str:
        return self.name_edit.text().strip()

    def get_semantic_ids(self) -> List[int]:
        return list(self.selected_semantic_ids)


class BatchAutomationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.semantic_label_options = self._load_semantic_label_options()
        self.semantic_group_editors: List[SemanticGroupEditor] = []
        self.setWindowTitle("PointScript 批量自动化")
        self.resize(1360, 920)
        self.thread = None  # type: Optional[RunnerThread]
        self.current_log_file = None  # type: Optional[str]
        self._build_ui()
        self._wire_events()
        self._set_defaults()
        QTimer.singleShot(0, self._set_initial_splitter_sizes)

    def _load_semantic_label_options(self) -> List[Dict[str, Any]]:
        labels_path = Path(__file__).resolve().parent / "PcotPoints" / "src" / "labeler" / "labels.json"
        if not labels_path.exists():
            return []
        try:
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        options: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            if item.get("is_hidden", False):
                continue
            name_cn = str(item.get("label_cn", "")).strip()
            name_en = str(item.get("label", "")).strip()
            if name_en.lower() == "deprecated":
                continue
            show_name = name_cn or name_en
            if not show_name:
                continue
            options.append({"id": idx, "name": show_name})
        return options

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        top = QHBoxLayout()
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("请选择批处理根目录，例如 .../testdata")
        self.browse_btn = QPushButton("浏览")
        self.scan_btn = QPushButton("扫描样本")
        self.start_btn = QPushButton("开始")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        top.addWidget(QLabel("根目录"))
        top.addWidget(self.root_edit, 1)
        top.addWidget(self.browse_btn)
        top.addWidget(self.scan_btn)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        outer.addLayout(top)

        self.splitter = QSplitter()
        outer.addWidget(self.splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_scroll.setWidget(left)
        self.left_scroll = left_scroll
        self.splitter.addWidget(left_scroll)

        g_global = QGroupBox("全局设置")
        g_global_layout = QFormLayout(g_global)
        self.enable_report = QCheckBox("生成报告")
        self.show_child_windows = QCheckBox("显示子窗口（调试）")
        g_global_layout.addRow(self.enable_report)
        g_global_layout.addRow(self.show_child_windows)
        left_layout.addWidget(g_global)

        g_pcot = QGroupBox("Pcot 参数")
        g_pcot_layout = QVBoxLayout(g_pcot)
        row_down = QHBoxLayout()
        row_down.addWidget(QLabel("降采样晶格边长（米）"))
        self.pcot_downsample_length = QLineEdit()
        row_down.addWidget(self.pcot_downsample_length)
        g_pcot_layout.addLayout(row_down)

        self.enable_semantic_group_export = QCheckBox("按语义分组导出点云")
        g_pcot_layout.addWidget(self.enable_semantic_group_export)

        row_group_top = QHBoxLayout()
        row_group_top.addWidget(QLabel("语义分组"))
        self.add_group_btn = QPushButton("新增组别")
        row_group_top.addWidget(self.add_group_btn)
        row_group_top.addStretch()
        g_pcot_layout.addLayout(row_group_top)

        self.semantic_groups_container = QWidget()
        self.semantic_groups_layout = QVBoxLayout(self.semantic_groups_container)
        self.semantic_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.semantic_groups_layout.setSpacing(8)
        g_pcot_layout.addWidget(self.semantic_groups_container)
        g_pcot_layout.addWidget(QLabel("提示：点击“+”添加语义按钮，再次点击语义按钮可删除。"))
        left_layout.addWidget(g_pcot)

        g_fact = QGroupBox("Fact 参数")
        g_fact_layout = QGridLayout(g_fact)
        g_fact_layout.setHorizontalSpacing(14)
        g_fact_layout.setVerticalSpacing(8)

        self.fact_mode = QComboBox()
        self.fact_mode.addItem("默认", "default")
        self.fact_mode.addItem("手动", "manual")
        self.fact_mode.addItem("自动优化", "auto")
        g_fact_layout.addWidget(QLabel("模式"), 0, 0)
        g_fact_layout.addWidget(self.fact_mode, 0, 1)

        self.fact_merge_rate = QLineEdit()
        self.fact_group_rate = QLineEdit()
        self.fact_tolerance_parallel_cos = QLineEdit()
        self.fact_mini_length = QLineEdit()
        self.fact_radius_ratio_tolerance = QLineEdit()
        self.fact_radius_diff_tolerance = QLineEdit()
        self.fact_patching_threshold_range = QLineEdit()
        self.fact_tolerance_angle = QLineEdit()

        params = [
            ("合并率档位", self.fact_merge_rate),
            ("打组降采样率", self.fact_group_rate),
            ("平行余弦误差", self.fact_tolerance_parallel_cos),
            ("最短补并管高度", self.fact_mini_length),
            ("半径比例容差", self.fact_radius_ratio_tolerance),
            ("半径绝对容差", self.fact_radius_diff_tolerance),
            ("补管范围最大值", self.fact_patching_threshold_range),
            ("平行角度容差", self.fact_tolerance_angle),
        ]
        rows_per_col = (len(params) + 1) // 2
        for idx, (label_text, edit) in enumerate(params):
            col_block = 0 if idx < rows_per_col else 1
            row = 1 + (idx if idx < rows_per_col else (idx - rows_per_col))
            base_col = col_block * 2
            g_fact_layout.addWidget(QLabel(label_text), row, base_col)
            g_fact_layout.addWidget(edit, row, base_col + 1)

        g_fact_layout.setColumnStretch(1, 1)
        g_fact_layout.setColumnStretch(3, 1)
        left_layout.addWidget(g_fact)
        left_layout.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.splitter.addWidget(right)

        self.sample_table = QTableWidget(0, 4)
        self.sample_table.setHorizontalHeaderLabels(["样本目录", "点云文件", "扩展名", "状态"])
        self.sample_table.horizontalHeader().setStretchLastSection(True)
        self.sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.sample_table.setSelectionBehavior(QTableWidget.SelectRows)
        right_layout.addWidget(self.sample_table, 3)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        right_layout.addWidget(self.progress)

        self.summary_label = QLabel("汇总：-")
        right_layout.addWidget(self.summary_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, 2)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(0, False)

    def _wire_events(self):
        self.browse_btn.clicked.connect(self._browse_root)
        self.scan_btn.clicked.connect(self._scan)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.add_group_btn.clicked.connect(self._add_group_editor)

    def _set_initial_splitter_sizes(self):
        left_hint = self.left_scroll.widget().minimumSizeHint().width()
        left_width = max(420, left_hint + 20)
        total = self.splitter.size().width() if self.splitter.size().width() > 0 else self.width()
        right_width = max(200, total - left_width)
        self.splitter.setSizes([left_width, right_width])

    def _set_defaults(self):
        default_root = Path(__file__).resolve().parent / "testdata"
        self.root_edit.setText(str(default_root))
        self.enable_report.setChecked(True)
        self.show_child_windows.setChecked(False)

        self.pcot_downsample_length.setText("0.001")
        self.enable_semantic_group_export.setChecked(True)
        self._clear_group_editors()
        self._add_group_editor("管道相关", [1, 10, 11, 12, 13, 14, 15, 24])
        self._add_group_editor("储罐相关", [3, 17, 18, 19, 20])

        self.fact_mode.setCurrentIndex(2)
        self.fact_merge_rate.setText("30")
        self.fact_group_rate.setText("80")
        self.fact_tolerance_parallel_cos.setText("0.8")
        self.fact_mini_length.setText("0.03")
        self.fact_radius_ratio_tolerance.setText("0.4")
        self.fact_radius_diff_tolerance.setText("0.2")
        self.fact_patching_threshold_range.setText("15")
        self.fact_tolerance_angle.setText("18")

    def _clear_group_editors(self):
        for editor in self.semantic_group_editors:
            self.semantic_groups_layout.removeWidget(editor)
            editor.deleteLater()
        self.semantic_group_editors = []

    def _add_group_editor(self, name: Any = "", semantic_ids: Optional[List[int]] = None):
        if isinstance(name, bool):
            name = ""
        editor = SemanticGroupEditor(
            label_options=self.semantic_label_options,
            on_remove_group=self._remove_group_editor,
        )
        editor.set_name(str(name))
        for sid in (semantic_ids or []):
            editor.add_semantic_id(int(sid))
        self.semantic_group_editors.append(editor)
        self.semantic_groups_layout.addWidget(editor)

    def _remove_group_editor(self, editor: SemanticGroupEditor):
        if editor not in self.semantic_group_editors:
            return
        self.semantic_group_editors.remove(editor)
        self.semantic_groups_layout.removeWidget(editor)
        editor.deleteLater()

    def _browse_root(self):
        folder = QFileDialog.getExistingDirectory(self, "选择根目录")
        if folder:
            self.root_edit.setText(folder)
            self._scan()

    def _append_log(self, msg: str):
        self.log_text.appendPlainText(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        self._write_log_file(msg)
        self._update_table_status_from_log(msg)

    def _update_table_status_from_log(self, msg: str):
        if not hasattr(self, "_current_row_idx"):
            self._current_row_idx = -1

        if "[start]" in msg:
            start_match = re.search(r"\[start\] \(\d+/\d+\) (.*)", msg)
            if start_match:
                pc_path = start_match.group(1).strip()
                for row in range(self.sample_table.rowCount()):
                    item = self.sample_table.item(row, 1)
                    if item and item.text().strip() == pc_path:
                        self._current_row_idx = row
                        self.sample_table.item(row, 3).setText("运行中（启动）")
                        break
                return

        if getattr(self, "_current_row_idx", -1) >= 0:
            if "[done]" in msg:
                self.sample_table.item(self._current_row_idx, 3).setText("已完成")
                self.sample_table.viewport().update()
            elif "[error]" in msg:
                self.sample_table.item(self._current_row_idx, 3).setText("失败")
                self.sample_table.viewport().update()
            elif "[pcot]" in msg or "[fact]" in msg:
                self.sample_table.item(self._current_row_idx, 3).setText(msg.strip())
                self.sample_table.viewport().update()

    def _prepare_log_file(self):
        root = self.root_edit.text().strip()
        base = (Path(root) / "automation_logs") if root else (Path(__file__).resolve().parent / "logs")
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = str(base / f"batch_automation_{ts}.log")
        latest = base / "latest.log"
        with open(self.current_log_file, "w", encoding="utf-8") as f:
            f.write("")
        with open(latest, "w", encoding="utf-8") as f:
            f.write(self.current_log_file + "\n")

    def _write_log_file(self, msg: str):
        if not self.current_log_file:
            return
        line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _scan(self):
        root = self.root_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "提示", "请先选择根目录")
            return

        self.sample_table.setRowCount(0)
        self._current_row_idx = -1
        try:
            runner = BatchAutomationRunner(
                config=self._collect_config(),
                log=lambda _: None,
                progress=lambda *_: None,
                should_stop=lambda: False,
            )
            tasks = runner.discover_tasks(root)
            self.sample_table.setRowCount(0)
            for i, task in enumerate(tasks):
                self.sample_table.insertRow(i)
                self.sample_table.setItem(i, 0, QTableWidgetItem(task.sample_dir))
                self.sample_table.setItem(i, 1, QTableWidgetItem(task.pointcloud_path))
                self.sample_table.setItem(i, 2, QTableWidgetItem(task.extension))
                self.sample_table.setItem(i, 3, QTableWidgetItem("待处理"))
            self.summary_label.setText(f"汇总：发现 {len(tasks)} 个样本")
            self._append_log(f"[扫描] 发现 {len(tasks)} 个样本")
        except Exception as e:
            QMessageBox.critical(self, "扫描失败", str(e))

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.scan_btn.setEnabled(not running)
        self.browse_btn.setEnabled(not running)

    def _start(self):
        root = self.root_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "提示", "请先选择根目录")
            return
        if not os.path.isdir(root):
            QMessageBox.warning(self, "提示", f"不是有效目录：{root}")
            return

        config = self._collect_config()
        self._prepare_log_file()
        self._append_log(f"[日志] 文件：{self.current_log_file}")
        self._append_log("[运行] 开始")
        self._set_running(True)
        self.progress.setValue(0)
        self.thread = RunnerThread(config=config, root_dir=root)
        self.thread.log_signal.connect(self._append_log)
        self.thread.progress_signal.connect(self._on_progress)
        self.thread.finished_signal.connect(self._on_finished)
        self.thread.failed_signal.connect(self._on_failed)
        self.thread.start()

    def _stop(self):
        if self.thread is not None:
            self.thread.stop()
            self._append_log("[运行] 正在停止...")

    def _on_progress(self, current: int, total: int):
        if total <= 0:
            self.progress.setValue(0)
            return
        value = int((current / total) * 100)
        self.progress.setValue(max(0, min(100, value)))

    def _on_finished(self, summary: Dict[str, Any]):
        self._set_running(False)
        self.progress.setValue(100)
        self.summary_label.setText(
            f"汇总：总数={summary['total']} 成功={summary['success']} 失败={summary['failed']}"
        )
        self._append_log("[运行] 完成")
        self._append_log(f"[汇总] JSON：{Path(self.root_edit.text().strip()) / 'batch_automation_summary.json'}")
        self._mark_table(summary.get("results", []))

    def _mark_table(self, results: List[Dict[str, Any]]):
        status_by_path = {r.get("pointcloud"): r.get("status", "unknown") for r in results}
        for row in range(self.sample_table.rowCount()):
            p = self.sample_table.item(row, 1).text()
            status = status_by_path.get(p, "pending")
            self.sample_table.setItem(row, 3, QTableWidgetItem(self._status_text_cn(status)))

    def _status_text_cn(self, status: str) -> str:
        mapping = {
            "success": "成功",
            "failed": "失败",
            "pending": "待处理",
            "unknown": "未知",
        }
        return mapping.get(status, status)

    def _on_failed(self, detail: str):
        self._set_running(False)
        self._append_log("[运行] 致命错误")
        self._append_log(detail)
        QMessageBox.critical(self, "运行失败", detail)

    def _collect_config(self) -> Dict[str, Any]:
        groups: List[Dict[str, Any]] = []
        for editor in self.semantic_group_editors:
            name = editor.get_name()
            semantic_ids = editor.get_semantic_ids()
            if not name or not semantic_ids:
                continue
            groups.append({"name": name, "semantic_ids": semantic_ids})

        cfg = {
            "enable_report": self.enable_report.isChecked(),
            "show_child_windows": self.show_child_windows.isChecked(),
            "pcot": {
                "downsample_length": float(self.pcot_downsample_length.text().strip() or "0.001"),
                "semantic_group_export_enabled": self.enable_semantic_group_export.isChecked(),
                "semantic_groups": groups,
            },
            "fact": {
                "mode": (self.fact_mode.currentData() or "default"),
                "manual": {
                    "merge_rate": int(self.fact_merge_rate.text().strip() or "30"),
                    "group_downsample_rate": int(self.fact_group_rate.text().strip() or "80"),
                    "auto_connect": {
                        "tolerance_parallel_cos": float(self.fact_tolerance_parallel_cos.text().strip() or "0.8"),
                        "mini_length": float(self.fact_mini_length.text().strip() or "0.03"),
                        "radius_ratio_tolerance": float(self.fact_radius_ratio_tolerance.text().strip() or "0.4"),
                        "radius_diff_tolerance": float(self.fact_radius_diff_tolerance.text().strip() or "0.2"),
                        "patching_threshold_range": float(self.fact_patching_threshold_range.text().strip() or "15"),
                        "tolerance_angle": float(self.fact_tolerance_angle.text().strip() or "18"),
                    },
                },
            },
        }
        return cfg

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            reply = QMessageBox.question(self, "退出", "任务正在运行，是否停止并退出？")
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.thread.stop()
            self.thread.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = BatchAutomationWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
