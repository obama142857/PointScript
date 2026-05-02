import json
import os
import re
import sys
import time
import traceback
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

import numpy as np
from automation.project_dirs import load_project_dirs


SUPPORTED_EXTS = (".npy", ".e57", ".pcd")
PRIORITY = {".npy": 0, ".e57": 1, ".pcd": 2}
FACT_DONE_MARKER = ".fact_automation_done.json"


@dataclass
class SampleTask:
    sample_dir: str
    pointcloud_path: str
    extension: str
    existing_label_path: Optional[str] = None

    @property
    def cloud_stem(self) -> str:
        return Path(self.pointcloud_path).stem

    @property
    def fact_project_dir(self) -> str:
        return str(Path(self.sample_dir) / f"{self.cloud_stem}_fact")

    @property
    def pcot_label_output(self) -> str:
        return str(Path(self.sample_dir) / f"{self.cloud_stem}_result.npy")


class BatchAutomationRunner:
    def __init__(
        self,
        config: dict,
        log: Callable[[str], None],
        progress: Callable[[int, int], None],
        should_stop: Callable[[], bool],
    ):
        self.config = config
        self.log = log
        self.progress = progress
        self.should_stop = should_stop
        self._root = Path(__file__).resolve().parent.parent
        self._project_dirs = load_project_dirs(self._root)

    def discover_tasks(self, root_dir: str) -> List[SampleTask]:
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Root folder not found: {root_dir}")

        tasks: List[SampleTask] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() in {"automation_logs", "__pycache__"}:
                continue
            done_marker = child / FACT_DONE_MARKER
            if done_marker.exists() and done_marker.is_file():
                self.log(f"[skip] {child}: found done marker {done_marker.name}")
                continue
            cloud = self._pick_pointcloud(child)
            if cloud is None:
                self.log(f"[skip] {child}: no point cloud found ({SUPPORTED_EXTS})")
                continue
            tasks.append(
                SampleTask(
                    sample_dir=str(child),
                    pointcloud_path=str(cloud),
                    extension=cloud.suffix.lower(),
                    existing_label_path=self._pick_existing_label(child, cloud),
                )
            )
        return tasks

    def run(self, root_dir: str) -> Dict[str, Any]:
        self.log(
            "[config] project dirs: "
            f"pcot={self._project_dirs['pcot_dir_name']}, "
            f"fact={self._project_dirs['fact_dir_name']}"
        )
        tasks = self.discover_tasks(root_dir)
        total = len(tasks)
        self.progress(0, total)
        if total == 0:
            return {"total": 0, "success": 0, "failed": 0, "results": []}

        results = []
        success = 0
        failed = 0

        for idx, task in enumerate(tasks, start=1):
            if self.should_stop():
                self.log("[stop] user requested stop")
                break
            self.progress(idx - 1, total)
            self.log(f"[start] ({idx}/{total}) {task.pointcloud_path}")
            started = time.time()
            try:
                one_result = self._run_one(task)
                one_result["elapsed_sec"] = round(time.time() - started, 2)
                one_result["status"] = "success"
                results.append(one_result)
                success += 1
                self.log(f"[done] {task.pointcloud_path}")
            except Exception as exc:
                failed += 1
                detail = traceback.format_exc()
                self.log(f"[error] {task.pointcloud_path}: {exc}")
                results.append(
                    {
                        "sample_dir": task.sample_dir,
                        "pointcloud": task.pointcloud_path,
                        "status": "failed",
                        "error": str(exc),
                        "traceback": detail,
                        "elapsed_sec": round(time.time() - started, 2),
                    }
                )

        self.progress(min(len(results), total), total)
        summary = {
            "total": total,
            "success": success,
            "failed": failed,
            "results": results,
        }
        self._write_summary(root_dir, summary)
        return summary

    def _run_one(self, task: SampleTask) -> Dict[str, Any]:
        if task.existing_label_path:
            self.log(f"  [pcot] skip, found existing label: {task.existing_label_path}")
            pcot_ctx = {"label_output": task.existing_label_path, "points": None, "colors": None}
        else:
            pcot_ctx = self._run_pcot(task)

        semantic_group_outputs: List[Dict[str, Any]] = []
        if self.config.get("pcot", {}).get("semantic_group_export_enabled", False):
            if self.should_stop():
                raise RuntimeError("Stopped by user")
            self.log("  [pcot] export semantic groups")
            semantic_group_outputs = self._export_semantic_group_pointclouds(task, pcot_ctx)

        fact_pointcloud_path = self._resolve_fact_pointcloud_path(task)
        fact_ctx = self._run_fact(task, pcot_ctx, fact_pointcloud_path)
        self._write_fact_done_marker(task, pcot_ctx, fact_ctx)
        return {
            "sample_dir": task.sample_dir,
            "pointcloud": task.pointcloud_path,
            "label_output": pcot_ctx["label_output"],
            "semantic_group_outputs": semantic_group_outputs,
            "fact_project_dir": fact_ctx["project_dir"],
            "report_dir": fact_ctx.get("report_dir"),
            "grouped_obj_path": fact_ctx.get("grouped_obj_path"),
        }

    def _write_fact_done_marker(self, task: SampleTask, pcot_ctx: Dict[str, Any], fact_ctx: Dict[str, Any]) -> None:
        marker_path = Path(task.sample_dir) / FACT_DONE_MARKER
        payload = {
            "status": "done",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pointcloud": task.pointcloud_path,
            "label_output": pcot_ctx.get("label_output"),
            "fact_project_dir": fact_ctx.get("project_dir"),
            "report_dir": fact_ctx.get("report_dir"),
        }
        marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"  [fact] wrote done marker: {marker_path}")

    def _run_pcot(self, task: SampleTask) -> Dict[str, Any]:
        self.log(f"  [pcot] headless pipeline for {task.cloud_stem}")

        cwd_prev = os.getcwd()
        pcot_root = self._pcot_root()
        os.chdir(str(pcot_root))
        self._prepend_syspath(str(pcot_root))

        try:
            self._reset_python_modules_for_pcot()
            self._apply_pcot_runtime_config(pcot_root)
            from src.labeler.labels import Labels  # type: ignore
            from src.labeler.action_controller import ActionController  # type: ignore

            # MainUI normally initializes label metadata before any controller action.
            Labels.random_rgb_colors(3000, seed=0, reserve_black=True)
            Labels.init_dict_pack(json_path=str(pcot_root / "src" / "labeler" / "labels.json"))

            ctrl = ActionController()
            ctrl.load_data(task.pointcloud_path)
            self.log(f"  [pcot] loaded points: {ctrl.points.shape[0]}")
            if Path(task.pointcloud_path).suffix.lower() == ".e57":
                npy_pointcloud_path = str(Path(task.pointcloud_path).with_suffix(".npy"))
                self.log(f"  [pcot] cache e57 as npy -> {npy_pointcloud_path}")
                ctrl.save_data(npy_pointcloud_path)

            downsample_length = float(self.config.get("pcot", {}).get("downsample_length", 0.0) or 0.0)
            if downsample_length > 0:
                if self.should_stop():
                    raise RuntimeError("Stopped by user")
                self.log(f"  [pcot] down_sample length={downsample_length}")
                ctrl.down_sample(downsample_length)
                self.log(f"  [pcot] points after down_sample: {ctrl.points.shape[0]}")

            if self.should_stop():
                raise RuntimeError("Stopped by user")
            self.log("  [pcot] gnd_extract")
            ctrl.gnd_extract()
            
            orig_seg_ins = ctrl.seg_ins
            def safe_seg_ins():
                try:
                    orig_seg_ins()
                except ValueError as e:
                    if "0 sample(s)" in str(e):
                        self.log("  [pcot] skip seg_ins: nothing to cluster")
                    else:
                        raise e
            
            steps = [
                ("seg_scene", lambda: ctrl.seg_scene([])),
                ("seg_pipe", ctrl.seg_pipe),
                ("seg_tank", ctrl.seg_tank),
                ("seg_steel", ctrl.seg_steel),
                ("axis_pred", ctrl.axis_pred),
                ("seg_ins", safe_seg_ins),
            ]
            for name, fn in steps:
                if self.should_stop():
                    raise RuntimeError("Stopped by user")
                self.log(f"  [pcot] {name}")
                fn()

            # FactPoints requires labels to align with original point count.
            # If we downsampled for segmentation, restore labels back to the
            # original cloud size before saving.
            if getattr(ctrl, "down_sample_inverse", None) is not None:
                self.log("  [pcot] up_sample labels to original point count")
                ctrl.up_sample()

            label_output = task.pcot_label_output
            self.log(f"  [pcot] save labels -> {label_output}")
            ctrl.save_labels(label_output)
            return {
                "label_output": label_output,
                "points": np.array(ctrl.points, copy=True),
                "colors": np.array(ctrl.colors, copy=True),
            }
        finally:
            self._remove_syspath(str(pcot_root))
            os.chdir(cwd_prev)

    def _export_semantic_group_pointclouds(self, task: SampleTask, pcot_ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        groups = self.config.get("pcot", {}).get("semantic_groups", [])
        if not groups:
            self.log("  [pcot] semantic groups empty, skip export")
            return []

        labels_all, semantic_labels = self._load_labels_with_semantic(pcot_ctx["label_output"])
        points = pcot_ctx.get("points")
        colors = pcot_ctx.get("colors")
        if points is not None and colors is not None and len(points) == len(semantic_labels):
            xyzrgb = np.hstack((points[:, :3], colors[:, :3]))
        else:
            xyzrgb = self._load_pointcloud_xyzrgb(task.pointcloud_path)
            if len(xyzrgb) != len(semantic_labels):
                raise RuntimeError(
                    "Semantic group export failed: label count does not match point cloud count. "
                    "Please regenerate labels in this run (avoid using mismatched existing labels)."
                )

        export_dir = Path(task.sample_dir) / f"{task.cloud_stem}_semantic_groups"
        export_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"  [pcot] semantic groups export dir -> {export_dir}")

        outputs: List[Dict[str, Any]] = []
        for group in groups:
            name = str(group.get("name", "")).strip()
            semantic_ids = [int(x) for x in group.get("semantic_ids", []) if str(x).strip()]
            if not name or not semantic_ids:
                continue

            mask = np.isin(semantic_labels, semantic_ids)
            out_data = xyzrgb[mask]
            safe_name = self._sanitize_filename_component(name)
            out_path = export_dir / f"{task.cloud_stem}_{safe_name}.npy"
            out_label_path = export_dir / f"{task.cloud_stem}_{safe_name}_result.npy"
            np.save(str(out_path), out_data)
            np.save(str(out_label_path), labels_all[:, mask].astype(np.int32))
            self.log(
                f"  [pcot] semantic group '{name}' -> {out_path}, {out_label_path} "
                f"(points={out_data.shape[0]}, ids={semantic_ids})"
            )
            outputs.append(
                {
                    "name": name,
                    "semantic_ids": semantic_ids,
                    "output_path": str(out_path),
                    "label_output_path": str(out_label_path),
                    "points": int(out_data.shape[0]),
                }
            )
        return outputs

    def _load_labels_with_semantic(self, label_path: str) -> tuple[np.ndarray, np.ndarray]:
        labels = np.load(label_path)
        labels = np.asarray(labels)
        if labels.ndim != 2 or labels.shape[0] < 1:
            raise RuntimeError(f"Invalid label file format: {label_path}")
        semantic = labels[0].astype(np.int32)
        semantic[semantic == -1] = 0
        if labels.shape[0] == 1:
            labels = np.vstack([labels, np.zeros_like(labels)])
        return labels.astype(np.int32), semantic

    def _load_pointcloud_xyzrgb(self, pointcloud_path: str) -> np.ndarray:
        ext = Path(pointcloud_path).suffix.lower()
        if ext == ".npy":
            arr = np.asarray(np.load(pointcloud_path))
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise RuntimeError(f"Invalid npy point cloud format: {pointcloud_path}")
            xyz = arr[:, :3]
            if arr.shape[1] >= 6:
                rgb = arr[:, 3:6]
            else:
                rgb = np.zeros((arr.shape[0], 3), dtype=arr.dtype)
            return np.hstack((xyz, rgb))

        # Reuse Pcot loader for e57/pcd compatibility.
        cwd_prev = os.getcwd()
        pcot_root = self._pcot_root()
        os.chdir(str(pcot_root))
        self._prepend_syspath(str(pcot_root))
        try:
            self._reset_python_modules_for_pcot()
            from src.labeler.file_utils import load_data  # type: ignore

            points, colors, _, _ = load_data(pointcloud_path)
            points = np.asarray(points)
            colors = np.asarray(colors)
            if colors.ndim != 2 or colors.shape[1] < 3:
                colors = np.zeros((points.shape[0], 3), dtype=points.dtype)
            return np.hstack((points[:, :3], colors[:, :3]))
        finally:
            self._remove_syspath(str(pcot_root))
            os.chdir(cwd_prev)

    def _sanitize_filename_component(self, name: str) -> str:
        safe = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
        return safe or "group"

    def _run_fact(self, task: SampleTask, pcot_ctx: Dict[str, Any], pointcloud_path: str) -> Dict[str, Any]:
        from PySide6.QtCore import QCoreApplication

        self.log(f"  [fact] boot UI for {task.cloud_stem}")
        cwd_prev = os.getcwd()
        fact_root = self._fact_root()
        os.chdir(str(fact_root))
        self._prepend_syspath(str(fact_root))

        try:
            self._reset_python_modules_for_fact()
            # FactPoints main() would create this folder, but we instantiate MainUI directly.
            os.makedirs("dev_tmp", exist_ok=True)
            dialog_guard = self._mute_qt_dialogs()
            dialog_guard.__enter__()
            import main as fact_main  # type: ignore
            from src.geometry.database import PointCloudDB, PrimitiveDB  # type: ignore
            from src.pipes_fitting.auto_connect_config.auto_connect_config import (
                AutoConnectConfigAdapter,
                AutoConnectConfigOptimizer,
            )
            import config as fact_cfg  # type: ignore

            ui = fact_main.MainUI()
            if not self.config.get("show_child_windows", False):
                ui.hide()

            project_dir = Path(task.fact_project_dir)
            project_dir.mkdir(parents=True, exist_ok=True)
            project_dir_str = str(project_dir).replace('\\', '/')
            self._prepare_fact_project(ui, project_dir_str)
            self._apply_fact_runtime_config(fact_cfg, AutoConnectConfigAdapter)

            # import pointcloud then label
            ui.file_controller.load_file_by_path(pointcloud_path, proj_path=project_dir_str)
            ui.file_controller.load_file_by_path(pcot_ctx["label_output"], proj_path=project_dir_str)

            # create matching db scenes (mimic unified import behavior)
            db_file = project_dir / "data.db"
            db = ui.db_controller.getdb(str(db_file).replace('\\', '/'))
            db.create_scene(Path(pointcloud_path).name, path=pointcloud_path, scene_type="pointcloud")
            db.create_scene(Path(pcot_ctx["label_output"]).name, path=pcot_ctx["label_output"], scene_type="pointcloud")

            # choose target pointcloud with label + path
            pointcloud = self._pick_fact_pointcloud(ui, project_dir_str)
            if pointcloud is None:
                raise RuntimeError("FactPoints: no labeled point cloud found after import")

            # fit
            self.log("  [fact] fit")
            ui.fit_controller.proj_path = project_dir_str
            ui.fit_controller.details = {"pointcloud": pointcloud.get_path()}
            fit_worker = ui.fit_controller.worker = ui.fit_controller.worker or None
            ui.fit_controller.is_running = False
            ui.fit_controller.result_name = f"{pointcloud.name}.fitting_result"
            from src.controller.base import workerThread  # type: ignore
            from src.pipes_fitting.init_fitting import init_fitting  # type: ignore
            wt = workerThread(target=init_fitting, args=(pointcloud.get_points(), pointcloud.get_label(), pointcloud.get_path()))
            ui.fit_controller.worker = wt
            done = {"ok": False, "exc": None}

            def on_fit(arg):
                try:
                    ui.fit_controller.post_fitting(arg)
                    done["ok"] = True
                except Exception as e:
                    done["exc"] = e

            wt.returnValue.connect(on_fit)
            wt.start()
            self._wait_until(lambda: done["ok"] or done["exc"] is not None, "fact fit")
            if done["exc"] is not None:
                raise done["exc"]

            fitting_primitive = self._find_primitive(ui, project_dir_str, suffix=".fitting_result")
            if fitting_primitive is None:
                raise RuntimeError("FactPoints: fitting result primitive not found")

            if self.config["fact"]["mode"] == "auto":
                self.log("  [fact] auto optimize params")
                primitives = fitting_primitive.update_primitive()
                AutoConnectConfigOptimizer.opitimize_all_params(primitives, 0.3)

            # patch
            self.log("  [fact] auto patch")
            ui.auto_merge_controller.proj_path = project_dir_str
            patched = self._run_auto_patch(ui, pointcloud, fitting_primitive)

            # merge
            self.log("  [fact] auto merge")
            merged = self._run_auto_merge(ui, pointcloud, patched)

            # group
            self.log("  [fact] group")
            ui.group_controller.details = {
                "pointcloud": pointcloud.get_path(),
                "primitive_scene": merged.name,
                "downsample_rate": self.config.get("fact", {}).get("group_rate", 100),
            }
            self._run_group(ui, pointcloud, merged)
            grouped_obj_path = self._export_grouped_obj(ui, task, merged)

            # fitting test
            self.log("  [fact] fitting test")
            self._run_fitting_test(ui, pointcloud, merged)

            report_dir = None
            if self.config.get("enable_report", True):
                self.log("  [fact] report")
                report_dir = self._run_report(ui, project_dir_str, task.cloud_stem)

            ui.close()
            return {
                "project_dir": project_dir_str,
                "report_dir": report_dir,
                "grouped_obj_path": grouped_obj_path,
            }
        finally:
            try:
                dialog_guard.__exit__(None, None, None)  # type: ignore[name-defined]
            except Exception:
                pass
            self._remove_syspath(str(fact_root))
            os.chdir(cwd_prev)

    def _pick_pointcloud(self, sample_dir: Path) -> Optional[Path]:
        cands = []
        for p in sample_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
            name = p.name.lower()
            # Avoid re-using generated label output as input point cloud.
            if "result" in name:
                continue
            cands.append(p)
        if not cands:
            return None
        cands.sort(key=lambda p: (PRIORITY[p.suffix.lower()], p.name.lower()))
        return cands[0]

    def _pick_existing_label(self, sample_dir: Path, cloud_path: Path) -> Optional[str]:
        # Prefer exact "<cloud_stem>_result.npy"; fallback to any file with "result" in folder.
        exact = sample_dir / f"{cloud_path.stem}_result.npy"
        if exact.exists() and exact.is_file():
            return str(exact)

        labels = sorted(
            [
                p
                for p in sample_dir.iterdir()
                if p.is_file() and "result" in p.name.lower() and p.name.lower() != cloud_path.name.lower()
            ],
            key=lambda x: x.name.lower(),
        )
        if labels:
            return str(labels[0])
        return None

    def _resolve_fact_pointcloud_path(self, task: SampleTask) -> str:
        cloud_path = Path(task.pointcloud_path)
        if cloud_path.suffix.lower() != ".e57":
            return task.pointcloud_path

        npy_path = cloud_path.with_suffix(".npy")
        if npy_path.exists() and npy_path.is_file():
            self.log(f"  [fact] use cached npy for e57 -> {npy_path}")
            return str(npy_path)

        self.log(f"  [fact] cached npy not found for e57, fallback -> {cloud_path}")
        return task.pointcloud_path

    def _reset_python_modules_for_fact(self) -> None:
        # Pcot and Fact both use top-level module names like "main" and package "src".
        # Ensure Fact imports from its own tree.
        to_drop = []
        for key in list(sys.modules.keys()):
            if key == "main" or key == "src" or key.startswith("src."):
                to_drop.append(key)
                continue
            if key in {"worker", "signals", "main_window", "main_window_ui", "icon_rc"}:
                to_drop.append(key)
        for key in to_drop:
            sys.modules.pop(key, None)

    def _reset_python_modules_for_pcot(self) -> None:
        to_drop = []
        for key in list(sys.modules.keys()):
            if key == "main" or key == "src" or key.startswith("src."):
                to_drop.append(key)
                continue
            if key in {"worker", "signals", "main_window", "main_window_ui", "icon_rc"}:
                to_drop.append(key)
        for key in to_drop:
            sys.modules.pop(key, None)

        # Inject fake src module to avoid namespace module error during pytorch/inspect loading
        import types
        src_mod = types.ModuleType("src")
        pcot_src_path = str(self._pcot_root() / "src")
        src_mod.__file__ = os.path.join(pcot_src_path, "__init__.py")
        src_mod.__path__ = [pcot_src_path]
        sys.modules["src"] = src_mod

    def _pcot_root(self) -> Path:
        return self._resolve_project_root("pcot_dir_name")

    def _fact_root(self) -> Path:
        return self._resolve_project_root("fact_dir_name")

    def _resolve_project_root(self, key: str) -> Path:
        dir_name = str(self._project_dirs.get(key, "")).strip()
        if not dir_name:
            raise RuntimeError(f"Project dir config '{key}' is empty")
        root = self._root / dir_name
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Project folder not found: {root}")
        return root

    def _prepend_syspath(self, path: str) -> None:
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    def _remove_syspath(self, path: str) -> None:
        while path in sys.path:
            sys.path.remove(path)

    @contextlib.contextmanager
    def _mute_qt_dialogs(self):
        """
        Prevent modal Qt dialogs from blocking automation in worker thread.
        Scope-limited monkey patch; restored on exit.
        """
        from PySide6 import QtWidgets

        msgbox = QtWidgets.QMessageBox
        qfile = QtWidgets.QFileDialog
        qinput = QtWidgets.QInputDialog

        orig_info = msgbox.information
        orig_warn = msgbox.warning
        orig_crit = msgbox.critical
        orig_question = msgbox.question
        orig_get_text = qinput.getText
        orig_get_item = qinput.getItem
        orig_get_int = qinput.getInt
        orig_get_double = qinput.getDouble
        orig_open_name = qfile.getOpenFileName
        orig_open_names = qfile.getOpenFileNames
        orig_save_name = qfile.getSaveFileName
        orig_open_dir = qfile.getExistingDirectory

        def _info(*args, **kwargs):
            text = args[2] if len(args) > 2 else kwargs.get("text", "")
            self.log(f"  [fact][dialog-muted] information: {text}")
            return msgbox.StandardButton.Ok

        def _warn(*args, **kwargs):
            text = args[2] if len(args) > 2 else kwargs.get("text", "")
            self.log(f"  [fact][dialog-muted] warning: {text}")
            return msgbox.StandardButton.Ok

        def _crit(*args, **kwargs):
            text = args[2] if len(args) > 2 else kwargs.get("text", "")
            self.log(f"  [fact][dialog-muted] critical: {text}")
            return msgbox.StandardButton.Ok

        def _question(*args, **kwargs):
            text = args[2] if len(args) > 2 else kwargs.get("text", "")
            self.log(f"  [fact][dialog-muted] question->Yes: {text}")
            return msgbox.StandardButton.Yes

        msgbox.information = _info
        msgbox.warning = _warn
        msgbox.critical = _crit
        msgbox.question = _question

        # Block file/input dialogs in automation mode.
        qfile.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
        qfile.getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
        qfile.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
        qfile.getExistingDirectory = staticmethod(lambda *a, **k: "")
        qinput.getText = staticmethod(lambda *a, **k: ("", False))
        qinput.getItem = staticmethod(lambda *a, **k: ("", False))
        qinput.getInt = staticmethod(lambda *a, **k: (0, False))
        qinput.getDouble = staticmethod(lambda *a, **k: (0.0, False))

        try:
            yield
        finally:
            msgbox.information = orig_info
            msgbox.warning = orig_warn
            msgbox.critical = orig_crit
            msgbox.question = orig_question
            qinput.getText = orig_get_text
            qinput.getItem = orig_get_item
            qinput.getInt = orig_get_int
            qinput.getDouble = orig_get_double
            qfile.getOpenFileName = orig_open_name
            qfile.getOpenFileNames = orig_open_names
            qfile.getSaveFileName = orig_save_name
            qfile.getExistingDirectory = orig_open_dir

    def _write_summary(self, root_dir: str, summary: Dict[str, Any]) -> None:
        out = Path(root_dir) / "batch_automation_summary.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"[summary] saved: {out}")

    def _run_worker_sync(self, ui, worker_cls, name: str, fn: Callable) -> None:
        self.log(f"  [pcot] {name}")
        done = {"ok": False, "err": None}
        worker = worker_cls(fn)
        worker.signals.result.connect(lambda _: None)
        worker.signals.error.connect(lambda e: done.update({"err": e}))
        worker.signals.finished.connect(lambda: done.update({"ok": True}))
        ui.threadpool.start(worker)
        self._wait_until(lambda: done["ok"], f"pcot:{name}")
        if done["err"] is not None:
            raise RuntimeError(f"Pcot step {name} failed: {done['err']}")

    def _wait_until(self, cond: Callable[[], bool], step_name: str, timeout_sec: float = 7200) -> None:
        from PySide6.QtCore import QCoreApplication

        t0 = time.time()
        while not cond():
            if self.should_stop():
                raise RuntimeError(f"Stopped by user during {step_name}")
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"Timeout in step: {step_name}")
            QCoreApplication.processEvents()
            time.sleep(0.05)

    def _prepare_fact_project(self, ui, project_dir: str) -> None:
        project_name = os.path.basename(project_dir)
        if project_dir not in ui.dbTree.projs:
            ui.dbTree.insertProject(project_name, project_dir)
        ui.dbTree.switchToProject(project_dir)
        ui.project_controller.load_project(project_dir)

    def _pick_fact_pointcloud(self, ui, proj_path: str):
        for item in ui.dbTree.entries.get(proj_path, []):
            if item.__class__.__name__ == "PointCloudRootItem":
                pcd = item.pointCloudDB
                if pcd.get_label() is not None and pcd.get_path() is not None:
                    return pcd
        return None

    def _find_primitive(self, ui, proj_path: str, suffix: str):
        for item in ui.dbTree.entries.get(proj_path, []):
            if item.__class__.__name__ == "PrimitiveRootItem":
                p = item.primitiveDB
                if p.name.endswith(suffix):
                    return p
        return None

    def _run_auto_patch(self, ui, pointcloud, primitive):
        from src.controller.base import workerThread  # type: ignore
        from src.pipes_fitting.auto_connection import auto_patching  # type: ignore

        ctrl = ui.auto_merge_controller
        ctrl.is_running = True
        ctrl.result_name = f"{primitive.name}.patched"
        ctrl.details = {"primitive_scene": primitive.name, "pointcloud": pointcloud.get_path()}
        done = {"ok": False, "exc": None, "patched": None}

        wt = workerThread(
            target=auto_patching,
            args=(
                pointcloud.get_points(),
                pointcloud.get_label(),
                pointcloud.get_path(),
                self._fact_merge_rate(ui),
                {primitive.scene_id: primitive.update_primitive()},
            ),
        )

        def on_ret(arg):
            try:
                ctrl.post_auto_patch(arg)
                done["patched"] = self._find_primitive(ui, ctrl.proj_path, ".patched")
                done["ok"] = True
            except Exception as e:
                done["exc"] = e

        wt.returnValue.connect(on_ret)
        wt.start()
        self._wait_until(lambda: done["ok"] or done["exc"] is not None, "fact auto_patch")
        if done["exc"] is not None:
            raise done["exc"]
        if done["patched"] is None:
            raise RuntimeError("FactPoints: patched primitive not found")
        return done["patched"]

    def _run_auto_merge(self, ui, pointcloud, primitive):
        from src.controller.base import workerThread  # type: ignore
        from src.pipes_fitting.auto_connection import auto_merging  # type: ignore

        ctrl = ui.auto_merge_controller
        ctrl.is_running = True
        ctrl.result_name = f"{primitive.name}.merged"
        ctrl.details = {"primitive_scene": primitive.name, "pointcloud": pointcloud.get_path(), "merge_rate": self._fact_merge_rate(ui)}
        done = {"ok": False, "exc": None, "merged": None}

        all_primitives = [p for p in ctrl.get_primitive() if p.get_project_path() == ctrl.proj_path]
        scene_dict = {p.scene_id: p.update_primitive() for p in all_primitives}
        wt = workerThread(
            target=auto_merging,
            args=(
                pointcloud.get_points(),
                pointcloud.get_label(),
                pointcloud.get_path(),
                self._fact_merge_rate(ui),
                scene_dict,
            ),
        )

        def on_ret(arg):
            try:
                ctrl.post_auto_merge(arg)
                done["merged"] = self._find_primitive(ui, ctrl.proj_path, ".merged")
                done["ok"] = True
            except Exception as e:
                done["exc"] = e

        wt.returnValue.connect(on_ret)
        wt.start()
        self._wait_until(lambda: done["ok"] or done["exc"] is not None, "fact auto_merge")
        if done["exc"] is not None:
            raise done["exc"]
        if done["merged"] is None:
            raise RuntimeError("FactPoints: merged primitive not found")
        return done["merged"]

    def _run_group(self, ui, pointcloud, primitive):
        from src.controller.base import workerThread  # type: ignore
        from src.group.get_group_example import get_group  # type: ignore

        ctrl = ui.group_controller
        ctrl.pointcloud = pointcloud
        ctrl.primitive = primitive
        ctrl.current_scene_id = primitive.scene_id

        cylinders, elbows, tees, boxes, steels = primitive.get_all()
        done = {"ok": False, "exc": None}
        wt = workerThread(
            target=get_group,
            args=(
                pointcloud.get_points(),
                pointcloud.get_label(),
                (cylinders, elbows, tees, boxes),
                pointcloud.get_axis(),
                pointcloud.get_direction(),
                pointcloud.get_path(),
            ),
        )

        def on_ret(group):
            try:
                ctrl.post_grouping(group)
                done["ok"] = True
            except Exception as e:
                done["exc"] = e

        wt.returnValue.connect(on_ret)
        wt.start()
        self._wait_until(lambda: done["ok"] or done["exc"] is not None, "fact group")
        if done["exc"] is not None:
            raise done["exc"]

    def _export_grouped_obj(self, ui, task: SampleTask, primitive) -> str:
        out_path = Path(task.sample_dir) / f"{task.cloud_stem}_grouped.obj"
        self.log(f"  [fact] export grouped obj -> {out_path}")
        ui.file_controller.save_fitted_obj(
            str(out_path),
            primitive_db=primitive,
            translation_vector=None,  # keep original coordinates and scale
        )
        return str(out_path)

    def _run_fitting_test(self, ui, pointcloud, primitive):
        from src.controller.base import workerThread  # type: ignore
        from src.pipes_fitting.FitTest import calculate_distances  # type: ignore

        ctrl = ui.fitting_test_controller
        ctrl.details = {
            "pointcloud": pointcloud.get_path(),
            "primitive_scene": primitive.name,
        }
        label = pointcloud.get_main_labels()
        points = pointcloud.get_main_points()
        cylinders, elbows, tees, boxes, steels = primitive.get_all()
        done = {"ok": False, "exc": None}
        wt = workerThread(
            target=calculate_distances,
            args=(
                cylinders,
                elbows,
                tees,
                np.hstack((points, label.T)),
            ),
        )

        def on_ret(rv):
            try:
                ctrl.result_name = f"{primitive.name}.fitting_test_result"
                ctrl.pointcloud = pointcloud
                ctrl.primitive = primitive
                ctrl.post_fitting_test(*rv)
                done["ok"] = True
            except Exception as e:
                done["exc"] = e

        wt.returnValue.connect(on_ret)
        wt.start()
        self._wait_until(lambda: done["ok"] or done["exc"] is not None, "fact fitting_test")
        if done["exc"] is not None:
            raise done["exc"]

    def _run_report(self, ui, proj_path: str, cloud_stem: str) -> str:
        # Use controller core logic without interactive dialog:
        # 1) locate pointclouds with fitting_test_result / failed
        success_pcd = None
        failed_pcd = None
        primitive_db = None
        for entry in ui.dbTree.entries.get(proj_path, []):
            cls_name = entry.__class__.__name__
            if cls_name == "PointCloudRootItem":
                pcd = entry.pointCloudDB
                if pcd.name.endswith("fitting_test_result"):
                    success_pcd = pcd
                elif pcd.name.endswith("fitting_test_result_failed"):
                    failed_pcd = pcd
            elif cls_name == "PrimitiveRootItem":
                p = entry.primitiveDB
                if p.name == cloud_stem or p.name.startswith(cloud_stem):
                    primitive_db = p
        if success_pcd is None:
            raise RuntimeError("Fact report: success pointcloud not found")

        controller = ui.report_gen_controller
        report_dir = str(Path(proj_path) / "webreport")
        os.makedirs(report_dir, exist_ok=True)
        # build data directly by reusing helper methods in controller
        point_cloud = success_pcd.get_points()
        distance = success_pcd.get_attribute("distance")
        labels = success_pcd.get_attribute("label")
        if distance is None:
            distance = np.zeros(point_cloud.shape[0])
        if labels is None:
            labels = [np.zeros(point_cloud.shape[0], dtype=np.uint8), np.zeros(point_cloud.shape[0], dtype=np.uint32)]

        xyz_original = point_cloud[:, :3]
        translation = xyz_original.mean(axis=0)
        xyz_centered = xyz_original - translation
        failed_centered = None
        failed_points = 0
        if failed_pcd is not None:
            failed = failed_pcd.get_points()
            failed_points = failed.shape[0]
            failed_centered = failed[:, :3] - translation

        plypath = os.path.join(report_dir, "output.ply")
        controller.save_ply_with_labels_plyfile(
            plypath, xyz_centered, distance, semantic_labels=labels[0], instance_labels=labels[1]
        )
        if failed_points > 0:
            controller.save_ply_with_labels_plyfile(os.path.join(report_dir, "failed.ply"), failed_centered)

        if primitive_db is not None:
            ui.file_controller.save_fitted_obj(
                os.path.join(report_dir, "output.obj"),
                primitive_db=primitive_db,
                translation_vector=-translation,
            )

        resources_src = Path("resources")
        import shutil
        if resources_src.exists():
            shutil.copytree(str(resources_src), report_dir, dirs_exist_ok=True)

        return report_dir

    def _make_o3d_pcd(self, xyz, rgb):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return pcd

    def _fact_merge_rate(self, ui) -> float:
        import config as fact_cfg  # type: ignore

        _ = ui  # keep signature
        return float(fact_cfg.merge_rate) / 100.0

    def _apply_pcot_runtime_config(self, pcot_root: Path) -> None:
        _ = pcot_root
        self._patch_pcot_dataloader_workers()

    def _patch_pcot_dataloader_workers(self) -> None:
        try:
            import torch.utils.data as tud  # type: ignore
        except Exception:
            return

        if getattr(tud.DataLoader, "__name__", "") == "_PointScriptPatchedDataLoader":
            return

        orig_loader = tud.DataLoader

        class _PointScriptPatchedDataLoader(orig_loader):
            def __init__(self, *args, **kwargs):
                kwargs["num_workers"] = 0
                kwargs["persistent_workers"] = False
                super().__init__(*args, **kwargs)

        tud.DataLoader = _PointScriptPatchedDataLoader
        self.log("  [pcot] patched DataLoader num_workers=0")

    def _apply_fact_runtime_config(self, fact_cfg, adapter_cls) -> None:
        mode = self.config["fact"]["mode"]
        if mode == "default":
            return
        manual = self.config["fact"]["manual"]
        if "merge_rate" in manual:
            merge_rate = int(manual["merge_rate"])
            levels = list(fact_cfg.merge_rate_levels)
            if merge_rate in levels:
                fact_cfg.merge_rate = merge_rate
                fact_cfg.merge_rate_index = levels.index(merge_rate)
        if "group_downsample_rate" in manual:
            ds = int(manual["group_downsample_rate"])
            levels = list(fact_cfg.group_downsample_rate_levels)
            if ds in levels:
                fact_cfg.group_downsample_rate = ds
                fact_cfg.group_downsample_rate_index = levels.index(ds)
        for key, value in manual.get("auto_connect", {}).items():
            try:
                adapter_cls.update_param(key, float(value))
            except Exception:
                pass
