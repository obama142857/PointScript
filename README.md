# 点云批处理自动化项目

本项目是一个点云数据处理的集成工作流，旨在整合点云分割、拟合及报告生成的自动化流程。项目包含两个核心模块 `FactPoints`（主攻拟合）和 `PcotPoints`（主攻深度学习处理），并通过 `batch_automation_ui.py` 提供统一的批处理界面。

## 项目结构

- **`FactPoints/`**: 点云拟合核心模块，包含钢结构、管道等拟合算法及可视化工具。
- **`PcotPoints/`**: 基于深度学习的点云处理模块，侧重于分割与标注。
- **`automation/`**: 自动化批处理逻辑的核心实现，包含 `runner.py` 任务调度程序。
- **`batch_automation_ui.py`**: 基于 PySide6 的图形化批处理工具，用于管理大规模点云任务。

## 核心功能

1. **自动任务发现**: 自动扫描根目录下包含 `.npy`, `.e57`, `.pcd` 的文件夹并生成处理任务。
2. **集成工作流**: 打通从原始点云到拟合结果的数据流。
3. **状态追踪**: 通过 `.fact_automation_done.json` 自动记录任务完成状态，支持故障断点续传。
4. **图形化管理**: 提供实时的日志输出、进度条展示以及处理结果汇总。

## 环境要求

- **操作系统**: Windows (推荐使用 PowerShell)
- **Python**: 3.9+
- **关键依赖**:
  - PySide6 (UI)
  - NumPy, SciPy (计算)
  - PyTorch (深度学习环境，详见 PcotPoints)

## 快速上手

### 1. 环境配置

建议分别为两个核心模块配置环境：

```bash
# FactPoints 环境
cd FactPoints
pip install -r requirements.txt

# PcotPoints 环境 (详见子目录 README)
# 注意：PcotPoints 需要 CUDA 12.1 和 PyTorch 2.1.2 环境
```

### 2. 运行批处理 UI

在主目录下运行：

```bash
python batch_automation_ui.py
```

### 3. 使用说明

- **选择任务目录**: 在 UI 中指定包含多个采样文件夹的根目录。
- **配置参数**: 设置各处理阶段的配置路径（.json/.yaml）。
- **开始任务**: 点击 "Start" 开始自动化批处理，处理结果将保存在各采样文件夹内对应的 `_fact` 或 `_result` 文件夹中。

## 注意事项

- **done 标记**: 程序会自动在已处理成功的目录生成隐藏标记文件。若需重新运行，请手动删除该标记。
- **日志**: 所有运行细节会保存在 `automation_logs` 目录下（若配置开启）。
