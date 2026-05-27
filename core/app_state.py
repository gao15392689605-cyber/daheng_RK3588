"""模块级单例 AppState — 存储用户、参数、模式、模型路径、检测结果

跨模块通过 from core.app_state import state 直接拿到唯一实例.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_RKNN, MODE_PHOTO


@dataclass
class AppState:
    # 当前登录用户
    username: str = ""
    avatar_path: str = ""

    # 推理参数
    conf_threshold: float = DEFAULT_CONF
    iou_threshold: float = DEFAULT_IOU

    # 模型
    model_path: str = str(DEFAULT_RKNN)
    model_name: str = "best.rknn"
    backend: str = ""           # "rknn" / "pt" — 推理模块初始化后回填
    device_name: str = ""       # 例如 "RK3588 NPU"

    # 模式
    detect_mode: str = MODE_PHOTO

    # 当前帧检测结果(过滤后)
    current_results: list[dict[str, Any]] = field(default_factory=list)
    # 当前帧总数(过滤前)
    current_raw_count: int = 0
    # 类别过滤集合(空集 = 全部显示)
    filter_classes: set[str] = field(default_factory=set)

    # 当前选中行(序号)
    selected_row: int = -1

    # 最近一次检测耗时(秒)
    last_infer_time: float = 0.0

    # 最近一次的原图(np.ndarray, RGB) — 用于导出原图
    last_image_rgb: Any = None
    last_annotated_rgb: Any = None
    last_file_path: str = ""

    # 用户预设的导出目录(空则每次弹窗选)
    export_dir: str = ""


state: AppState = AppState()
