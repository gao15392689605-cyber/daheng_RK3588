"""全局配置 — 路径、UI 颜色、默认参数、类别字典转发"""
from __future__ import annotations

from pathlib import Path

# ── 项目路径 ──────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODEL_DIR: Path = PROJECT_ROOT / "model"
DB_PATH: Path = PROJECT_ROOT / "db" / "tobacco.db"
DB_INIT_SQL: Path = PROJECT_ROOT / "db" / "init.sql"
FONT_PATH: Path = PROJECT_ROOT / "resources" / "fonts" / "font.ttc"
LOG_DIR: Path = PROJECT_ROOT / "logs"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# 默认模型路径(优先 rknn, 回退 pt)
DEFAULT_RKNN: Path = MODEL_DIR / "best.rknn"

# ── UI 主题色 (深色工业风) ─────────────────────────────
COLOR_BG_MAIN: str = "#1A1D24"
COLOR_BG_SIDE: str = "#20242C"
COLOR_BTN_PRIMARY: str = "#2D8CFE"
COLOR_BTN_HOVER: str = "#3E9CFF"
COLOR_TEXT: str = "#F0F2F7"
COLOR_TEXT_DIM: str = "#9099A8"
COLOR_BORDER: str = "#3A3F4E"
COLOR_TABLE_ALT: str = "#252830"
COLOR_HIGHLIGHT: str = "#FFD23F"

# ── 默认参数 ──────────────────────────────────────────
DEFAULT_CONF: float = 0.5
DEFAULT_IOU: float = 0.5
DEFAULT_USERNAME: str = "operator"
DEFAULT_PASSWORD: str = "123456"

# ── 视频/图像格式 ────────────────────────────────────
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
VIDEO_EXTS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".flv")

# ── 检测模式 ──────────────────────────────────────────
MODE_PHOTO: str = "photo"
MODE_VIDEO: str = "video"
MODE_FOLDER: str = "folder"
MODE_CAMERA: str = "camera"

# ── 10 类异物中文名(与 inference.CN_NAMES 同步, 供 UI 选择列表用) ──
CLASS_LIST_CN: tuple[str, ...] = (
    "螺帽", "细杆", "石块红砖", "树叶", "金属碎片",
    "金属", "棉絮", "透明塑料绳", "麻绳", "黑发",
)
