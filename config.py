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

# ── 报警分级(严重异物 → 视觉强报警 + 分流 + 查上游;其余 → 仅计数)──
SEVERE_CLASSES: tuple[str, ...] = ("metal", "metal_debris", "screw_cap", "stone")  # 英文类名
ALARM_COOLDOWN_SEC: float = 3.0      # 同类报警冷却(同一异物移动多帧只报一次)
ESCALATE_WINDOW_SEC: float = 300.0   # 频次升级时间窗
ESCALATE_COUNT: int = 3              # 窗口内严重报警达此数 → 升级"建议停线检修"
DIVERT_DELAY_MS: int = 0             # 下游分流闸门延时(按带速标定, 预留)

# ── 工人拍摄存图固定目录(看到离谱东西随手拍, 自动落盘, 不弹窗)──
CAPTURE_DIR: Path = Path("/home/firefly/gaoyifan_yolo_rknn/异物图")
# 本机/非板端回退: 上面路径不可写时存到项目 logs 下
CAPTURE_DIR_FALLBACK: Path = LOG_DIR / "captures"

# ── 工人录制视频固定目录 ──
VIDEO_DIR: Path = Path("/home/firefly/gaoyifan_yolo_rknn/录制视频")
VIDEO_DIR_FALLBACK: Path = LOG_DIR / "videos"
VIDEO_FPS: float = 15.0   # 录制帧率(检测流约 10-20fps, 取中)

# ── 相机解耦显示(画面按相机原速刷, 推理在后台线程跑, 让画面顺) ──
# 只影响"相机显示方式", 不碰推理/合并逻辑。
# 关闭原因(板端反馈): 解耦下画面跑最新帧、检测框是几帧前算的, 框会"漂移/偏离"物体;
# 改 False 走逐帧 —— 框与画面同帧、严丝合缝(OBB ~7fps / seg ~2.6fps, 慢但不偏)。
DECOUPLED_CAMERA: bool = False

# ── 产线生产配置持久化(技术员调好后"保存生效", 工人据此跑)──
LINE_CONFIG_PATH: Path = PROJECT_ROOT / "db" / "line_config.json"

# ── 审计操作中文名(管理面板审计日志列显示用)──
AUDIT_ACTION_CN: dict[str, str] = {
    "login": "登录",
    "logout": "退出登录",
    "param_change": "调整参数",
    "model_switch": "切换模型",
    "config_publish": "发布生产配置",
    "user_create": "新建账号",
    "user_role": "修改角色",
    "user_active": "启用/停用账号",
    "user_reset_pwd": "重置密码",
    "user_delete": "删除账号",
    "user_edit": "编辑账号信息",
    "alarm_handled": "处置报警",
    "batch_start": "开始批次",
    "batch_end": "结束批次",
    "capture": "拍摄存图",
    "video_record": "录制视频",
    "clear_demo": "清空演示数据",
}

# ── 通用按钮样式(统一 hover/pressed 点击反馈)──
def btn_qss(bg: str, hover: str | None = None, pressed: str | None = None,
            fg: str = "white", radius: int = 4, pad: str = "6px 14px") -> str:
    hover = hover or COLOR_BTN_HOVER
    pressed = pressed or bg
    return (
        f"QPushButton {{ background:{bg}; color:{fg}; border:none;"
        f"border-radius:{radius}px; padding:{pad}; }}"
        f"QPushButton:hover {{ background:{hover}; }}"
        f"QPushButton:pressed {{ background:{pressed}; padding-top:7px; padding-bottom:5px; }}"
        f"QPushButton:checked {{ background:{pressed}; border:2px solid {COLOR_HIGHLIGHT}; font-weight:bold; }}"
        f"QPushButton:disabled {{ background:#3A3F4E; color:{COLOR_TEXT_DIM}; }}"
    )
