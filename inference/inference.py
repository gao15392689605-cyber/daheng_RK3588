"""推理 + 后处理（同类合并）+ 中文标注绘制

支持两套后端, 自动检测:
  - PT:   开发/桌面 (Windows/Linux + PyTorch + 本地 ultralytics)
  - RKNN: 嵌入式 (RK3588/RK3588S + rknn-toolkit-lite2)

切换逻辑: 找到 best.rknn + rknnlite 可导入 → RKNN; 否则回退 PT.
对外 API (load_model, run_inference, PRESETS, CN_NAMES, DEVICE_NAME) 完全相同.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.errors import GEOSException
from shapely.geometry import Polygon

# ── 路径配置 ──────────────────────────────────────────
HERE = Path(__file__).resolve().parent
G2_ROOT = HERE.parent  # G2模型/
PROJECT_ROOT = G2_ROOT.parent  # gaoyifan/

PT_PATH = G2_ROOT / "模型权重" / "best.pt"
RKNN_PATH = G2_ROOT / "模型权重" / "best.rknn"


# ── 后端自动检测 ──────────────────────────────────────
# RKNN_API_MODE:
#   "lite"       Station M3 真机 (rknn-toolkit-lite2, 设备端)
#   "simulator"  PC 服务器 (rknn-toolkit2 + simulator, 仅开发验证)
#   None         未装 RKNN
RKNN_API_MODE: str | None = None
ONNX_PATH = G2_ROOT / "模型权重" / "best.onnx"  # toolkit2 simulator 需要从 ONNX 重 build


def _detect_backend() -> tuple[str, Path]:
    """优先 RKNN (嵌入式), 其次 RKNN simulator (PC), 最后回退 PT."""
    global RKNN_API_MODE
    # 1. 真机端: rknn-toolkit-lite2
    try:
        from rknnlite.api import RKNNLite  # noqa: F401
        if RKNN_PATH.exists():
            RKNN_API_MODE = "lite"
            return "rknn", RKNN_PATH
    except ImportError:
        pass
    # 2. PC 端模拟: rknn-toolkit2 + simulator (需要 ONNX 重 build, 启动慢)
    try:
        from rknn.api import RKNN  # noqa: F401
        if ONNX_PATH.exists():
            RKNN_API_MODE = "simulator"
            return "rknn", ONNX_PATH  # 注意: simulator 加载的是 ONNX 不是 .rknn
    except ImportError:
        pass
    # 3. 回退 PT
    return "pt", PT_PATH


BACKEND, MODEL_PATH = _detect_backend()

# ── PT 后端: 仅在选中 PT 时导入 ───────────────────────
if BACKEND == "pt":
    import torch
    LOCAL_ULTRA = PROJECT_ROOT / "ultralytics"
    if LOCAL_ULTRA.exists():
        sys.path.insert(0, str(LOCAL_ULTRA))
    from ultralytics import YOLO  # noqa: E402
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME = (
        f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "CPU"
    )
else:
    # RKNN 后端: 不需要 torch/ultralytics
    YOLO = None  # type: ignore
    DEVICE = "npu"
    DEVICE_NAME = "RK3588 NPU"

print(f"[inference] 后端: {BACKEND.upper()}  设备: {DEVICE_NAME}  模型: {MODEL_PATH.name}")

# ── 类别中文映射 ──────────────────────────────────────
CN_NAMES = {
    "screw_cap":    "螺帽",
    "thin_rod":     "细杆",
    "stone":        "石块红砖",
    "leaf":         "树叶",
    "metal_debris": "金属碎片",
    "metal":        "金属",
    "mian_xu":      "棉絮",
    "sls":          "透明塑料绳",
    "hemp_rope":    "麻绳",
    "hair":         "黑发",
}

# 每类一个颜色 (BGR for OpenCV)
CLASS_COLORS_BGR = {
    "screw_cap":    (0, 215, 255),     # 金黄
    "thin_rod":     (255, 144, 30),    # 道奇蓝
    "stone":        (75, 80, 165),     # 砖红
    "leaf":         (50, 220, 50),     # 中绿
    "metal_debris": (192, 192, 192),   # 银
    "metal":        (100, 220, 240),   # 浅金
    "mian_xu":      (240, 240, 240),   # 雪白
    "sls":          (0, 165, 255),     # 橙
    "hemp_rope":    (60, 60, 220),     # 红
    "hair":         (200, 100, 240),   # 紫
}

# ── 三档模式 ──────────────────────────────────────────
PRESETS = {
    "高敏感度": {"conf": 0.4, "iou": 0.5, "desc": "排查不放过, 不能漏检场景"},
    "标准":     {"conf": 0.5, "iou": 0.5, "desc": "平衡之选, 推荐日常使用"},
    "高准确度": {"conf": 0.6, "iou": 0.5, "desc": "只报最确定的, 不能误报场景"},
}


def conf_tier(c: float) -> tuple[str, str]:
    """返回 (中文档位, hex 颜色 - 科技感配色)"""
    if c >= 0.80:
        return "高", "#00ff88"   # 亮绿
    if c >= 0.50:
        return "中", "#ffaa00"   # 橙黄
    return "低", "#ff3366"        # 亮红


# ════════════════════════════════════════════════════
# 模型单例 (惰性加载)
# ════════════════════════════════════════════════════
_MODEL = None  # PT: YOLO 实例; RKNN: RKNNLite 实例
_CLS_NAMES_LIST = list(CN_NAMES.keys())  # 类别索引 → 英文名


def load_model():
    """加载模型 (惰性), 在 splash 屏上调用"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    assert MODEL_PATH.exists(), f"找不到模型文件: {MODEL_PATH}"

    if BACKEND == "rknn":
        if RKNN_API_MODE == "lite":
            # 真机: 直接 load .rknn → init NPU
            from rknnlite.api import RKNNLite
            _MODEL = RKNNLite()
            if _MODEL.load_rknn(str(MODEL_PATH)) != 0:
                raise RuntimeError(f"RKNN load_rknn 失败: {MODEL_PATH}")
            if _MODEL.init_runtime() != 0:
                raise RuntimeError("RKNN init_runtime 失败")
        elif RKNN_API_MODE == "simulator":
            # PC: 从 ONNX 重 build (启动 1-2 分钟, 仅用于开发验证)
            from rknn.api import RKNN
            _MODEL = RKNN(verbose=False)
            calib = PROJECT_ROOT / "calib_list.txt"
            _MODEL.config(
                mean_values=[[0]], std_values=[[255]],
                target_platform="rk3588",
                quantized_dtype="asymmetric_quantized-8",
                optimization_level=3,
            )
            print(f"[inference] RKNN simulator 启动中 (从 ONNX build, ~1 分钟)...")
            if _MODEL.load_onnx(model=str(MODEL_PATH)) != 0:
                raise RuntimeError("RKNN load_onnx 失败")
            do_quant = calib.exists()
            if _MODEL.build(do_quantization=do_quant,
                            dataset=str(calib) if do_quant else None) != 0:
                raise RuntimeError("RKNN build 失败")
            if _MODEL.init_runtime(target=None) != 0:  # target=None = simulator
                raise RuntimeError("RKNN init_runtime (simulator) 失败")
            print(f"[inference] RKNN simulator 就绪")
    else:
        _MODEL = YOLO(str(MODEL_PATH))
        _MODEL.to(DEVICE)
    return _MODEL


def get_model():
    if _MODEL is None:
        return load_model()
    return _MODEL


# ════════════════════════════════════════════════════
# RKNN 后处理: DFL 解码 + NMS (纯 numpy + cv2, 无 torch)
# ════════════════════════════════════════════════════
_IMGSZ = 640
_STRIDES = [8, 16, 32]
_REG_MAX = 16
_NC = 10
_ANCHORS_XY: np.ndarray | None = None
_STRIDE_T: np.ndarray | None = None


def _make_anchors():
    """生成 8400 个 anchor (cx, cy, stride), 只算一次缓存"""
    global _ANCHORS_XY, _STRIDE_T
    if _ANCHORS_XY is not None:
        return
    xs_all, ys_all, ss_all = [], [], []
    for stride in _STRIDES:
        H = W = _IMGSZ // stride
        sx = (np.arange(W) + 0.5) * stride
        sy = (np.arange(H) + 0.5) * stride
        yv, xv = np.meshgrid(sy, sx, indexing="ij")
        xs_all.append(xv.flatten())
        ys_all.append(yv.flatten())
        ss_all.append(np.full(H * W, stride, dtype=np.float32))
    _ANCHORS_XY = np.stack([np.concatenate(xs_all), np.concatenate(ys_all)], axis=1).astype(np.float32)
    _STRIDE_T = np.concatenate(ss_all).astype(np.float32)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _decode_head(head_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(1, 74, H, W) → (dist (1,4,N), cls_logits (1,nc,N))"""
    bs, _, H, W = head_raw.shape
    N = H * W
    box_reg = head_raw[:, :4 * _REG_MAX].reshape(bs, 4, _REG_MAX, N)
    cls_log = head_raw[:, 4 * _REG_MAX:].reshape(bs, _NC, N)
    box_reg = _softmax(box_reg, axis=2)
    weights = np.arange(_REG_MAX, dtype=np.float32).reshape(1, 1, _REG_MAX, 1)
    dist = (box_reg * weights).sum(axis=2)  # (1, 4, N)
    return dist, cls_log


def _assemble_rknn_outputs(outs: list[np.ndarray]) -> np.ndarray:
    """outs: rknn.inference 的 5 个返回值
    [0]: 拼装版 (坏的, 量化压死了)
    [1-3]: P3/P4/P5 head raw (1, 74, h, w)
    [4]: angle (1, 1, 8400)
    → (1, 15, 8400) = box(cxcywh) + cls(sigmoid) + angle
    """
    _make_anchors()
    dists, cls_logs = [], []
    for h in [outs[1], outs[2], outs[3]]:
        d, c = _decode_head(h)
        dists.append(d)
        cls_logs.append(c)
    dist = np.concatenate(dists, axis=-1)            # (1, 4, 8400)
    cls_log = np.concatenate(cls_logs, axis=-1)      # (1, 10, 8400)

    # ltrb (stride 单位) → cxcywh (像素)
    stride = _STRIDE_T[None, None, :]
    l, t, r, b = dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4]
    ax = _ANCHORS_XY[:, 0][None, None]
    ay = _ANCHORS_XY[:, 1][None, None]
    x1, y1 = ax - l * stride, ay - t * stride
    x2, y2 = ax + r * stride, ay + b * stride
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h_box = x2 - x1, y2 - y1
    box = np.concatenate([cx, cy, w, h_box], axis=1)  # (1, 4, 8400)
    cls = _sigmoid(cls_log)
    angle = outs[4]
    return np.concatenate([box, cls, angle], axis=1).astype(np.float32)


def _obb_nms(preds: np.ndarray, conf_th: float, iou_th: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """轻量 OBB NMS (用 shapely 算 IoU).
    preds: (1, 15, 8400) → (boxes_xyxyxyxy, cls_ids, confs)
    """
    p = preds[0].T  # (8400, 15)
    box_cxcywh = p[:, :4]
    cls_scores = p[:, 4:14]
    angles = p[:, 14]

    # 取每个 anchor 最高类得分
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)
    mask = confs >= conf_th
    if mask.sum() == 0:
        return np.zeros((0, 4, 2), dtype=np.float32), np.zeros(0, dtype=int), np.zeros(0)

    box_cxcywh = box_cxcywh[mask]
    cls_ids = cls_ids[mask]
    confs = confs[mask]
    angles = angles[mask]

    # 转 4 角点 (在 640x640 空间)
    polys_pts = []
    polys_shapely = []
    for (cx, cy, w, h), ang in zip(box_cxcywh, angles):
        rect = ((cx, cy), (w, h), np.rad2deg(ang))
        pts = cv2.boxPoints(rect)
        polys_pts.append(pts)
        try:
            poly = Polygon(pts)
            polys_shapely.append(poly if poly.is_valid else poly.buffer(0))
        except Exception:
            polys_shapely.append(None)

    # 按 conf 降序排
    order = np.argsort(-confs)

    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i_idx, i in enumerate(order):
        if suppressed[i_idx]:
            continue
        keep.append(int(i))
        poly_i = polys_shapely[i]
        if poly_i is None:
            continue
        a_i = poly_i.area
        for j_idx in range(i_idx + 1, len(order)):
            if suppressed[j_idx]:
                continue
            j = order[j_idx]
            if cls_ids[i] != cls_ids[j]:
                continue  # 同类才比 IoU
            poly_j = polys_shapely[j]
            if poly_j is None:
                continue
            try:
                inter = poly_i.intersection(poly_j).area
                union = a_i + poly_j.area - inter
                if union > 0 and inter / union > iou_th:
                    suppressed[j_idx] = True
            except GEOSException:
                continue

    keep = np.array(keep, dtype=int)
    return (
        np.array([polys_pts[k] for k in keep], dtype=np.float32),
        cls_ids[keep],
        confs[keep],
    )


def _rknn_predict_obb(
    gray_hw1: np.ndarray, conf_th: float, iou_th: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RKNN 推理 + 后处理. 返回 (boxes_4pts_in_orig, cls_ids, confs)."""
    rknn = get_model()
    orig_h, orig_w = gray_hw1.shape[:2]

    # letterbox 到 640
    scale = _IMGSZ / max(orig_h, orig_w)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = cv2.resize(gray_hw1[:, :, 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = _IMGSZ - new_w
    pad_h = _IMGSZ - new_h
    top, left = pad_h // 2, pad_w // 2
    canvas = np.full((_IMGSZ, _IMGSZ), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    # NHWC 输入
    inp = canvas[None, :, :, None]
    outs = rknn.inference(inputs=[inp])
    preds = _assemble_rknn_outputs(outs)  # (1, 15, 8400)

    pts, cids, confs = _obb_nms(preds, conf_th, iou_th)
    if len(pts) == 0:
        return pts, cids, confs

    # 反 letterbox: (640 坐标) → (原图坐标)
    pts[..., 0] = (pts[..., 0] - left) / scale
    pts[..., 1] = (pts[..., 1] - top) / scale
    return pts, cids, confs


# ════════════════════════════════════════════════════
# 同类重叠框合并
# ════════════════════════════════════════════════════
def merge_same_class_overlap(
    boxes: np.ndarray, cls_ids: np.ndarray, confs: np.ndarray
) -> tuple[list[np.ndarray], list[int], list[float], list[int]]:
    """同类任何接触即合并为最小外接旋转矩形"""
    N = len(boxes)
    if N == 0:
        return [], [], [], []

    class_indices: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cls_ids):
        class_indices[int(c)].append(i)

    out_boxes: list[np.ndarray] = []
    out_cls: list[int] = []
    out_confs: list[float] = []
    out_sizes: list[int] = []

    for cid, indices in class_indices.items():
        if len(indices) == 1:
            i = indices[0]
            out_boxes.append(boxes[i])
            out_cls.append(cid)
            out_confs.append(float(confs[i]))
            out_sizes.append(1)
            continue

        polys = []
        for i in indices:
            try:
                p = Polygon(boxes[i])
                if not p.is_valid:
                    p = p.buffer(0)
                polys.append(p)
            except Exception:
                polys.append(None)

        n = len(indices)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for ii in range(n):
            for jj in range(ii + 1, n):
                if polys[ii] is None or polys[jj] is None:
                    continue
                try:
                    inter = polys[ii].intersection(polys[jj])
                    if not inter.is_empty and inter.area > 0:
                        union(ii, jj)
                except GEOSException:
                    continue

        groups: dict[int, list[int]] = defaultdict(list)
        for ii in range(n):
            groups[find(ii)].append(ii)

        for grp in groups.values():
            if len(grp) == 1:
                i = indices[grp[0]]
                out_boxes.append(boxes[i])
                out_cls.append(cid)
                out_confs.append(float(confs[i]))
                out_sizes.append(1)
            else:
                pts = np.concatenate(
                    [boxes[indices[ii]] for ii in grp], axis=0
                ).astype(np.float32)
                rect = cv2.minAreaRect(pts)
                merged_pts = cv2.boxPoints(rect)
                out_boxes.append(merged_pts)
                out_cls.append(cid)
                out_confs.append(max(float(confs[indices[ii]]) for ii in grp))
                out_sizes.append(len(grp))

    return out_boxes, out_cls, out_confs, out_sizes


# ════════════════════════════════════════════════════
# 中文标注绘制 (PIL)
# ════════════════════════════════════════════════════
def _load_font(size: int) -> ImageFont.ImageFont:
    """加载中文字体, 优先用打包字体, 然后系统回退"""
    bundled = HERE / "assets" / "font.ttc"
    candidates = [
        str(bundled),
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_annotations(
    img_bgr: np.ndarray,
    boxes: list[np.ndarray],
    cls_names_eng: list[str],
    confs: list[float],
    sizes: list[int],
) -> np.ndarray:
    """用 OpenCV 画框 + PIL 写中文标签"""
    out = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    short_edge = min(h, w)

    # 1. 画框 (OpenCV)
    for box, cls_eng, sz in zip(boxes, cls_names_eng, sizes):
        color = CLASS_COLORS_BGR.get(cls_eng, (0, 255, 0))
        pts = np.asarray(box).astype(np.int32)
        thickness = max(3, short_edge // 500) if sz > 1 else max(2, short_edge // 700)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

    # 2. 写中文标签 (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_size = max(20, short_edge // 60)
    font = _load_font(font_size)

    for box, cls_eng, conf, sz in zip(boxes, cls_names_eng, confs, sizes):
        cn_name = CN_NAMES.get(cls_eng, cls_eng)
        tier, _ = conf_tier(conf)
        color_bgr = CLASS_COLORS_BGR.get(cls_eng, (0, 255, 0))
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        suffix = f" ×{sz}" if sz > 1 else ""
        label = f"{cn_name} {tier} {conf:.2f}{suffix}"

        pts = np.asarray(box).astype(np.int32)
        x, y = int(pts[0][0]), int(pts[0][1])

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        pad = max(4, font_size // 5)
        # 标签放在框上方, 边缘检查
        ly = max(th + pad * 2, y)
        draw.rectangle(
            [x, ly - th - pad * 2, x + tw + pad * 2, ly],
            fill=color_rgb,
        )
        draw.text(
            (x + pad, ly - th - pad - 2),
            label,
            fill=(20, 20, 20),
            font=font,
        )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ════════════════════════════════════════════════════
# 主推理接口
# ════════════════════════════════════════════════════
def run_inference(image_rgb: np.ndarray, mode: str = "标准", skip_draw: bool = False) -> dict:
    """主推理函数

    Args:
        image_rgb: [H, W, 3] uint8 RGB 图
        mode: '高敏感度' / '标准' / '高准确度'

    Returns:
        {
            'annotated_rgb': [H, W, 3] 标注后 RGB 图,
            'detections': [
                {'class': '树叶', 'class_eng': 'leaf',
                 'tier': '高', 'tier_color': '#00ff88',
                 'conf': 0.97, 'merged': 1},
                ...
            ],
            'raw_count': 推理输出的原始框数,
            'final_count': 合并后的框数,
        }
    """
    t_start = time.perf_counter()
    cfg = PRESETS.get(mode, PRESETS["标准"])
    conf_th, iou_th = cfg["conf"], cfg["iou"]

    # 记录原始图像信息 (缩放前)
    orig_h, orig_w = image_rgb.shape[:2]
    orig_bytes = image_rgb.nbytes

    # 缩放大图
    h, w = image_rgb.shape[:2]
    if max(h, w) > 4096:
        scale = 4096 / max(h, w)
        image_rgb = cv2.resize(
            image_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        h, w = image_rgb.shape[:2]

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_1ch = gray[:, :, None]

    img_area = float(h) * float(w)

    common = {
        "image_shape": (orig_h, orig_w),
        "image_bytes": int(orig_bytes),
        "settings": {
            "mode": mode,
            "conf": conf_th,
            "iou": iou_th,
        },
        "inference_time": 0.0,  # 占位, 下面覆盖
    }

    # ── 后端分发 ──
    if BACKEND == "rknn":
        raw_boxes, raw_cls, raw_confs = _rknn_predict_obb(gray_1ch, conf_th, iou_th)
        raw_n = len(raw_boxes)
        # 类别索引 → 英文名映射 (RKNN 没有 names dict, 用全局 _CLS_NAMES_LIST)
        eng_lookup = {i: name for i, name in enumerate(_CLS_NAMES_LIST)}
    else:
        model = get_model()
        results = model.predict(
            source=gray_1ch, conf=conf_th, iou=iou_th, imgsz=640,
            device=DEVICE, verbose=False,
        )
        r = results[0]
        if r.obb is None or len(r.obb) == 0:
            raw_boxes = np.zeros((0, 4, 2), dtype=np.float32)
            raw_cls = np.zeros(0, dtype=int)
            raw_confs = np.zeros(0)
            raw_n = 0
        else:
            raw_boxes = r.obb.xyxyxyxy.cpu().numpy()
            raw_cls = r.obb.cls.cpu().numpy().astype(int)
            raw_confs = r.obb.conf.cpu().numpy()
            raw_n = len(raw_boxes)
        eng_lookup = r.names if raw_n > 0 else {i: n for i, n in enumerate(_CLS_NAMES_LIST)}

    if raw_n == 0:
        common["inference_time"] = time.perf_counter() - t_start
        return {
            "annotated_rgb": image_rgb,
            "detections": [],
            "raw_count": 0,
            "final_count": 0,
            **common,
        }

    boxes, cls_ids, confs, sizes = merge_same_class_overlap(
        raw_boxes, raw_cls, raw_confs
    )
    cls_names_eng = [eng_lookup[int(c)] for c in cls_ids]

    # 跳过画框: UI 自己用 ObbOverlay 渲染框, 省 10-80ms PIL/cv2 开销
    if skip_draw:
        annotated_rgb = image_rgb
    else:
        annotated_bgr = draw_annotations(img_bgr, boxes, cls_names_eng, confs, sizes)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # 计算面积占比 + 写入 points (UI overlay 直接用)
    detections = []
    total_area = 0.0
    for box, eng, c, s in zip(boxes, cls_names_eng, confs, sizes):
        tier, tier_color = conf_tier(c)
        area = float(cv2.contourArea(np.asarray(box, dtype=np.float32)))
        ratio = area / img_area if img_area > 0 else 0.0
        total_area += area
        detections.append({
            "class": CN_NAMES.get(eng, eng),
            "class_eng": eng,
            "tier": tier,
            "tier_color": tier_color,
            "conf": float(c),
            "merged": int(s),
            "area_ratio": ratio,
            "points": [(float(p[0]), float(p[1])) for p in box],
        })

    common["inference_time"] = time.perf_counter() - t_start
    return {
        "annotated_rgb": annotated_rgb,
        "detections": detections,
        "raw_count": raw_n,
        "final_count": len(boxes),
        "total_area_ratio": total_area / img_area if img_area > 0 else 0.0,
        **common,
    }
