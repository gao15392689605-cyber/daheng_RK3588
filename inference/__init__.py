"""推理模块封装层

inference.py 在模块顶层就跑了 _detect_backend(), 期望在
G2_ROOT/模型权重/best.rknn 找到模型. 由于本项目把模型放在 model/ 下,
必须在 import inference **之前** 建好符号链接 模型权重 → model/, 否则
板端 (无 torch) 会落到 PT 分支并直接 `import torch` 崩溃.

对外暴露:
    load_model(), run_inference_runtime(img, conf, iou), release_model(),
    PRESETS, CN_NAMES, CLASS_COLORS_BGR, DEVICE_NAME, BACKEND
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ════════════════════════════════════════════════════════════
# 关键: 在 import inference **之前** 建好符号链接
# inference.py 的 G2_ROOT = HERE.parent = 项目根, 它会查 G2_ROOT/模型权重/best.rknn
# ════════════════════════════════════════════════════════════
_PROJ_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJ_ROOT / "model"
_FAKE_G2 = _PROJ_ROOT / "模型权重"

if not _FAKE_G2.exists() and _MODEL_DIR.exists():
    try:
        _FAKE_G2.symlink_to(_MODEL_DIR, target_is_directory=True)
    except (OSError, FileExistsError):
        # 兜底: 文件系统不支持 symlink 时, 复制到目标 (慎用, 13MB)
        try:
            import shutil
            _FAKE_G2.mkdir(parents=True, exist_ok=True)
            for f in _MODEL_DIR.iterdir():
                dst = _FAKE_G2 / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
        except Exception:
            pass

# 现在再 import — inference.py 顶层的 _detect_backend() 就能找到 best.rknn
from . import inference as _core

# 再做一次显式覆盖, 防止后续有人手动改路径
_core.RKNN_PATH = _MODEL_DIR / "best.rknn"
_core.PT_PATH = _MODEL_DIR / "best.pt"
_core.ONNX_PATH = _MODEL_DIR / "best.onnx"

# 字体: 让 inference._load_font 优先用项目字体
try:
    from config import FONT_PATH
    _core_assets = _core.HERE / "assets"
    _core_assets.mkdir(parents=True, exist_ok=True)
    bundled = _core_assets / "font.ttc"
    if not bundled.exists() and FONT_PATH.exists():
        import shutil
        shutil.copy(FONT_PATH, bundled)
except Exception:
    pass


# ── 转发 ──────────────────────────────────────────────
PRESETS = _core.PRESETS
CN_NAMES = _core.CN_NAMES
CLASS_COLORS_BGR = _core.CLASS_COLORS_BGR
BACKEND = _core.BACKEND
DEVICE_NAME = _core.DEVICE_NAME
load_model = _core.load_model
get_model = _core.get_model


# ════════════════════════════════════════════════════════════
# 快速 OBB NMS + 同类合并: 用 cv2.rotatedRectangleIntersection (C++)
# 替代 inference.py 里的 shapely Polygon (纯 Python), 省 60-150ms/帧
# ════════════════════════════════════════════════════════════
import cv2
from collections import defaultdict


def _obb_nms_cv2(preds: np.ndarray, conf_th: float, iou_th: float):
    """cv2 加速版 OBB NMS — 替代 shapely 实现, 同接口"""
    p = preds[0].T  # (8400, 15)
    box_cxcywh = p[:, :4]
    cls_scores = p[:, 4:14]
    angles = p[:, 14]

    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)
    mask = confs >= conf_th
    if mask.sum() == 0:
        return (np.zeros((0, 4, 2), dtype=np.float32),
                np.zeros(0, dtype=int), np.zeros(0))

    box_cxcywh = box_cxcywh[mask]
    cls_ids = cls_ids[mask]
    confs = confs[mask]
    angles = angles[mask]

    rects = []
    polys_pts = []
    areas = np.empty(len(confs), dtype=np.float32)
    for k, ((cx, cy, w, h), ang) in enumerate(zip(box_cxcywh, angles)):
        rect = ((float(cx), float(cy)), (float(w), float(h)),
                float(np.rad2deg(ang)))
        rects.append(rect)
        polys_pts.append(cv2.boxPoints(rect))
        areas[k] = float(w) * float(h)

    order = np.argsort(-confs)
    keep: list[int] = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i_idx, i in enumerate(order):
        if suppressed[i_idx]:
            continue
        keep.append(int(i))
        rect_i = rects[i]; ai = areas[i]; cid_i = cls_ids[i]
        for j_idx in range(i_idx + 1, len(order)):
            if suppressed[j_idx]:
                continue
            j = order[j_idx]
            if cls_ids[j] != cid_i:  # 同类才比 IoU
                continue
            try:
                ret, ipts = cv2.rotatedRectangleIntersection(rect_i, rects[j])
                if ret == 0 or ipts is None or len(ipts) < 3:
                    continue
                inter = float(cv2.contourArea(ipts))
                union = ai + areas[j] - inter
                if union > 0 and inter / union > iou_th:
                    suppressed[j_idx] = True
            except cv2.error:
                continue

    keep_arr = np.array(keep, dtype=int)
    return (
        np.array([polys_pts[k] for k in keep_arr], dtype=np.float32),
        cls_ids[keep_arr],
        confs[keep_arr],
    )


def _merge_overlap_cv2(boxes, cls_ids, confs):
    """cv2 加速版同类合并 — 用 union-find + cv2.rotatedRectangleIntersection"""
    n_total = len(boxes)
    if n_total == 0:
        return [], [], [], []

    cls_idx_map: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cls_ids):
        cls_idx_map[int(c)].append(i)

    out_boxes: list[np.ndarray] = []
    out_cls: list[int] = []
    out_confs: list[float] = []
    out_sizes: list[int] = []

    for cid, indices in cls_idx_map.items():
        if len(indices) == 1:
            i = indices[0]
            out_boxes.append(boxes[i]); out_cls.append(cid)
            out_confs.append(float(confs[i])); out_sizes.append(1)
            continue

        rects = [cv2.minAreaRect(boxes[i].astype(np.float32)) for i in indices]
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
                try:
                    ret, ipts = cv2.rotatedRectangleIntersection(rects[ii], rects[jj])
                    if ret != 0 and ipts is not None and len(ipts) >= 3 \
                            and cv2.contourArea(ipts) > 0:
                        union(ii, jj)
                except cv2.error:
                    continue

        groups: dict[int, list[int]] = defaultdict(list)
        for ii in range(n):
            groups[find(ii)].append(ii)

        for grp in groups.values():
            if len(grp) == 1:
                i = indices[grp[0]]
                out_boxes.append(boxes[i]); out_cls.append(cid)
                out_confs.append(float(confs[i])); out_sizes.append(1)
            else:
                pts = np.concatenate(
                    [boxes[indices[ii]] for ii in grp], axis=0
                ).astype(np.float32)
                rect = cv2.minAreaRect(pts)
                out_boxes.append(cv2.boxPoints(rect))
                out_cls.append(cid)
                out_confs.append(max(float(confs[indices[ii]]) for ii in grp))
                out_sizes.append(len(grp))

    return out_boxes, out_cls, out_confs, out_sizes


# 替换 inference 里的两个慢函数
_core._obb_nms = _obb_nms_cv2
_core.merge_same_class_overlap = _merge_overlap_cv2


# ════════════════════════════════════════════════════════════
# 快速 DFL 解码 + assemble — einsum 合并 softmax+加权和, 减少中间数组
# 替代 inference._decode_head + _assemble_rknn_outputs
# ════════════════════════════════════════════════════════════
_REG_MAX = _core._REG_MAX
_DFL_WEIGHTS = np.arange(_REG_MAX, dtype=np.float32)


def _decode_head_fast(head_raw: np.ndarray):
    """DFL 解码: reshape -> softmax(axis=2) -> 加权和, 用 einsum 合并"""
    bs, _, H, W = head_raw.shape
    N = H * W
    h32 = head_raw if head_raw.dtype == np.float32 else head_raw.astype(np.float32, copy=False)
    box_reg = h32[:, :4 * _REG_MAX].reshape(bs, 4, _REG_MAX, N)
    cls_log = h32[:, 4 * _REG_MAX:].reshape(bs, _core._NC, N)
    # softmax: 数值稳定的 exp(x - max) / sum
    m = box_reg.max(axis=2, keepdims=True)
    e = np.exp(box_reg - m)
    s_inv = 1.0 / e.sum(axis=2)        # (bs, 4, N)
    # dist = sum_k (e[..k..] * k) / s, 用 einsum 一次完成
    dist = np.einsum('brkn,k->brn', e, _DFL_WEIGHTS) * s_inv
    return dist, cls_log


def _assemble_rknn_outputs_fast(outs):
    _core._make_anchors()
    d1, c1 = _decode_head_fast(outs[1])
    d2, c2 = _decode_head_fast(outs[2])
    d3, c3 = _decode_head_fast(outs[3])
    dist = np.concatenate([d1, d2, d3], axis=-1)        # (1, 4, 8400)
    cls_log = np.concatenate([c1, c2, c3], axis=-1)     # (1, 10, 8400)

    stride = _core._STRIDE_T                            # (8400,)
    ax = _core._ANCHORS_XY[:, 0]
    ay = _core._ANCHORS_XY[:, 1]

    # ltrb (stride 单位) × stride -> 像素; 再 cxcywh
    sl = dist[0, 0] * stride
    st = dist[0, 1] * stride
    sr = dist[0, 2] * stride
    sb = dist[0, 3] * stride
    cx = ax + (sr - sl) * 0.5
    cy = ay + (sb - st) * 0.5
    w_box = sl + sr
    h_box = st + sb
    box = np.stack([cx, cy, w_box, h_box], axis=0)[None]  # (1, 4, 8400)

    # sigmoid 整块
    cls = 1.0 / (1.0 + np.exp(-cls_log))
    angle = outs[4]
    return np.concatenate([box, cls, angle], axis=1).astype(np.float32)


_core._assemble_rknn_outputs = _assemble_rknn_outputs_fast
_core._decode_head = _decode_head_fast


# ════════════════════════════════════════════════════════════
# 快速 letterbox — cv2.copyMakeBorder 替代手动 numpy slice 填充
# 替代 _rknn_predict_obb 中的 letterbox 部分; 整个 _rknn_predict_obb 重写
# ════════════════════════════════════════════════════════════
_IMGSZ = _core._IMGSZ


def _rknn_predict_obb_fast(gray_hw1, conf_th, iou_th):
    """优化版 _rknn_predict_obb: cv2.copyMakeBorder 加速 letterbox + 减少拷贝"""
    rknn = _core.get_model()
    orig_h, orig_w = gray_hw1.shape[:2]

    # letterbox 到 640 (cv2.resize + copyMakeBorder 两步都在 C++ 层)
    scale = _IMGSZ / max(orig_h, orig_w)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = cv2.resize(gray_hw1[:, :, 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = _IMGSZ - new_w
    pad_h = _IMGSZ - new_h
    top, left = pad_h // 2, pad_w // 2
    bot, right = pad_h - top, pad_w - left
    canvas = cv2.copyMakeBorder(
        resized, top, bot, left, right,
        cv2.BORDER_CONSTANT, value=114,
    )

    # NHWC 输入
    inp = canvas[None, :, :, None]
    outs = rknn.inference(inputs=[inp])
    preds = _assemble_rknn_outputs_fast(outs)

    pts, cids, confs = _obb_nms_cv2(preds, conf_th, iou_th)
    if len(pts) == 0:
        return pts, cids, confs

    # 反 letterbox: (640 坐标) → (原图坐标)
    pts[..., 0] = (pts[..., 0] - left) / scale
    pts[..., 1] = (pts[..., 1] - top) / scale
    return pts, cids, confs


_core._rknn_predict_obb = _rknn_predict_obb_fast


def run_inference_runtime(
    image_rgb: np.ndarray,
    conf_th: float,
    iou_th: float,
    skip_draw: bool = True,
) -> dict[str, Any]:
    """运行时 conf/iou; skip_draw=True 跳过 PIL/cv2 画框, UI 自己用 ObbOverlay 渲染

    Returns:
        annotated_rgb 在 skip_draw=True 时是原图; detections 自带 points 字段.
    """
    _core.PRESETS["__runtime__"] = {
        "conf": float(conf_th),
        "iou": float(iou_th),
        "desc": "runtime",
    }
    return _core.run_inference(image_rgb, mode="__runtime__", skip_draw=skip_draw)


def paint_annotations(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """给定原图 + detections(含 points/class_eng/conf/merged), 调用原 PIL 画框函数生成带框图.

    导出用 — UI overlay 模式下 annotated_rgb 是原图, 导出时需要带框版本.
    """
    if not detections:
        return image_rgb
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    boxes = [np.array(d["points"], dtype=np.float32) for d in detections]
    cls_names_eng = [d.get("class_eng", "") for d in detections]
    confs = [float(d.get("conf", d.get("confidence", 0.0))) for d in detections]
    sizes = [int(d.get("merged", 1)) for d in detections]
    annotated_bgr = _core.draw_annotations(img_bgr, boxes, cls_names_eng, confs, sizes)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def release_model() -> None:
    """释放模型资源 (退出时调用)"""
    try:
        model = getattr(_core, "_MODEL", None)
        if model is None:
            return
        if _core.BACKEND == "rknn":
            if hasattr(model, "release"):
                model.release()
        _core._MODEL = None
    except Exception:
        pass
