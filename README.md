# 烟草异物智能检测系统

面向烟草工厂车间 RK3588 ARM64 Linux 离线工控环境的单机异物检测系统。
集成 **PySide6 GUI + RKNN NPU 推理 + 大恒工业相机 + SQLite 审计日志**, 实现 10 类烟草异物(螺帽 / 细杆 / 石块红砖 / 树叶 / 金属碎片 / 金属 / 棉絮 / 透明塑料绳 / 麻绳 / 黑发)的实时 OBB 旋转框检测与可视化。

---

## 1. 部署根路径

```
/root/kaohe/tobacco_detection_system/
```

所有命令均假设 **当前工作目录在该根路径下** 执行.

---

## 2. 功能概览

| 模块 | 说明 |
|------|------|
| 启动闪屏 | 模型加载 / 数据库初始化进度提示, 完成后跳登录页 |
| 操作员登录 | 用户名 + 密码 + 离线 4 位验证码, 支持注册 / 改密 / 上传头像 |
| 检测主界面 | 1600×900 深色工业风, 4 栏布局, 实时 conf / iou 滑块联动 |
| 4 种数据源 | 单图 / 视频 / 文件夹批量 / 大恒工业相机 |
| OBB 渲染 | QGraphicsView + draw_annotations 中文标签 + 类别配色 |
| 结果过滤 | 10 类异物 CheckBox 实时过滤, 类别统计纯数字 |
| 导出 | 一键导出 CSV + 带框图片 (格式写死) |
| 个人中心 | 圆形头像 + 改密 + 注册 + **历史检测记录查询** + 退出登录 |
| 审计日志 | 每次启动 / 停止检测自动入库, 关联用户名 + 时间 + 类别统计 |
| 资源释放 | closeEvent 自动停止线程 / 关闭相机 / 释放 RKNN / 关闭数据库 |

---

## 3. 默认账号

| 字段 | 值 |
|------|------|
| 用户名 | `operator` |
| 密码 | `123456` |

首次启动会自动创建. 可在 **个人中心 → 修改信息** 改密.

---

## 4. 部署流程

### 4.1 新环境从零部署

```bash
cd /root/kaohe/tobacco_detection_system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# (RK3588 上还需手工安装大恒相机 SDK: gxipy)
python main.py
```

### 4.2 RK3588 现有虚拟环境增量部署 (推荐)

板端已有 `~/gaoyifan_yolo_rknn/RK3588_部署/venv`, 预装了
numpy / opencv (含 headless) / shapely / Pillow / rknn-toolkit-lite2 / gxipy.
本项目只需增量安装 PySide6:

```bash
cd /root/kaohe/tobacco_detection_system
source ~/gaoyifan_yolo_rknn/RK3588_部署/venv/bin/activate
pip install -r requirements-rk3588.txt
python main.py
```

> 启动后 5 秒内会自动: 初始化数据库 → 加载 best.rknn → 关闭闪屏 → 弹出登录窗.

### 4.3 Windows 桌面调试 (跨平台验证 · PT 模式)

同一套代码内置 **PT / RKNN 双后端自动切换**: 板端找到 `best.rknn` + `rknnlite` 走 NPU,
桌面 (Windows/无 NPU) 自动回退 **PT (PyTorch)**。用于在 Windows 上验证跨平台运行。

**先决条件 (两样):**

| 准备 | 放到 | 说明 |
|------|------|------|
| 改版灰度 `ultralytics` 源码 | `./ultralytics/` (含 `pyproject.toml`) | 模型是单通道灰度 OBB, 必须用改版源码, 标准 PyPI 版会因通道不符出错 |
| `best.pt` 权重 | `model/best.pt` | PT 模式用 `.pt`, 不是 `.rknn` |

**一键安装 (在项目根目录执行):**

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements-windows.txt
python main.py
```

`-e ./ultralytics` 会把你的改版源码装成 ultralytics 包(优先生效)并自动拉齐
torch / numpy / opencv / Pillow 等全部依赖 (默认 **CPU 版**)。启动终端打印
`[inference] 后端: PT  设备: CPU` 即正常。

> - **GPU 加速** (有 N 卡): 装完后用本机 CUDA 版替换 CPU torch ——
>   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade`
>   (`cuXXX` 选 ≤ `nvidia-smi` 显示的 CUDA 版本); 成功后显示 `设备: GPU (显卡名)`。
> - **工业相机**: gxipy 不在 PyPI, 需去大恒图像官网装 **Galaxy SDK (Windows 版)** 并勾选
>   "Python (gxipy)" 组件; 不装则相机模式提示"相机不可用", 不影响照片/视频/文件夹检测。
> - 可验证: UI 全流程 + 照片/视频/文件夹检测 (CPU 几秒/张属正常); 验不了: NPU 加速。

---

## 5. 启动命令速查

| 用途 | 命令 |
|------|------|
| 启动主程序 | `python main.py` |
| 仅查看日志 | `tail -f logs/app_$(date +%Y%m%d).log` |
| 清空数据库 (慎用) | `rm db/tobacco.db && python main.py` |

---

## 6. 异常排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 闪屏报「模型加载失败」 | `model/best.rknn` 缺失或不可读 | 检查 `ls -lh model/best.rknn`, 重新拷贝 |
| 闪屏报「数据库初始化失败」 | `db/` 目录无写权限 | `chmod -R u+rw db/` |
| 推理后端显示 `pt` | RK3588 上 rknn-toolkit-lite2 未生效 | 板端 `python -c "from rknnlite.api import RKNNLite"` 验证 |
| 工业相机模式提示「相机不可用」 | gxipy 未安装 / USB 未连 | `python -c "import gxipy"` 验证, 检查 `lsusb` 是否识别大恒设备 |
| 检测框中文显示为方块 | 字体未生效 | 检查 `resources/fonts/font.ttc` 存在 |
| GUI 无法启动 / 提示 Xlib | RK3588 无图形环境 | 确认已登录桌面 (Wayland/X11) 或设置 `DISPLAY=:0` |
| 登录提示「验证码错误」 | 验证码为大小写不敏感, 但需 4 位完整输入 | 点 ⟳ 刷新重试 |
| 历史记录为空 | 当前账号无历史 / 未点过开始检测 | 切到检测模式跑一次后再查 |
| 导出无响应 | 选择目录无写权限 | 换个目录 / `chmod` |

---

## 7. 工程目录

```
tobacco_detection_system/
├── main.py                    # 入口
├── config.py                  # 全局配置 (颜色/路径/默认参数)
├── requirements.txt           # 完整依赖
├── requirements-rk3588.txt    # RK3588 增量依赖
├── requirements-windows.txt   # Windows 桌面调试 (PT 模式) 依赖
├── inference/
│   ├── __init__.py           # monkey-patch 路径的 wrapper
│   └── inference.py          # 复用的 PT/RKNN 双模式推理核心 (不改源)
├── ui/
│   ├── splash_screen.py      # 启动闪屏
│   ├── login_window.py       # 登录 + 注册 + 改密
│   ├── widgets.py            # 通用控件 (Captcha/Table/RightPanel/History/Profile)
│   └── main_window.py        # 主窗口
├── core/
│   ├── app_state.py          # 模块级单例 state
│   ├── source.py             # Photo/Video/Folder/CameraSource 抽象
│   ├── detector.py           # DetectionWorker + FolderBatchWorker (QThread)
│   ├── camera_manager.py     # 大恒相机封装
│   └── exporter.py           # CSV/图 导出
├── db/
│   ├── init.sql              # users + detection_logs schema
│   ├── db_helper.py          # DbHelper 单例
│   └── tobacco.db            # 运行时生成
├── utils/
│   ├── common.py             # logger / 密码哈希 / 文件枚举
│   ├── captcha.py            # PIL 离线验证码
│   └── overlay.py            # OverlayDispatcher (obb/det/seg)
├── model/best.rknn           # 模型权重
├── resources/fonts/font.ttc  # 中文字体
├── assets/                   # logo 等静态资源 (按需)
└── logs/                     # 运行日志 (自动生成)
```

---

## 8. 设计要点

- **线程**: 所有 I/O / 推理 / 导出走 QThread, 主线程仅信号槽; 停止用 `_running` 标志位, 不暴力 terminate.
- **状态**: 跨模块共享通过 `core.app_state.state` 单例, UI 只读不直写.
- **路径解耦**: `inference/inference.py` 源码硬编码了 G2 训练目录路径, 在 `inference/__init__.py` 通过 monkey-patch (`_core.RKNN_PATH = ...`) 重定向到本项目 `model/` 目录, 不修改源文件.
- **推理阈值**: 滑块值通过临时插入 `PRESETS["__runtime__"]` 实现运行时生效, 无需改 inference.py.
- **审计**: 每次 _on_start / _on_stop 都对 `detection_logs` 表 insert + finalize, 关联到当前 `state.username`.

---

## 9. 版权

© 2026 烟草异物智能检测系统 · RK3588 部署版
