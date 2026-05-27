"""输入源抽象 — 统一 photo / video / folder / camera 的读取接口

每个 source 暴露:
    open() -> bool
    read() -> tuple[bool, np.ndarray | None]  # 返回 RGB 帧
    close() -> None
    total: int   # 视频/文件夹总数, 单图为 1, 相机为 -1
    info() -> str
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BaseSource:
    total: int = 0

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> str:
        return ""


class PhotoSource(BaseSource):
    """单图源"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.total = 1
        self._read = False

    def open(self) -> bool:
        return Path(self.file_path).exists()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._read:
            return False, None
        bgr = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, None
        self._read = True
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def info(self) -> str:
        return Path(self.file_path).name


class VideoSource(BaseSource):
    """视频文件源 — 顺序读帧"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.cap: cv2.VideoCapture | None = None
        self.total = 0
        self.fps: float = 25.0

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            return False
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return False, None
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def info(self) -> str:
        return f"{Path(self.file_path).name} ({self.total}帧/{self.fps:.1f}fps)"


class FolderSource(BaseSource):
    """文件夹批量源 — 顺序遍历图片"""

    def __init__(self, folder: str, files: list[Path]) -> None:
        self.folder = folder
        self.files = files
        self.total = len(files)
        self._idx = 0

    def open(self) -> bool:
        return self.total > 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._idx >= self.total:
            return False, None
        fp = str(self.files[self._idx])
        self._idx += 1
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, None
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def current_file(self) -> str:
        idx = max(0, self._idx - 1)
        return str(self.files[idx]) if idx < self.total else ""

    def info(self) -> str:
        return f"{self.folder} ({self.total}个文件)"


class CameraSource(BaseSource):
    """工业相机源 — 流式连续推理"""

    def __init__(self, camera_manager: Any) -> None:
        self.cm = camera_manager
        self.total = -1

    def open(self) -> bool:
        return self.cm.open()

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.cm.read()

    def close(self) -> None:
        self.cm.close()

    def info(self) -> str:
        return self.cm.info()


class CameraSnapSource(BaseSource):
    """相机抓拍 — 只读 1 帧, 行为同单图"""

    def __init__(self, camera_manager: Any) -> None:
        self.cm = camera_manager
        self.total = 1
        self._taken = False

    def open(self) -> bool:
        return self.cm.open()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._taken:
            return False, None
        # 等待相机出一帧 (最多 3 次尝试)
        for _ in range(3):
            ok, frame = self.cm.read()
            if ok and frame is not None:
                self._taken = True
                return True, frame
            import time as _t
            _t.sleep(0.05)
        return False, None

    def close(self) -> None:
        # 不彻底关相机, 只停流, 下次 open 快速 stream_on
        self.cm.close()

    def info(self) -> str:
        return f"相机抓拍 - {self.cm.info()}"
