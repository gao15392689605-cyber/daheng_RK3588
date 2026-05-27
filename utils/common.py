"""通用工具: 日志、密码哈希、JSON 安全序列化"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from config import LOG_DIR


_LOGGER_INITED = False


def get_logger(name: str = "tobacco") -> logging.Logger:
    """返回 logger, 中文输出到文件与控制台.

    handler 加到 root logger, 所有子 logger 通过 propagate 复用,
    保证 get_logger("db") / get_logger("main") 等都能正常打印.
    """
    global _LOGGER_INITED
    if not _LOGGER_INITED:
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

        _LOGGER_INITED = True
    return logging.getLogger(name)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """SHA256+salt 哈希; salt 为 None 时随机生成"""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()
    return digest, salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    digest, _ = hash_password(password, salt)
    return secrets.compare_digest(digest, password_hash)


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_image_file(path: str | Path) -> bool:
    from config import IMAGE_EXTS
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video_file(path: str | Path) -> bool:
    from config import VIDEO_EXTS
    return Path(path).suffix.lower() in VIDEO_EXTS


def list_media_files(folder: str | Path, recursive: bool = False) -> list[Path]:
    """递归/单层列出图片+视频文件"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    glob = folder.rglob("*") if recursive else folder.glob("*")
    out: list[Path] = []
    for p in glob:
        if p.is_file() and (is_image_file(p) or is_video_file(p)):
            out.append(p)
    return sorted(out)


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
