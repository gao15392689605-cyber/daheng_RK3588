"""SQLite 操作封装 — 用户、检测日志的 CRUD"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import DB_INIT_SQL, DB_PATH, DEFAULT_PASSWORD, DEFAULT_USERNAME
from utils.common import get_logger, hash_password, safe_json_dumps, verify_password

log = get_logger("db")


class DbHelper:
    """模块级单例 — 通过 DbHelper.instance() 获取"""

    _instance: "DbHelper | None" = None

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._ensure_default_user()

    @classmethod
    def instance(cls) -> "DbHelper":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            sql = DB_INIT_SQL.read_text(encoding="utf-8")
            with self._conn() as c:
                c.executescript(sql)
            log.info("数据库初始化完成: %s", self.db_path)
        except Exception as e:
            log.error("数据库初始化失败: %s", e)
            raise

    def _ensure_default_user(self) -> None:
        if self.get_user(DEFAULT_USERNAME) is None:
            self.add_user(DEFAULT_USERNAME, DEFAULT_PASSWORD)
            log.info("已创建默认账号: %s", DEFAULT_USERNAME)

    # ── 用户表 ────────────────────────────────────────
    def add_user(self, username: str, password: str, avatar_path: str = "") -> bool:
        if not username or not password:
            return False
        digest, salt = hash_password(password)
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO users(username, password_hash, salt, avatar_path) VALUES (?,?,?,?)",
                    (username, digest, salt, avatar_path),
                )
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            log.error("创建用户失败: %s", e)
            return False

    def get_user(self, username: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM users WHERE username=?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def verify_user(self, username: str, password: str) -> bool:
        u = self.get_user(username)
        if not u:
            return False
        return verify_password(password, u["password_hash"], u["salt"])

    def update_password(self, username: str, old_pw: str, new_pw: str) -> bool:
        if not self.verify_user(username, old_pw):
            return False
        digest, salt = hash_password(new_pw)
        with self._conn() as c:
            c.execute(
                "UPDATE users SET password_hash=?, salt=? WHERE username=?",
                (digest, salt, username),
            )
        return True

    def update_avatar(self, username: str, avatar_path: str) -> bool:
        with self._conn() as c:
            c.execute(
                "UPDATE users SET avatar_path=? WHERE username=?",
                (avatar_path, username),
            )
        return True

    # ── 检测日志 ──────────────────────────────────────
    def insert_log(
        self,
        username: str,
        detect_mode: str,
        model_name: str,
        conf_threshold: float,
        iou_threshold: float,
        file_path: str = "",
        total_targets: int = 0,
        result_summary: dict | None = None,
        status: str = "success",
    ) -> int:
        summary = safe_json_dumps(result_summary or {})
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO detection_logs(
                    username, detect_mode, file_path, model_name,
                    conf_threshold, iou_threshold, total_targets,
                    result_summary, status
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (username, detect_mode, file_path, model_name,
                 conf_threshold, iou_threshold, total_targets, summary, status),
            )
            return int(cur.lastrowid)

    def finalize_log(
        self,
        log_id: int,
        total_targets: int,
        result_summary: dict,
        status: str = "success",
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                UPDATE detection_logs
                SET total_targets=?, result_summary=?, status=?,
                    end_time=datetime('now','localtime')
                WHERE id=?
                """,
                (total_targets, safe_json_dumps(result_summary), status, log_id),
            )

    def query_logs(self, username: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT * FROM detection_logs
                WHERE username=?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (username, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_logs(self, username: str) -> int:
        """清空指定用户的所有检测日志, 返回删除条数"""
        with self._conn() as c:
            cur = c.execute("DELETE FROM detection_logs WHERE username=?", (username,))
            return int(cur.rowcount)
