"""SQLite 操作封装 — 用户、检测日志的 CRUD"""
from __future__ import annotations

import json
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
        self._migrate_schema()      # 给旧库补 role/must_change_pwd/active/full_name/emp_id 列
        self._ensure_admin()        # 确保存在一个管理员账号
        try:
            self._close_orphan_batches()  # 收尾上次没结束就退出/崩溃残留的 running 批次
        except Exception as e:
            log.error("收尾残留批次失败: %s", e)
        try:
            self.purge_old(15)      # 只保留最近半个月的检测/报警/批次
        except Exception as e:
            log.error("清理过期数据失败: %s", e)

    def _close_orphan_batches(self) -> None:
        """启动时把残留的 running 批次收尾(上次没点结束批次就退/崩)。
        否则登录时 _refresh_batch_ui 会把「开始批次」置灰、点不动。"""
        with self._conn() as c:
            c.execute(
                "UPDATE batches SET status='done', "
                "end_time=COALESCE(end_time, datetime('now','localtime')) "
                "WHERE status='running'"
            )

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

    def _migrate_schema(self) -> None:
        """给已存在的旧 users 表补新列(CREATE TABLE IF NOT EXISTS 不会改旧表)."""
        cols_def = {
            "role": "TEXT DEFAULT 'worker'",
            "must_change_pwd": "INTEGER DEFAULT 0",
            "active": "INTEGER DEFAULT 1",
            "full_name": "TEXT DEFAULT ''",
            "emp_id": "TEXT DEFAULT ''",
            "gender": "TEXT DEFAULT ''",
        }
        try:
            with self._conn() as c:
                have = {r["name"] for r in c.execute("PRAGMA table_info(users)").fetchall()}
                for col, ddl in cols_def.items():
                    if col not in have:
                        c.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
                        log.info("users 表补列: %s", col)
        except Exception as e:
            log.error("用户表迁移失败: %s", e)

    def _ensure_admin(self) -> None:
        """确保至少有一个管理员账号(首次部署预置 admin / admin123, 首登须改密)."""
        try:
            with self._conn() as c:
                n = c.execute("SELECT COUNT(*) AS n FROM users WHERE role='admin'").fetchone()["n"]
            if n == 0:
                self.add_user("admin", "admin123", role="admin", must_change=True,
                              full_name="管理员")
                log.info("已预置默认管理员 admin (首登须改密)")
        except Exception as e:
            log.error("预置管理员失败: %s", e)

    # ── 用户表 ────────────────────────────────────────
    def add_user(self, username: str, password: str, avatar_path: str = "",
                 role: str = "worker", must_change: bool = False,
                 full_name: str = "", emp_id: str = "", gender: str = "") -> bool:
        if not username or not password:
            return False
        if role not in ("worker", "technician", "admin"):
            role = "worker"
        digest, salt = hash_password(password)
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO users(username, password_hash, salt, avatar_path, "
                    "role, must_change_pwd, full_name, emp_id, gender) VALUES (?,?,?,?,?,?,?,?,?)",
                    (username, digest, salt, avatar_path, role,
                     1 if must_change else 0, full_name, emp_id, gender),
                )
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            log.error("创建用户失败: %s", e)
            return False

    def delete_user(self, username: str) -> bool:
        """删除账号(管理员操作, 调用前应已校验管理员密码)。"""
        with self._conn() as c:
            c.execute("DELETE FROM users WHERE username=?", (username,))
        return True

    def update_user_info(self, username: str, emp_id: str, full_name: str,
                         role: str | None = None, gender: str | None = None) -> bool:
        """编辑账号信息(工号/姓名/性别/角色)。管理员用。"""
        with self._conn() as c:
            c.execute("UPDATE users SET emp_id=?, full_name=? WHERE username=?",
                      (emp_id, full_name, username))
            if gender is not None:
                c.execute("UPDATE users SET gender=? WHERE username=?", (gender, username))
            if role in ("worker", "technician", "admin"):
                c.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        return True

    # ── 角色 / 账号管理(管理员用)──────────────────────
    def get_role(self, username: str) -> str:
        u = self.get_user(username)
        return (u or {}).get("role", "worker")

    def list_users(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, username, emp_id, full_name, gender, role, active, create_time "
                "FROM users ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_role(self, username: str, role: str) -> bool:
        if role not in ("worker", "technician", "admin"):
            return False
        with self._conn() as c:
            c.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        return True

    def set_active(self, username: str, active: bool) -> bool:
        with self._conn() as c:
            c.execute("UPDATE users SET active=? WHERE username=?",
                      (1 if active else 0, username))
        return True

    def admin_reset_password(self, username: str, new_pw: str) -> bool:
        """管理员重置密码(不校验旧密码),并置首登须改密."""
        digest, salt = hash_password(new_pw)
        with self._conn() as c:
            c.execute("UPDATE users SET password_hash=?, salt=?, must_change_pwd=1 "
                      "WHERE username=?", (digest, salt, username))
        return True

    def clear_must_change(self, username: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE users SET must_change_pwd=0 WHERE username=?", (username,))

    # ── 审计日志 ──────────────────────────────────────
    def add_audit(self, username: str, role: str, action: str, detail: str = "") -> None:
        """记录关键操作(登录/改参数/换模型/账号操作)。失败不抛, 不影响主流程。"""
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO audit_logs(username, role, action, detail) VALUES (?,?,?,?)",
                    (username, role, action, detail),
                )
        except Exception as e:
            log.error("写审计失败: %s", e)

    def query_audit(self, limit: int = 500, since: str | None = None,
                    until: str | None = None, username: str | None = None,
                    role: str | None = None) -> list[dict[str, Any]]:
        conds, args = [], []
        if since:
            conds.append("time >= ?"); args.append(since)
        if until:
            conds.append("time <= ?"); args.append(until)
        if username:
            conds.append("username LIKE ?"); args.append(f"%{username}%")
        if role:
            conds.append("role = ?"); args.append(role)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT time, username, role, action, detail FROM audit_logs "
                f"{where} ORDER BY id DESC LIMIT ?", tuple(args)
            ).fetchall()
        return [dict(r) for r in rows]

    def distinct_audit_users(self) -> list[str]:
        """审计日志里出现过的用户名(供下拉筛选)。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT username FROM audit_logs "
                "WHERE username IS NOT NULL AND username != '' ORDER BY username").fetchall()
        return [r["username"] for r in rows]

    def distinct_operators(self) -> list[str]:
        """批次里出现过的当班人(供批次追溯下拉筛选)。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT operator FROM batches "
                "WHERE operator IS NOT NULL AND operator != '' ORDER BY operator").fetchall()
        return [r["operator"] for r in rows]

    def distinct_alarm_operators(self) -> list[str]:
        """报警记录里出现过的当班人(供报警记录下拉筛选)。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT username FROM alarms "
                "WHERE username IS NOT NULL AND username != '' ORDER BY username").fetchall()
        return [r["username"] for r in rows]

    # ── 报表聚合(从 detection_logs 统计)────────────────
    def report_summary(self, since: str | None = None, until: str | None = None,
                        batch_id: str | None = None, worker_only: bool = True,
                        only_classes: set[str] | None = None) -> dict[str, Any]:
        """汇总检测量/异物分布/模式分布。
        since/until 形如 '2026-06-04 00:00:00'(闭区间); batch_id 指定则只统计该批次时间窗内。
        only_classes 指定则只统计这些类别: runs=含这些类的检测次数, total/by_class 也只算这些类。
        worker_only=True: 只算工人角色的检测(技术员是调试, 不计入生产报表)。"""
        if batch_id:
            b = next((x for x in self.list_batches(500) if x["batch_id"] == batch_id), None)
            if b:
                since = b["start_time"]
                until = b["end_time"] or None
        conds, args = [], []
        if since:
            conds.append("start_time >= ?"); args.append(since)
        if until:
            conds.append("start_time <= ?"); args.append(until)
        if worker_only:
            conds.append("username IN (SELECT username FROM users WHERE role='worker')")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT detect_mode, total_targets, result_summary FROM detection_logs {where}",
                tuple(args),
            ).fetchall()
        by_mode: dict[str, int] = {}
        by_class: dict[str, int] = {}
        total_targets = 0
        runs = 0
        for r in rows:
            try:
                summ = {k: int(v) for k, v in json.loads(r["result_summary"] or "{}").items()
                        if isinstance(v, (int, float))}
            except Exception:
                summ = {}
            if only_classes is not None:
                # 仅统计指定类别: 只有含这些类的检测才计入 runs/total
                summ = {k: v for k, v in summ.items() if k in only_classes}
                if not summ:
                    continue
                runs += 1
                total_targets += sum(summ.values())
            else:
                runs += 1
                total_targets += int(r["total_targets"] or 0)
            by_mode[r["detect_mode"]] = by_mode.get(r["detect_mode"], 0) + 1
            for k, v in summ.items():
                by_class[k] = by_class.get(k, 0) + v
        return {
            "runs": runs,
            "total_targets": total_targets,
            "by_mode": by_mode,
            "by_class": dict(sorted(by_class.items(), key=lambda x: -x[1])),
        }

    # ── 报警记录 + 处置闭环 ──────────────────────────
    def add_alarm(self, username: str, class_eng: str, class_cn: str,
                  severity: str = "severe", diverted: bool = False,
                  image_path: str = "", batch_id: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO alarms(username, class_eng, class_cn, severity, "
                "diverted, image_path, batch_id) VALUES (?,?,?,?,?,?,?)",
                (username, class_eng, class_cn, severity,
                 1 if diverted else 0, image_path, batch_id),
            )
            return cur.lastrowid

    def handle_alarm(self, alarm_id: int, action: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE alarms SET handled=1, action=? WHERE id=?", (action, alarm_id))

    def query_alarms(self, limit: int = 200, only_unhandled: bool = False,
                     worker_only: bool = True, since: str | None = None,
                     until: str | None = None, batch_id: str | None = None) -> list[dict[str, Any]]:
        conds, args = [], []
        if only_unhandled:
            conds.append("handled=0")
        if batch_id:
            conds.append("batch_id=?"); args.append(batch_id)
        else:
            if since:
                conds.append("time >= ?"); args.append(since)
            if until:
                conds.append("time <= ?"); args.append(until)
        if worker_only:
            conds.append("username IN (SELECT username FROM users WHERE role='worker')")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM alarms {where} ORDER BY id DESC LIMIT ?", tuple(args)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_alarms(self, since: str | None = None, until: str | None = None,
                     batch_id: str | None = None, worker_only: bool = True) -> int:
        """统计某范围/批次内的报警条数(供报表与报警记录交叉关联)。"""
        conds, args = [], []
        if batch_id:
            conds.append("batch_id=?"); args.append(batch_id)
        else:
            if since:
                conds.append("time >= ?"); args.append(since)
            if until:
                conds.append("time <= ?"); args.append(until)
        if worker_only:
            conds.append("username IN (SELECT username FROM users WHERE role='worker')")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        with self._conn() as c:
            row = c.execute(f"SELECT COUNT(*) AS n FROM alarms {where}", tuple(args)).fetchone()
        return int(row["n"]) if row else 0

    def count_recent_severe(self, seconds: int) -> int:
        """最近 seconds 秒内的严重报警数(用于频次升级判定)。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM alarms WHERE severity='severe' "
                "AND time >= datetime('now','localtime', ?)",
                (f"-{int(seconds)} seconds",),
            ).fetchone()
        return int(row["n"]) if row else 0

    # ── 批次追溯 ──────────────────────────────────────
    def get_running_batch(self) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM batches WHERE status='running' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def start_batch(self, batch_id: str, variety: str, operator: str,
                    model_name: str, conf: float, iou: float) -> int:
        """开批次(先结掉未结束的)。快照当前配置(模型+阈值)。返回行 id。"""
        self.end_batch()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO batches(batch_id, variety, operator, model_name, conf, iou) "
                "VALUES (?,?,?,?,?,?)",
                (batch_id, variety, operator, model_name, conf, iou),
            )
            return cur.lastrowid

    def end_batch(self) -> bool:
        """结束当前 running 批次, 汇总报警数 + 检出目标数。无 running 则不动。"""
        b = self.get_running_batch()
        if not b:
            return False
        with self._conn() as c:
            n_alarm = c.execute("SELECT COUNT(*) AS n FROM alarms WHERE batch_id=?",
                                (b["batch_id"],)).fetchone()["n"]
            tgt = c.execute(
                "SELECT COALESCE(SUM(total_targets),0) AS s FROM detection_logs "
                "WHERE start_time >= ?", (b["start_time"],)).fetchone()["s"]
            c.execute(
                "UPDATE batches SET end_time=datetime('now','localtime'), status='done', "
                "alarm_count=?, total_targets=? WHERE id=?",
                (int(n_alarm), int(tgt), b["id"]))
        return True

    def list_batches(self, limit: int = 200, since: str | None = None,
                     until: str | None = None, batch_kw: str | None = None,
                     operator: str | None = None) -> list[dict[str, Any]]:
        conds, args = [], []
        if since:
            conds.append("start_time >= ?"); args.append(since)
        if until:
            conds.append("start_time <= ?"); args.append(until)
        if batch_kw:
            conds.append("batch_id LIKE ?"); args.append(f"%{batch_kw}%")
        if operator:
            conds.append("operator = ?"); args.append(operator)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM batches {where} ORDER BY id DESC LIMIT ?", tuple(args)).fetchall()
        return [dict(r) for r in rows]

    def delete_user_batches(self, operator: str) -> int:
        """删除某操作员的全部批次(及挂在这些批次上的报警)。返回删除批次数。"""
        with self._conn() as c:
            bids = [r["batch_id"] for r in c.execute(
                "SELECT batch_id FROM batches WHERE operator=?", (operator,)).fetchall()]
            for bid in bids:
                c.execute("DELETE FROM alarms WHERE batch_id=?", (bid,))
            cur = c.execute("DELETE FROM batches WHERE operator=?", (operator,))
            return cur.rowcount

    def purge_old(self, days: int = 15) -> dict[str, int]:
        """只保留最近 days 天的检测/报警/批次记录(默认半个月), 自动清理更早的。"""
        cutoff = f"-{int(days)} days"
        out: dict[str, int] = {}
        with self._conn() as c:
            out["detection_logs"] = c.execute(
                "DELETE FROM detection_logs WHERE start_time < datetime('now','localtime',?)",
                (cutoff,)).rowcount
            out["alarms"] = c.execute(
                "DELETE FROM alarms WHERE time < datetime('now','localtime',?)", (cutoff,)).rowcount
            out["batches"] = c.execute(
                "DELETE FROM batches WHERE start_time < datetime('now','localtime',?) AND status='done'",
                (cutoff,)).rowcount
        return out

    def batch_alarms(self, batch_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM alarms WHERE batch_id=? ORDER BY id DESC",
                             (batch_id,)).fetchall()
        return [dict(r) for r in rows]

    def list_varieties(self, limit: int = 50) -> list[str]:
        """历史批次用过的品种(开批次时下拉补全, 工人少打字)。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT variety FROM batches WHERE variety<>'' "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r["variety"] for r in rows]

    def clear_demo_data(self) -> dict[str, int]:
        """清空演示/测试数据: 检测日志 + 报警 + 批次(不动账号/审计)。返回各表删除条数。"""
        out: dict[str, int] = {}
        with self._conn() as c:
            for t in ("detection_logs", "alarms", "batches"):
                cur = c.execute(f"DELETE FROM {t}")
                out[t] = int(cur.rowcount)
        return out

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
            # 改完密码即清掉"首登须改密"标志, 否则每次登录都重复提示(原 bug)
            c.execute(
                "UPDATE users SET password_hash=?, salt=?, must_change_pwd=0 WHERE username=?",
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
