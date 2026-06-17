"""管理员管理面板 — 账号管理 / 批次追溯 / 报警记录 / 审计日志 / 报表统计。

管理员登录后进这里(不进检测界面)。职责单一:管人、管账、看报告,不碰检测参数。
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from config import (
    AUDIT_ACTION_CN, COLOR_BG_MAIN, COLOR_BG_SIDE, COLOR_BORDER,
    COLOR_BTN_PRIMARY, COLOR_TABLE_ALT, COLOR_TEXT, COLOR_TEXT_DIM, SEVERE_CLASSES, btn_qss,
)
from core.app_state import state
from db.db_helper import DbHelper
from inference import CN_NAMES
from ui.chart import BarChart, PieChart

# 严重异物的中文名集合(报表"严重/其他"区分用)
SEVERE_CN: set[str] = {CN_NAMES.get(e, e) for e in SEVERE_CLASSES}
from utils.common import get_logger

log = get_logger("admin")

_ROLE_CN = {"worker": "工人", "technician": "技术人员", "admin": "管理员"}
_ROLE_KEYS = ["worker", "technician", "admin"]
_GREEN = QColor("#2DB95B")
_RED = QColor("#E66662")

# 科技感配色: 青色强调 + 角色徽章配色
_ACCENT = "#21D4C8"
_ROLE_COLOR = {"worker": "#2D8CFE", "technician": "#21D4C8", "admin": "#FFD23F"}

# 弹窗统一样式: 输入框/下拉清晰可辨, 下拉箭头青色三角, 弹出列表选中项青底深字
_DIALOG_QSS = (
    f"QDialog{{background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};}}"
    f"QLabel{{color:{COLOR_TEXT}; background:transparent;}}"
    f"QLineEdit,QComboBox{{background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
    f"border:1px solid {COLOR_BORDER}; border-radius:6px; padding:7px 10px; min-height:24px;}}"
    f"QLineEdit:focus,QComboBox:focus{{border:1px solid {_ACCENT};}}"
    f"QComboBox::drop-down{{border:none; width:26px;}}"
    f"QComboBox::down-arrow{{image:none; width:0; height:0;"
    f"border-left:5px solid transparent; border-right:5px solid transparent;"
    f"border-top:7px solid {_ACCENT}; margin-right:9px;}}"
    f"QComboBox QAbstractItemView{{background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
    f"border:1px solid {_ACCENT}; outline:none; padding:2px;"
    f"selection-background-color:{_ACCENT}; selection-color:#10141A;}}"
    f"QComboBox QAbstractItemView::item{{min-height:28px; padding:2px 8px;}}"
)


def _audit(action: str, detail: str = "") -> None:
    DbHelper.instance().add_audit(state.username, state.role, action, detail)


def _action_cn(action: str) -> str:
    return AUDIT_ACTION_CN.get(action, action)


class _CreateUserDialog(QDialog):
    """新建账号对话框 — 用户名/姓名/角色/初始密码。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建账号")
        self.setMinimumWidth(360)
        self.setStyleSheet(_DIALOG_QSS)
        form = QFormLayout(self)
        self.emp_id = QLineEdit(); self.emp_id.setPlaceholderText("必填, 如 G001")
        self.username = QLineEdit()
        self.full_name = QLineEdit()
        self.gender = QComboBox()
        for g in ("男", "女"):
            self.gender.addItem(g, g)
        self.password = QLineEdit(); self.password.setText("123456")
        self.role = QComboBox()
        for k in _ROLE_KEYS:
            self.role.addItem(_ROLE_CN[k], k)
        form.addRow("工号", self.emp_id)
        form.addRow("用户名", self.username)
        form.addRow("姓名", self.full_name)
        form.addRow("性别", self.gender)
        form.addRow("角色", self.role)
        form.addRow("初始密码", self.password)
        row = QHBoxLayout()
        ok = QPushButton("创建"); cancel = QPushButton("取消")
        ok.setStyleSheet(btn_qss(COLOR_BTN_PRIMARY)); cancel.setStyleSheet(btn_qss(COLOR_BG_SIDE))
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        row.addStretch(); row.addWidget(cancel); row.addWidget(ok)
        form.addRow(row)

    def values(self) -> dict[str, Any]:
        return {
            "emp_id": self.emp_id.text().strip(),
            "username": self.username.text().strip(),
            "full_name": self.full_name.text().strip(),
            "gender": self.gender.currentData(),
            "password": self.password.text(),
            "role": self.role.currentData(),
        }


class _EditUserDialog(QDialog):
    """编辑账号信息 — 工号/姓名/角色(用户名不可改, 作主键)。"""

    def __init__(self, user: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"编辑账号 — {user['username']}")
        self.setMinimumWidth(360)
        self.setStyleSheet(_DIALOG_QSS)
        form = QFormLayout(self)
        uname = QLabel(user["username"])
        self.emp_id = QLineEdit(user.get("emp_id", "") or "")
        self.full_name = QLineEdit(user.get("full_name", "") or "")
        self.gender = QComboBox()
        for g in ("男", "女"):
            self.gender.addItem(g, g)
        cur_g = user.get("gender", "") or ""
        gi = self.gender.findData(cur_g)
        if gi >= 0:
            self.gender.setCurrentIndex(gi)
        self.role = QComboBox()
        for k in _ROLE_KEYS:
            self.role.addItem(_ROLE_CN[k], k)
        cur = user.get("role", "worker")
        self.role.setCurrentIndex(_ROLE_KEYS.index(cur) if cur in _ROLE_KEYS else 0)
        form.addRow("用户名(不可改)", uname)
        form.addRow("工号", self.emp_id)
        form.addRow("姓名", self.full_name)
        form.addRow("性别", self.gender)
        form.addRow("角色", self.role)
        row = QHBoxLayout()
        ok = QPushButton("保存"); cancel = QPushButton("取消")
        ok.setStyleSheet(btn_qss(COLOR_BTN_PRIMARY)); cancel.setStyleSheet(btn_qss(COLOR_BG_SIDE))
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        row.addStretch(); row.addWidget(cancel); row.addWidget(ok)
        form.addRow(row)

    def values(self) -> dict[str, Any]:
        return {
            "emp_id": self.emp_id.text().strip(),
            "full_name": self.full_name.text().strip(),
            "gender": self.gender.currentData(),
            "role": self.role.currentData(),
        }


class AdminPanel(QMainWindow):
    """管理员主窗口。对外接口与 MainWindow 对齐: logout_requested / init_for_user。"""

    logout_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("烟草异物检测系统 — 管理面板")
        self.setMinimumSize(900, 600)   # 可自由缩放, 设个合理下限
        self.resize(1160, 720)
        # 面板底色 + 所有下拉统一清晰样式(青色箭头, 弹出列表选中青底深字)
        self.setStyleSheet(
            f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"QComboBox{{background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:6px; padding:4px 8px; min-height:22px;}}"
            f"QComboBox:focus{{border:1px solid {_ACCENT};}}"
            f"QComboBox::drop-down{{border:none; width:22px;}}"
            f"QComboBox::down-arrow{{image:none; width:0; height:0;"
            f"border-left:5px solid transparent; border-right:5px solid transparent;"
            f"border-top:7px solid {_ACCENT}; margin-right:7px;}}"
            f"QComboBox QAbstractItemView{{background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"border:1px solid {_ACCENT}; outline:none;"
            f"selection-background-color:{_ACCENT}; selection-color:#10141A;}}"
            f"QComboBox QAbstractItemView::item{{min-height:26px; padding:2px 8px;}}")
        self._build()

    # ── 构建 ──
    def _build(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(14, 12, 14, 12); root.setSpacing(12)

        # ── 顶部科技感标题条(左侧青色高亮边 + 六边形图标)──
        header_bar = QFrame()
        header_bar.setStyleSheet(
            f"QFrame{{background:{COLOR_BG_SIDE}; border:1px solid {COLOR_BORDER};"
            f"border-left:3px solid {_ACCENT}; border-radius:8px;}}")
        hb = QHBoxLayout(header_bar); hb.setContentsMargins(18, 12, 16, 12)
        self.header = QLabel("⬢  烟草异物检测 · 管理控制台")
        self.header.setStyleSheet(f"color:{COLOR_TEXT}; font-size:19px; font-weight:bold; border:none;")
        self.who = QLabel("")
        self.who.setStyleSheet(f"color:{_ACCENT}; font-size:13px; border:none;")
        logout = QPushButton("退出登录")
        logout.setStyleSheet(btn_qss("#d9534f", hover="#e66662"))
        logout.clicked.connect(self.logout_requested.emit)
        hb.addWidget(self.header); hb.addSpacing(14); hb.addWidget(self.who)
        hb.addStretch(); hb.addWidget(logout)
        root.addWidget(header_bar)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {COLOR_BORDER}; border-radius:8px;"
            f"top:-1px; background:{COLOR_BG_MAIN};}}"
            f"QTabBar::tab{{background:{COLOR_BG_SIDE}; color:{COLOR_TEXT_DIM};"
            f"padding:8px 22px; margin-right:4px; border:1px solid {COLOR_BORDER};"
            f"border-bottom:none; border-top-left-radius:6px; border-top-right-radius:6px;}}"
            f"QTabBar::tab:selected{{color:{_ACCENT}; background:{COLOR_BG_MAIN};"
            f"border-bottom:2px solid {_ACCENT}; font-weight:bold;}}"
            f"QTabBar::tab:hover{{color:{COLOR_TEXT};}}")
        tabs.addTab(self._build_users_tab(), "账号管理")
        tabs.addTab(self._build_batches_tab(), "批次追溯")
        tabs.addTab(self._build_alarms_tab(), "报警记录")
        tabs.addTab(self._build_audit_tab(), "审计日志")
        tabs.addTab(self._build_report_tab(), "报表统计")
        root.addWidget(tabs, 1)

    def init_for_user(self, username: str) -> None:
        self.who.setText(f"当前: {username}")
        self._refresh_users(); self._refresh_batches(); self._refresh_alarms()
        self._refresh_audit(); self._refresh_report()

    def _bar_button(self, text: str, fn, bg: str = COLOR_BTN_PRIMARY, checkable: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setStyleSheet(btn_qss(bg))
        b.setCheckable(checkable)
        if checkable:
            b.toggled.connect(fn)
        else:
            b.clicked.connect(fn)
        return b

    # ── Tab1 账号管理 ──
    def _build_users_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        # 「改角色」已并入「编辑信息」(编辑里可改角色), 不再单列, 避免两个入口
        for text, fn in (("+ 新建账号", self._create_user), ("编辑信息", self._edit_user),
                         ("停用/启用", self._toggle_active),
                         ("重置密码", self._reset_pwd), ("刷新", self._refresh_users)):
            bar.addWidget(self._bar_button(text, fn))
        bar.addWidget(self._bar_button("删除账号", self._delete_user, bg="#d9534f"))
        bar.addStretch()
        bar.addWidget(QLabel("搜索"))
        self.user_search = QLineEdit()
        self.user_search.setPlaceholderText("工号 / 用户名 / 姓名")
        self.user_search.setFixedWidth(180)
        self.user_search.setStyleSheet(
            f"QLineEdit{{background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:5px 8px;}}")
        self.user_search.textChanged.connect(self._apply_user_filter)
        bar.addWidget(self.user_search)
        lay.addLayout(bar)
        self.user_table = self._make_table(["工号", "用户名", "姓名", "性别", "角色", "状态", "创建时间"])
        lay.addWidget(self.user_table, 1)
        return w

    def _cell_label(self, text: str, color: str):
        from PySide6.QtWidgets import QLabel as _QL
        lb = _QL(text); lb.setAlignment(Qt.AlignCenter)
        lb.setStyleSheet(f"color:{color}; font-weight:bold; background:transparent;")
        return lb

    @staticmethod
    def _emp_sort_key(emp: str):
        """工号自然排序键: 数字段按数值比(G2 < G10), 空工号排最后。"""
        emp = (emp or "").strip()
        chunks = re.findall(r"\d+|\D+", emp)
        norm = [(0, int(c)) if c.isdigit() else (1, c.lower()) for c in chunks]
        return (emp == "", norm)

    def _refresh_users(self) -> None:
        users = DbHelper.instance().list_users()
        users.sort(key=lambda u: self._emp_sort_key(u.get("emp_id", "")))   # 按工号排序
        self._users = users
        self._render_users(self._users)

    def _render_users(self, users: list[dict[str, Any]]) -> None:
        self.user_table.setRowCount(len(users))
        for i, u in enumerate(users):
            active = bool(u.get("active", 1))
            emp = u.get("emp_id") or "—"
            role = u["role"]
            cells = [emp, u["username"], u.get("full_name", ""),
                     u.get("gender", "") or "—",
                     "", "", str(u.get("create_time", ""))]
            for j, val in enumerate(cells):
                self.user_table.setItem(i, j, QTableWidgetItem(val))
            # 角色 / 状态用彩色文字(无控件, 隔行灰下不露黑块)
            self.user_table.setItem(
                i, 4, self._color_item(_ROLE_CN.get(role, role), _ROLE_COLOR.get(role, COLOR_TEXT)))
            self.user_table.setItem(
                i, 5, self._color_item("启用" if active else "停用",
                                       "#2DB95B" if active else "#E66662"))

    def _apply_user_filter(self, text: str) -> None:
        kw = text.strip().lower()
        if not kw:
            self._render_users(getattr(self, "_users", []))
            return
        hit = [u for u in getattr(self, "_users", [])
               if kw in str(u.get("emp_id", "")).lower() or kw in u["username"].lower()
               or kw in (u.get("full_name", "") or "").lower()]
        self._render_users(hit)

    def _delete_user(self) -> None:
        u = self._selected_username()
        if not u:
            return
        if u == state.username:
            QMessageBox.warning(self, "提示", "不能删除自己"); return
        ret = QMessageBox.question(
            self, "删除账号", f"确认永久删除账号【{u}】? 此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        # 二次校验: 管理员重新输入自己的登录密码
        pwd, ok = QInputDialog.getText(
            self, "管理员验证", f"请输入你(管理员 {state.username})的登录密码以确认删除:",
            QLineEdit.Password)
        if not ok:
            return
        if not DbHelper.instance().verify_user(state.username, pwd):
            QMessageBox.warning(self, "密码错误", "密码不正确, 已取消删除"); return
        DbHelper.instance().delete_user(u)
        _audit("user_delete", u)
        self._refresh_users(); self._refresh_audit()
        QMessageBox.information(self, "已删除", f"账号 {u} 已删除")

    def _selected_username(self) -> str | None:
        r = self.user_table.currentRow()
        if r < 0 or self.user_table.item(r, 1) is None:
            QMessageBox.information(self, "提示", "请先在列表里选中一个账号")
            return None
        return self.user_table.item(r, 1).text()

    def _create_user(self) -> None:
        dlg = _CreateUserDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["username"] or not v["password"]:
            QMessageBox.warning(self, "提示", "用户名和初始密码不能为空"); return
        if not v["emp_id"]:
            QMessageBox.warning(self, "提示", "工号必填"); return
        ok = DbHelper.instance().add_user(
            v["username"], v["password"], role=v["role"],
            must_change=False, full_name=v["full_name"], emp_id=v["emp_id"],
            gender=v.get("gender", ""))
        if not ok:
            QMessageBox.warning(self, "提示", "创建失败(用户名可能已存在)"); return
        _audit("user_create", f"{v['emp_id']}/{v['username']}({_ROLE_CN.get(v['role'], v['role'])})")
        self._refresh_users(); self._refresh_audit()
        QMessageBox.information(self, "成功", f"已创建账号 {v['username']}")

    def _edit_user(self) -> None:
        u = self._selected_username()
        if not u:
            return
        user = DbHelper.instance().get_user(u) or {}
        dlg = _EditUserDialog(user, self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["emp_id"]:
            QMessageBox.warning(self, "提示", "工号不能为空"); return
        DbHelper.instance().update_user_info(u, v["emp_id"], v["full_name"], v["role"],
                                             gender=v.get("gender", ""))
        _audit("user_edit", f"{u} 工号={v['emp_id']} 姓名={v['full_name']} 性别={v.get('gender','')} 角色={_ROLE_CN.get(v['role'], v['role'])}")
        self._refresh_users(); self._refresh_audit()
        QMessageBox.information(self, "成功", f"账号 {u} 信息已更新")

    def _toggle_active(self) -> None:
        u = self._selected_username()
        if not u:
            return
        if u == state.username:
            QMessageBox.warning(self, "提示", "不能停用自己"); return
        cur = DbHelper.instance().get_user(u) or {}
        new_active = not bool(cur.get("active", 1))
        DbHelper.instance().set_active(u, new_active)
        _audit("user_active", f"{u} → {'启用' if new_active else '停用'}")
        self._refresh_users(); self._refresh_audit()

    def _reset_pwd(self) -> None:
        u = self._selected_username()
        if not u:
            return
        new_pw, ok = QInputDialog.getText(self, "重置密码", f"给 {u} 设新密码:", QLineEdit.Normal, "123456")
        if not ok or not new_pw:
            return
        DbHelper.instance().admin_reset_password(u, new_pw)
        _audit("user_reset_pwd", u)
        self._refresh_audit()
        QMessageBox.information(self, "成功", f"{u} 密码已重置(其下次登录须改密)")

    # ── Tab 批次追溯 ──
    def _build_batches_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("最近"))
        # 用下拉而非 QSpinBox: 深色主题下 spinbox 上下箭头是黑的看不见
        self.batch_days = QComboBox()
        for label, d in (("今天", 1), ("近3天", 3), ("近7天", 7), ("近15天", 15)):
            self.batch_days.addItem(label, d)
        self.batch_days.setCurrentIndex(3)   # 默认近15天
        self.batch_days.setFixedWidth(96)
        self.batch_days.currentIndexChanged.connect(self._refresh_batches)
        bar.addWidget(self.batch_days)
        # 品种: 可选"全部品种" / 选已有 / 自己输入(模糊匹配)
        bar.addSpacing(8); bar.addWidget(QLabel("品种"))
        self.batch_variety = QComboBox(); self.batch_variety.setFixedWidth(140)
        self.batch_variety.setEditable(True)
        self.batch_variety.setInsertPolicy(QComboBox.NoInsert)
        self.batch_variety.lineEdit().setPlaceholderText("输入品种筛选")
        self.batch_variety.currentTextChanged.connect(self._refresh_batches)
        bar.addWidget(self.batch_variety)
        bar.addSpacing(8); bar.addWidget(QLabel("当班人"))
        self.batch_operator = QComboBox(); self.batch_operator.setFixedWidth(140)
        self.batch_operator.setEditable(True)
        self.batch_operator.setInsertPolicy(QComboBox.NoInsert)
        self.batch_operator.lineEdit().setPlaceholderText("输入当班人筛选")
        self.batch_operator.currentTextChanged.connect(self._refresh_batches)
        bar.addWidget(self.batch_operator)
        bar.addSpacing(8); bar.addWidget(QLabel("批次号"))
        # 批次号: 可手输, 也可下拉选(下拉随上面的品种联动); 多个用逗号隔开
        self.batch_search = QComboBox(); self.batch_search.setFixedWidth(200)
        self.batch_search.setEditable(True)
        self.batch_search.setInsertPolicy(QComboBox.NoInsert)
        self.batch_search.lineEdit().setPlaceholderText("选/输批次号(随品种联动)")
        self.batch_search.currentTextChanged.connect(self._refresh_batches)
        bar.addWidget(self.batch_search)
        bar.addWidget(self._bar_button("刷新", self._refresh_batches))
        bar.addWidget(self._bar_button("导出 CSV", self._export_batches, bg="#F18B3A"))
        bar.addStretch()
        lay.addLayout(bar)
        self.batch_table = self._make_table(
            ["品种", "批次号", "当班人", "开始时间", "结束时间", "状态", "报警数", "目标数"])
        self.batch_table.cellDoubleClicked.connect(self._show_batch_detail)
        lay.addWidget(self.batch_table, 1)
        return w

    @staticmethod
    def _reload_editable_combo(combo, all_label: str, items: list[str]) -> None:
        """刷新可编辑下拉(保留用户当前输入/选择)。
        关键: 用户正在该框里输入时不重建下拉, 否则每敲一下键就把"全部XX"塞回去,
        导致删不掉默认文字、打不进字(原 bug)。失焦后(切别处/点刷新)再重建。
        注意: 可编辑下拉的焦点在内部 lineEdit 上, 要查 lineEdit().hasFocus()。"""
        le = combo.lineEdit()
        if combo.hasFocus() or (le is not None and le.hasFocus()):
            return
        cur = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(all_label)         # 下拉第一项: 全部XX(点它=查全部)
        for it in items:
            combo.addItem(it)
        # 输入框默认空 → 显示灰色 placeholder(清爽, 空=全部); 选/输了具体值才回填
        combo.setEditText("" if (not cur or cur == all_label) else cur)
        combo.blockSignals(False)

    @staticmethod
    def _parse_filter_terms(text: str, all_label: str) -> list[str]:
        """把筛选框文本拆成多个词(逗号/空格/、/分号 隔开)。
        空 或 等于"全部XX" → 返回 []  (= 不过滤, 全部)。"""
        text = (text or "").strip()
        if not text or text == all_label:
            return []
        return [t.lower() for t in re.split(r"[,，、;；\s]+", text) if t]

    def _build_name_map(self) -> None:
        """用户名 → 姓名 映射(没填姓名则退回用户名)。当班人列显示姓名用。"""
        self._name_map = {u["username"]: (u.get("full_name") or u["username"])
                          for u in DbHelper.instance().list_users()}

    def _disp_name(self, username: str) -> str:
        return getattr(self, "_name_map", {}).get(username, username) or username

    def _batch_ids_by_variety(self, vterms: list[str]) -> list[str]:
        """批次号列表, 按品种过滤(vterms 空=全部品种)。供批次下拉跟品种联动。"""
        bs = DbHelper.instance().list_batches(500)
        if vterms:
            bs = [b for b in bs if self._match_any(b["variety"], vterms)]
        return list(dict.fromkeys(b["batch_id"] for b in bs))   # 去重

    @staticmethod
    def _match_any(value: str, terms: list[str]) -> bool:
        """value 是否包含 terms 里任意一个(子串, 不分大小写)。terms 空 → True(不过滤)。"""
        if not terms:
            return True
        v = (value or "").lower()
        return any(t in v for t in terms)

    def _refresh_batches(self, *_a) -> None:
        db = DbHelper.instance()
        self._build_name_map()
        # 刷新品种/当班人可编辑下拉(当班人显示姓名)
        self._reload_editable_combo(self.batch_variety, "全部品种", db.list_varieties())
        op_names = list(dict.fromkeys(self._disp_name(op) for op in db.distinct_operators()))
        self._reload_editable_combo(self.batch_operator, "全部当班人", op_names)
        vterms = self._parse_filter_terms(self.batch_variety.currentText(), "全部品种")
        oterms = self._parse_filter_terms(self.batch_operator.currentText(), "全部当班人")
        # 批次号下拉随品种联动: 选了品种就只列该品种的批次
        self._reload_editable_combo(self.batch_search, "全部批次", self._batch_ids_by_variety(vterms))
        bterms = self._parse_filter_terms(self.batch_search.currentText(), "全部批次")

        days = self.batch_days.currentData() if hasattr(self, "batch_days") else 15
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        rows = db.list_batches(200, since=since)
        # 批次号 / 品种 / 当班人: 均支持多个(逗号/空格/、/分号 隔开), 任一匹配即显示; 子串模糊不分大小写
        if bterms:
            rows = [b for b in rows if self._match_any(b["batch_id"], bterms)]
        if vterms:
            rows = [b for b in rows if self._match_any(b["variety"], vterms)]
        if oterms:  # 当班人按显示姓名匹配
            rows = [b for b in rows if self._match_any(self._disp_name(b["operator"]), oterms)]
        self._batches = rows
        self.batch_table.setRowCount(len(rows))
        for i, b in enumerate(rows):
            is_run = b["status"] == "running"
            cells = [b["variety"] or "—", b["batch_id"], self._disp_name(b["operator"]),
                     str(b["start_time"]), str(b["end_time"] or "—"),
                     "", str(b["alarm_count"]), str(b["total_targets"])]
            for j, val in enumerate(cells):
                self.batch_table.setItem(i, j, QTableWidgetItem(str(val)))
            # 状态彩色文字: 进行中=青, 已结束=灰
            self.batch_table.setItem(
                i, 5, self._color_item("进行中" if is_run else "已结束",
                                       _ACCENT if is_run else COLOR_TEXT_DIM))

    def _export_batches(self) -> None:
        """导出批次追溯(含每批全部异物分布, 方便存档)。不含图片。"""
        rows = getattr(self, "_batches", [])
        if not rows:
            QMessageBox.information(self, "提示", "当前没有批次可导出"); return
        db = DbHelper.instance()
        data = []
        for b in rows:
            by_class = db.report_summary(batch_id=b["batch_id"])["by_class"]
            dist = "、".join(f"{k}×{v}" for k, v in by_class.items()) or "无"
            data.append({
                "品种": b["variety"] or "—", "批次号": b["batch_id"],
                "当班人": self._disp_name(b["operator"]), "模型": b["model_name"],
                "开始时间": str(b["start_time"]), "结束时间": str(b["end_time"] or "—"),
                "状态": "进行中" if b["status"] == "running" else "已结束",
                "严重报警数": b["alarm_count"], "累计检出": b["total_targets"], "异物分布": dist,
            })
        # 导出抬头: 跟当前筛选一致
        days = self.batch_days.currentData()
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        kw = self.batch_search.currentText().strip()
        if kw == "全部批次":
            kw = ""
        vtxt = self.batch_variety.currentText().strip()
        otxt = self.batch_operator.currentText().strip()
        meta = [
            ("导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("时间范围", f"{self.batch_days.currentText()}（{since} ~ {today}）"),
            ("品种", vtxt if (vtxt and vtxt != "全部品种") else "全部品种"),
            ("当班人", otxt if (otxt and otxt != "全部当班人") else "全部当班人"),
            ("批次号筛选", kw if kw else "无"),
            ("批次数", str(len(rows))),
        ]
        self._export_csv("批次追溯", data, meta)

    def _show_batch_detail(self, row: int, _col: int) -> None:
        """双击批次 → 该批次全部检出异物分布 + 严重报警明细。"""
        if row < 0 or row >= len(getattr(self, "_batches", [])):
            return
        b = self._batches[row]
        bid = b["batch_id"]
        db = DbHelper.instance()
        by_class = db.report_summary(batch_id=bid)["by_class"]
        parts = [f"批次【{bid}】  品种={b['variety'] or '—'}",
                 f"当班人={self._disp_name(b['operator'])}  起={b['start_time']}  止={b['end_time'] or '—'}", ""]
        if by_class:
            parts.append("【检出异物分布(全部)】")
            for cls, n in by_class.items():
                parts.append(f"  · {cls}: {n}" + ("  ⚠严重" if cls in SEVERE_CN else ""))
            parts.append(f"  ── 合计: {sum(by_class.values())} 个")
        else:
            parts.append("该批次无检出异物")
        alarms = db.batch_alarms(bid)
        parts.append("")
        if alarms:
            parts.append(f"【严重报警明细({len(alarms)})】")
            for a in alarms:
                parts.append(f"  · {a['time']}  {a['class_cn']}  "
                             f"处置={'已处置' if a['handled'] else '未处置'}")
        else:
            parts.append("【严重报警】无")
        QMessageBox.information(self, f"批次 {bid} — 明细", "\n".join(parts))

    # ── Tab 报警记录(只严重物品) ──
    def _build_alarms_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.alarm_only_unhandled = False
        bar.addWidget(QLabel("时间范围"))
        self.alarm_range = QComboBox(); self.alarm_range.addItems(self._RANGES)
        self.alarm_range.setCurrentText("今天")
        self.alarm_range.currentTextChanged.connect(self._refresh_alarms)
        bar.addWidget(self.alarm_range)
        bar.addSpacing(8); bar.addWidget(QLabel("当班人"))
        self.alarm_operator = QComboBox(); self.alarm_operator.setFixedWidth(140)
        self.alarm_operator.setEditable(True)
        self.alarm_operator.setInsertPolicy(QComboBox.NoInsert)
        self.alarm_operator.lineEdit().setPlaceholderText("输入当班人筛选")
        self.alarm_operator.currentTextChanged.connect(self._refresh_alarms)
        bar.addWidget(self.alarm_operator)
        bar.addSpacing(8); bar.addWidget(QLabel("品种"))
        self.alarm_variety = QComboBox(); self.alarm_variety.setFixedWidth(130)
        self.alarm_variety.setEditable(True)
        self.alarm_variety.setInsertPolicy(QComboBox.NoInsert)
        self.alarm_variety.lineEdit().setPlaceholderText("输入品种筛选")
        self.alarm_variety.currentTextChanged.connect(self._refresh_alarms)
        bar.addWidget(self.alarm_variety)
        bar.addSpacing(8); bar.addWidget(QLabel("按批次"))
        self.alarm_batch = QComboBox(); self.alarm_batch.setFixedWidth(180)   # 可选可输, 随品种联动
        self.alarm_batch.setEditable(True)
        self.alarm_batch.setInsertPolicy(QComboBox.NoInsert)
        self.alarm_batch.lineEdit().setPlaceholderText("选/输批次号(随品种联动)")
        self.alarm_batch.currentTextChanged.connect(self._refresh_alarms)
        bar.addWidget(self.alarm_batch)
        bar.addWidget(self._bar_button("刷新", self._refresh_alarms))
        bar.addWidget(self._bar_button("只看未处置", self._toggle_alarm_filter, checkable=True))
        bar.addWidget(self._bar_button("导出 CSV", self._export_alarms, bg="#F18B3A"))
        bar.addStretch()
        lay.addLayout(bar)
        # 汇总卡片(替代原先一行文字)
        self.alarm_scope_lbl = QLabel("—")
        self.alarm_scope_lbl.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:13px; padding:2px 0;")
        lay.addWidget(self.alarm_scope_lbl)
        self._alarm_stat = self._make_stat_cards(lay, [
            ("total", "报警总计", COLOR_TEXT),
            ("unhandled", "未处置", "#E66662"),
            ("handled", "已处置", "#2DB95B"),
        ])
        self.alarm_dist_lbl = QLabel("")
        self.alarm_dist_lbl.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:12px; padding:2px 0;")
        lay.addWidget(self.alarm_dist_lbl)
        # 报警都是严重物品, 不再单列"严重度"
        self.alarm_table = self._make_table(
            ["时间", "当班人", "异物", "批次", "已分流", "已处置"])
        lay.addWidget(self.alarm_table, 1)
        return w

    def _toggle_alarm_filter(self, on: bool) -> None:
        self.alarm_only_unhandled = on
        self._refresh_alarms()

    def _refresh_alarms(self, *_a) -> None:
        db = DbHelper.instance()
        self._build_name_map()
        # 当班人/品种下拉; 批次下拉随品种联动, 可选可输
        op_names = list(dict.fromkeys(self._disp_name(op) for op in db.distinct_alarm_operators()))
        self._reload_editable_combo(self.alarm_operator, "全部当班人", op_names)
        self._reload_editable_combo(self.alarm_variety, "全部品种", db.list_varieties())
        var_of = {b["batch_id"]: (b["variety"] or "") for b in db.list_batches(500)}

        vterms = self._parse_filter_terms(self.alarm_variety.currentText(), "全部品种")
        self._reload_editable_combo(self.alarm_batch, "全部批次", self._batch_ids_by_variety(vterms))
        bterms = self._parse_filter_terms(self.alarm_batch.currentText(), "全部批次")
        oterms = self._parse_filter_terms(self.alarm_operator.currentText(), "全部当班人")
        if bterms or vterms:
            # 指定批次/品种: 不限时间, 取近 500 条按条件模糊筛
            rows = db.query_alarms(500, only_unhandled=self.alarm_only_unhandled)
            scope = "筛选"
        else:
            since, until = self._range_bounds(self.alarm_range.currentText())
            rows = db.query_alarms(500, only_unhandled=self.alarm_only_unhandled, since=since, until=until)
            scope = self.alarm_range.currentText()
        if bterms:
            rows = [r for r in rows if self._match_any(r["batch_id"] or "", bterms)]
            scope += " · 批次 " + "/".join(bterms)
        if vterms:  # 品种: 按该报警所属批次的品种筛
            rows = [r for r in rows if self._match_any(var_of.get(r["batch_id"], ""), vterms)]
            scope += " · 品种 " + "/".join(vterms)
        # 当班人筛选(可多个): 按显示姓名模糊筛
        if oterms:
            rows = [r for r in rows if self._match_any(self._disp_name(r["username"]), oterms)]
            scope += " · 当班人 " + "/".join(oterms)
        self._alarm_rows = rows

        # 时间段总计: 每类多少 + 全部总计 + 处置情况
        by_cls: dict[str, int] = {}
        for r in rows:
            by_cls[r["class_cn"]] = by_cls.get(r["class_cn"], 0) + 1
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1])) or "无"
        unhandled = sum(1 for r in rows if not r["handled"])
        self.alarm_scope_lbl.setText(f"统计范围:{scope}")
        self._alarm_stat["total"].setText(str(len(rows)))
        self._alarm_stat["unhandled"].setText(str(unhandled))
        self._alarm_stat["handled"].setText(str(len(rows) - unhandled))
        self.alarm_dist_lbl.setText(f"各异物:{parts}")

        self.alarm_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            handled = bool(r["handled"])
            cells = [str(r["time"]), self._disp_name(r["username"]), r["class_cn"], r["batch_id"] or "—",
                     "是" if r["diverted"] else "否", ""]
            for j, val in enumerate(cells):
                self.alarm_table.setItem(i, j, QTableWidgetItem(str(val)))
            self.alarm_table.setItem(
                i, 5, self._color_item("已处置" if handled else "未处置",
                                       "#2DB95B" if handled else "#E66662"))

    def _export_alarms(self) -> None:
        rows = getattr(self, "_alarm_rows", [])
        if not rows:
            QMessageBox.information(self, "提示", "当前范围没有报警可导出"); return
        # 导出不含图片(图片只在技术员单张图/文件夹推理处可导出)
        data = [{"时间": r["time"], "当班人": self._disp_name(r["username"]), "异物": r["class_cn"],
                 "批次": r["batch_id"],
                 "已分流": "是" if r["diverted"] else "否",
                 "已处置": "是" if r["handled"] else "否"} for r in rows]
        # 导出抬头: 跟当前筛选一致
        btxt = self.alarm_batch.currentText().strip()
        by_batch = bool(btxt and btxt != "全部批次")
        otxt = self.alarm_operator.currentText().strip()
        vtxt = self.alarm_variety.currentText().strip()
        meta = [
            ("导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("时间范围", "(按批次/品种)" if (by_batch or (vtxt and vtxt != "全部品种")) else f"{self.alarm_range.currentText()}（{self._range_date_str(self.alarm_range.currentText())}）"),
            ("批次", btxt if by_batch else "全部批次"),
            ("品种", vtxt if (vtxt and vtxt != "全部品种") else "全部品种"),
            ("当班人", otxt if (otxt and otxt != "全部当班人") else "全部当班人"),
            ("只看未处置", "是" if self.alarm_only_unhandled else "否"),
            ("记录条数", str(len(rows))),
        ]
        self._export_csv("报警记录", data, meta)

    # ── Tab 审计 ──
    def _build_audit_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("时间范围"))
        self.audit_range = QComboBox(); self.audit_range.addItems(self._RANGES)
        self.audit_range.setCurrentText("近半月")
        self.audit_range.currentTextChanged.connect(self._refresh_audit)
        bar.addWidget(self.audit_range)
        bar.addSpacing(8); bar.addWidget(QLabel("角色"))
        self.audit_role = QComboBox(); self.audit_role.setFixedWidth(110)
        self.audit_role.addItem("全部角色", "")
        for r in ("worker", "technician", "admin"):
            self.audit_role.addItem(_ROLE_CN[r], r)
        self.audit_role.currentIndexChanged.connect(self._refresh_audit)
        bar.addWidget(self.audit_role)
        bar.addSpacing(8); bar.addWidget(QLabel("用户"))
        self.audit_user = QComboBox(); self.audit_user.setMinimumWidth(150)
        self.audit_user.setEditable(True)
        self.audit_user.setInsertPolicy(QComboBox.NoInsert)
        self.audit_user.lineEdit().setPlaceholderText("输入用户名筛选")
        self.audit_user.currentTextChanged.connect(self._refresh_audit)
        bar.addWidget(self.audit_user)
        bar.addWidget(self._bar_button("刷新", self._refresh_audit))
        bar.addWidget(self._bar_button("导出 CSV", self._export_audit, bg="#F18B3A"))
        bar.addStretch()
        lay.addLayout(bar)
        self.audit_table = self._make_table(["时间", "用户", "角色", "操作", "详情"])
        lay.addWidget(self.audit_table, 1)
        return w

    def _refresh_audit(self, *_a) -> None:
        db = DbHelper.instance()
        self._build_name_map()
        since, until = (self._range_bounds(self.audit_range.currentText())
                        if hasattr(self, "audit_range") else (None, None))
        # 刷新用户可编辑下拉(显示姓名, 焦点保护可删可输)
        user_names = list(dict.fromkeys(self._disp_name(u) for u in db.distinct_audit_users()))
        self._reload_editable_combo(self.audit_user, "全部用户", user_names)
        uterms = self._parse_filter_terms(self.audit_user.currentText(), "全部用户")
        role = self.audit_role.currentData() or None
        rows = db.query_audit(500, since=since, until=until, role=role)
        # 用户: 支持多个(逗号/空格 隔开), 按显示姓名模糊匹配
        if uterms:
            rows = [r for r in rows if self._match_any(self._disp_name(r["username"]), uterms)]
        self._audit_rows = rows
        self.audit_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cells = [str(r["time"]), self._disp_name(r["username"]), "",
                     _action_cn(r["action"]), r["detail"]]
            for j, val in enumerate(cells):
                self.audit_table.setItem(i, j, QTableWidgetItem(val))
            # 角色彩色文字
            self.audit_table.setItem(
                i, 2, self._color_item(_ROLE_CN.get(r["role"], r["role"]),
                                       _ROLE_COLOR.get(r["role"], COLOR_TEXT)))

    def _export_audit(self) -> None:
        rows = getattr(self, "_audit_rows", [])
        if not rows:
            QMessageBox.information(self, "提示", "没有可导出的审计记录"); return
        data = [{"时间": r["time"], "用户": self._disp_name(r["username"]),
                 "角色": _ROLE_CN.get(r["role"], r["role"]),
                 "操作": _action_cn(r["action"]), "详情": r["detail"]} for r in rows]
        # 导出抬头: 跟当前筛选一致
        utxt = self.audit_user.currentText().strip()
        role = self.audit_role.currentData() or None
        meta = [
            ("导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("时间范围", f"{self.audit_range.currentText()}（{self._range_date_str(self.audit_range.currentText())}）"),
            ("角色", _ROLE_CN.get(role, role) if role else "全部角色"),
            ("用户", utxt if (utxt and utxt != "全部用户") else "全部用户"),
            ("记录条数", str(len(rows))),
        ]
        self._export_csv("审计日志", data, meta)

    # ── Tab 报表 ──
    _RANGES = ["今天", "本周", "近半月", "全部"]

    def _build_report_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("时间范围"))
        self.range_combo = QComboBox()
        self.range_combo.addItems(self._RANGES)
        self.range_combo.setCurrentText("今天")
        self.range_combo.currentTextChanged.connect(self._refresh_report)
        bar.addWidget(self.range_combo)
        bar.addSpacing(12)
        bar.addWidget(QLabel("品种"))
        self.report_variety = QComboBox(); self.report_variety.setFixedWidth(130)
        self.report_variety.setEditable(True)
        self.report_variety.setInsertPolicy(QComboBox.NoInsert)
        self.report_variety.lineEdit().setPlaceholderText("输入品种筛选")
        self.report_variety.currentTextChanged.connect(self._refresh_report)
        bar.addWidget(self.report_variety)
        bar.addSpacing(12)
        bar.addWidget(QLabel("按批次"))
        self.batch_combo = QComboBox(); self.batch_combo.setFixedWidth(180)   # 可选可输, 随品种联动
        self.batch_combo.setEditable(True)
        self.batch_combo.setInsertPolicy(QComboBox.NoInsert)
        self.batch_combo.lineEdit().setPlaceholderText("选/输批次号(随品种联动)")
        self.batch_combo.currentTextChanged.connect(self._refresh_report)
        bar.addWidget(self.batch_combo)
        bar.addSpacing(12)
        bar.addWidget(QLabel("显示"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["全部异物", "仅严重异物"])
        self.scope_combo.currentTextChanged.connect(self._refresh_report)
        bar.addWidget(self.scope_combo)
        bar.addWidget(self._bar_button("刷新", self._refresh_report))
        bar.addWidget(self._bar_button("导出 CSV", self._export_report, bg="#F18B3A"))
        bar.addStretch()
        lay.addLayout(bar)

        # 汇总改成几张小卡片(替代原先挤成一行的文字), 更清爽
        self.report_scope_lbl = QLabel("—")
        self.report_scope_lbl.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:13px; padding:2px 0;")
        lay.addWidget(self.report_scope_lbl)
        cards = QHBoxLayout(); cards.setSpacing(10)
        self._stat_value: dict[str, QLabel] = {}
        for key, cap, color in (
            ("runs", "检测次数", COLOR_TEXT),
            ("targets", "累计异物", COLOR_TEXT),
            ("alarm", "严重报警", "#F18B3A"),
        ):
            card = QFrame()
            card.setStyleSheet(
                f"QFrame{{background:{COLOR_BG_SIDE}; border:1px solid {COLOR_BORDER};"
                f"border-radius:6px;}}")
            cv = QVBoxLayout(card); cv.setContentsMargins(14, 8, 14, 8); cv.setSpacing(2)
            val = QLabel("0"); val.setStyleSheet(
                f"color:{color}; font-size:22px; font-weight:bold; border:none;")
            capl = QLabel(cap); capl.setStyleSheet(
                f"color:{COLOR_TEXT_DIM}; font-size:12px; border:none;")
            cv.addWidget(val); cv.addWidget(capl)
            self._stat_value[key] = val
            cards.addWidget(card)
        cards.addStretch()
        lay.addLayout(cards)

        # 图表区 + 明细表 之间放可拖拽分隔条: 往上拉表格, 图表相应变小
        charts_w = QWidget(); ch = QHBoxLayout(charts_w); ch.setContentsMargins(0, 0, 0, 0)
        self.chart = BarChart()      # 各类异物检出数量
        self.pie = PieChart()        # 严重异物 vs 其他异物 占比
        ch.addWidget(self.chart, 2)
        ch.addWidget(self.pie, 1)
        self.report_table = self._make_table(["异物类别", "累计检出数"])
        split = QSplitter(Qt.Vertical)
        split.addWidget(charts_w)
        split.addWidget(self.report_table)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        split.setSizes([420, 240])   # 初始高度, 可拖拽调整
        split.setChildrenCollapsible(False)
        split.setHandleWidth(6)
        split.setStyleSheet(
            f"QSplitter::handle{{background:{COLOR_BORDER};}}"
            f"QSplitter::handle:hover{{background:{_ACCENT};}}")
        lay.addWidget(split, 1)
        return w

    def _range_bounds(self, name: str) -> tuple[str | None, str | None]:
        now = datetime.now()
        if name == "今天":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif name == "本周":
            monday = now - timedelta(days=now.weekday())
            since = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif name == "近半月":
            since = now - timedelta(days=15)
        else:  # 全部
            return None, None
        return since.strftime("%Y-%m-%d %H:%M:%S"), None

    def _refresh_report(self, *_a) -> None:
        db = DbHelper.instance()
        # 品种下拉; 批次下拉随品种联动, 可选可输
        self._reload_editable_combo(self.report_variety, "全部品种", db.list_varieties())
        vterms = self._parse_filter_terms(self.report_variety.currentText(), "全部品种")
        self._reload_editable_combo(self.batch_combo, "全部批次", self._batch_ids_by_variety(vterms))
        bterms = self._parse_filter_terms(self.batch_combo.currentText(), "全部批次")
        range_name = self.range_combo.currentText()
        # 显示口径: 全部 / 仅严重 → 直接在数据层按类别过滤, 所有指标(检测次数/累计异物/分布)联动
        only_severe = self.scope_combo.currentText() == "仅严重异物"
        only_classes = SEVERE_CN if only_severe else None
        if bterms or vterms:
            # 指定批次/品种(可多个): 找出匹配的批次, 汇总各批统计
            matched = [b["batch_id"] for b in db.list_batches(500)
                       if (not bterms or self._match_any(b["batch_id"], bterms))
                       and (not vterms or self._match_any(b["variety"], vterms))]
            agg_runs = agg_targets = n_alarm = 0
            agg_cls: dict[str, int] = {}
            for bid in matched:
                r = db.report_summary(batch_id=bid, only_classes=only_classes)
                agg_runs += r["runs"]; agg_targets += r["total_targets"]
                for k, v in r["by_class"].items():
                    agg_cls[k] = agg_cls.get(k, 0) + v
                n_alarm += db.count_alarms(batch_id=bid)
            rep = {"runs": agg_runs, "total_targets": agg_targets,
                   "by_class": dict(sorted(agg_cls.items(), key=lambda x: -x[1]))}
            parts = []
            if bterms: parts.append("批次 " + "/".join(bterms))
            if vterms: parts.append("品种 " + "/".join(vterms))
            lbl = " · ".join(parts) + f"(匹配{len(matched)}个批次)"
            title = f"{lbl} — 异物检出统计"
            scope = lbl
        else:
            since, until = self._range_bounds(range_name)
            rep = db.report_summary(since=since, until=until, only_classes=only_classes)
            n_alarm = db.count_alarms(since=since, until=until)
            title = f"{range_name} — 异物检出统计"
            scope = range_name

        by_class = rep["by_class"]   # 已按显示口径过滤
        if only_severe:
            title = title.replace("异物检出统计", "严重异物检出统计")
        # 顶部卡片: 全部指标都跟随当前筛选(时间/批次/显示)
        self.report_scope_lbl.setText(f"统计范围:{scope}")
        self._stat_value["runs"].setText(str(rep["runs"]))
        self._stat_value["targets"].setText(str(rep["total_targets"]))
        self._stat_value["alarm"].setText(str(n_alarm))
        self._report_by_class = by_class
        self.chart.set_data(by_class, title)
        # 饼图: 跟随所有筛选(时间/批次/显示)
        if only_severe:
            # 仅严重 → 各严重类占比
            self.pie.set_data(dict(by_class), "严重异物各类占比")
        else:
            # 全部 → 严重 vs 其他 占比
            sev = sum(v for k, v in by_class.items() if k in SEVERE_CN)
            oth = sum(v for k, v in by_class.items() if k not in SEVERE_CN)
            self.pie.set_data({"严重异物": sev, "其他异物": oth}, "严重 / 其他 占比")
        self.report_table.setRowCount(len(by_class) + 1)
        for i, (cls, n) in enumerate(by_class.items()):
            self.report_table.setItem(i, 0, QTableWidgetItem(str(cls)))
            self.report_table.setItem(i, 1, QTableWidgetItem(str(n)))
        # 末行: 所有类合计
        total_item = QTableWidgetItem("合计(所有类)")
        total_item.setForeground(QColor(COLOR_BTN_PRIMARY))
        sum_item = QTableWidgetItem(str(sum(by_class.values())))
        sum_item.setForeground(QColor(COLOR_BTN_PRIMARY))
        self.report_table.setItem(len(by_class), 0, total_item)
        self.report_table.setItem(len(by_class), 1, sum_item)

    def _export_report(self) -> None:
        by_class = getattr(self, "_report_by_class", {})
        if not by_class:
            QMessageBox.information(self, "提示", "当前范围没有可导出的数据"); return
        # 当前筛选条件
        btxt = self.batch_combo.currentText().strip()
        if btxt == "全部批次":
            btxt = ""
        by_batch = bool(btxt and btxt != "全部批次")
        vtxt = self.report_variety.currentText().strip()
        rng = self.range_combo.currentText()
        scope = self.scope_combo.currentText()
        severe_sum = sum(v for k, v in by_class.items() if k in SEVERE_CN)
        # 把"今天/本周/近半月"换算成具体日期, 避免"今天"无从判断是哪天
        since, _u = self._range_bounds(rng)
        today = datetime.now().strftime("%Y-%m-%d")
        date_str = "全部时间" if not since else f"{since[:10]} ~ {today}"
        range_cell = "(按批次, 见下)" if by_batch else f"{rng}（{date_str}）"
        # CSV: 先一段筛选条件抬头, 再各类明细, 最后合计 + 严重合计
        rows = [
            {"异物类别": "导出时间", "累计检出数": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"异物类别": "【筛选条件】", "累计检出数": ""},
            {"异物类别": "时间范围", "累计检出数": range_cell},
            {"异物类别": "品种", "累计检出数": vtxt if (vtxt and vtxt != "全部品种") else "全部品种"},
            {"异物类别": "批次", "累计检出数": btxt if by_batch else "全部批次"},
            {"异物类别": "显示范围", "累计检出数": scope},
            {"异物类别": "", "累计检出数": ""},
            {"异物类别": "【各类检出】", "累计检出数": ""},
        ]
        for k, v in by_class.items():
            rows.append({"异物类别": k + ("(严重)" if k in SEVERE_CN else ""), "累计检出数": v})
        rows.append({"异物类别": "合计(所有类)", "累计检出数": sum(by_class.values())})
        rows.append({"异物类别": "其中严重合计", "累计检出数": severe_sum})
        tag = btxt if by_batch else rng
        self._export_csv(f"异物报表_{tag}", rows)

    # ── 公共 ──
    def _range_date_str(self, range_name: str) -> str:
        """把"今天/本周/近半月/全部"换算成具体日期区间, 写进导出抬头, 避免日后看不懂"今天"是哪天。"""
        since, _u = self._range_bounds(range_name)
        today = datetime.now().strftime("%Y-%m-%d")
        return "全部时间" if not since else f"{since[:10]} ~ {today}"

    def _export_csv(self, name: str, rows: list[dict[str, Any]],
                    meta: list[tuple[str, str]] | None = None) -> None:
        default = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fp, _ = QFileDialog.getSaveFileName(self, "导出 CSV", str(Path.home() / default), "CSV (*.csv)")
        if not fp:
            return
        try:
            with open(fp, "w", newline="", encoding="utf-8-sig") as f:
                if meta:
                    # 筛选条件抬头(独立两列), 与下方表格之间空一行
                    mw = csv.writer(f)
                    mw.writerow(["【筛选条件】", ""])
                    for k, v in meta:
                        mw.writerow([k, v])
                    mw.writerow([])
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            QMessageBox.information(self, "已导出", f"已保存到\n{fp}")
        except Exception as e:
            log.error("导出 CSV 失败: %s", e)
            QMessageBox.warning(self, "导出失败", str(e))

    def _make_stat_cards(self, parent_layout, specs: list[tuple[str, str, str]]) -> dict[str, QLabel]:
        """生成一行小卡片(数值大 + 标签小)。specs=[(key, 标签, 颜色)...], 返回 {key: 数值QLabel}。"""
        cards = QHBoxLayout(); cards.setSpacing(10)
        out: dict[str, QLabel] = {}
        for key, cap, color in specs:
            card = QFrame()
            card.setStyleSheet(
                f"QFrame{{background:{COLOR_BG_SIDE}; border:1px solid {COLOR_BORDER};"
                f"border-radius:6px;}}")
            cv = QVBoxLayout(card); cv.setContentsMargins(14, 8, 14, 8); cv.setSpacing(2)
            val = QLabel("0"); val.setStyleSheet(
                f"color:{color}; font-size:22px; font-weight:bold; border:none;")
            capl = QLabel(cap); capl.setStyleSheet(
                f"color:{COLOR_TEXT_DIM}; font-size:12px; border:none;")
            cv.addWidget(val); cv.addWidget(capl)
            out[key] = val
            cards.addWidget(card)
        cards.addStretch()
        parent_layout.addLayout(cards)
        return out

    def _make_table(self, headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setSelectionMode(QTableWidget.SingleSelection)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.horizontalHeader().setHighlightSections(False)
        t.verticalHeader().setVisible(False)            # 隐藏左侧行号, 更干净
        t.verticalHeader().setDefaultSectionSize(36)    # 行高大一点, 透气
        t.setShowGrid(False)
        t.setAlternatingRowColors(True)                 # 隔行底色
        # 科技感: 青色表头下划线 + 隔行底色 + 半透明青色选中 + 无网格无外框
        # 注意: 不在 ::item 里写 color, 否则会盖掉 setForeground 的彩色文字
        t.setStyleSheet(
            f"QTableWidget {{ background:{COLOR_BG_MAIN}; alternate-background-color:{COLOR_TABLE_ALT};"
            f"color:{COLOR_TEXT}; gridline-color:transparent; border:none; }}"
            f"QTableWidget::item {{ padding:6px; border:none; }}"
            f"QTableWidget::item:selected {{ background:rgba(33,212,200,0.18); }}"
            f"QHeaderView::section {{ background:{COLOR_BG_SIDE}; color:{_ACCENT};"
            f"padding:8px 6px; border:none; border-bottom:2px solid {_ACCENT}; font-weight:bold; }}"
            f"QTableCornerButton::section {{ background:{COLOR_BG_SIDE}; border:none; }}")
        return t

    @staticmethod
    def _color_item(text: str, color: str) -> QTableWidgetItem:
        """彩色加粗文字单元格(无控件无底色, 隔行灰下不露黑块)。用于角色/状态等。"""
        it = QTableWidgetItem(text)
        it.setForeground(QColor(color))
        f = it.font(); f.setBold(True); it.setFont(f)
        it.setTextAlignment(Qt.AlignCenter)
        return it
