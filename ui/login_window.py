"""登录窗口 + 注册对话框 + 修改信息对话框"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from config import (
    COLOR_BG_MAIN, COLOR_BG_SIDE, COLOR_BORDER, COLOR_BTN_PRIMARY,
    COLOR_TEXT, COLOR_TEXT_DIM,
)
from core.app_state import state
from db.db_helper import DbHelper
from ui.widgets import CaptchaWidget
from utils.common import get_logger

log = get_logger("login")


_INPUT_QSS = (
    f"QLineEdit {{ background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
    f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:8px; }}"
    f"QLineEdit:focus {{ border:1px solid {COLOR_BTN_PRIMARY}; }}"
)
_BTN_PRIMARY_QSS = (
    f"QPushButton {{ background:{COLOR_BTN_PRIMARY}; color:white;"
    f"border:none; border-radius:4px; padding:10px; font-size:14px; font-weight:bold; }}"
    f"QPushButton:hover {{ background:#3E9CFF; }}"
)
_LINK_QSS = (
    f"QPushButton {{ background:transparent; color:{COLOR_BTN_PRIMARY};"
    f"border:none; padding:4px; }}"
    f"QPushButton:hover {{ color:#5BB0FF; text-decoration:underline; }}"
)


class _LogoCircle(QWidget):
    """登录页左侧圆形 logo: 浅蓝渐变圆 + 中央三根烟草条"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(220, 220)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 6

        # 圆形浅蓝底
        p.setPen(QPen(QColor(120, 170, 230, 200), 2))
        p.setBrush(QColor(220, 235, 250))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # 中央三根烟条 (角度略错开)
        p.setBrush(QColor(180, 190, 200))
        p.setPen(QPen(QColor(140, 150, 160), 1))
        bar_w, bar_h = 22, 90
        for i, ang_deg in enumerate((-12, 0, 12)):
            p.save()
            p.translate(cx + (i - 1) * 24, cy)
            p.rotate(ang_deg)
            p.drawRoundedRect(-bar_w / 2, -bar_h / 2, bar_w, bar_h, 4, 4)
            p.restore()


class LoginWindow(QWidget):
    """登录窗口 — 双栏布局, 左侧 logo + 标语, 右侧表单"""

    login_success = Signal(str)  # username

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(960, 600)
        self.resize(1080, 640)
        self.setWindowTitle("烟草异物检测系统 — 登录")
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_left(), 1)
        root.addWidget(self._build_right(), 1)

    def _build_left(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(f"background:{COLOR_BG_MAIN};")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(80, 60, 80, 60)
        lay.setSpacing(20)
        lay.addStretch()

        title = QLabel("烟草异物检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:30px; font-weight:bold; letter-spacing:2px;")
        lay.addWidget(title)

        logo_row = QHBoxLayout()
        logo_row.addStretch(); logo_row.addWidget(_LogoCircle()); logo_row.addStretch()
        lay.addLayout(logo_row)

        sub1 = QLabel("烟草异物检测  ·  智能识别")
        sub1.setAlignment(Qt.AlignCenter)
        sub1.setStyleSheet(f"color:{COLOR_BTN_PRIMARY}; font-size:15px; padding-top:8px;")
        lay.addWidget(sub1)

        sub2 = QLabel("基于多光谱YOLO的烟草异物实时检测\n北京智谱灵瞳科技有限公司")
        sub2.setAlignment(Qt.AlignCenter)
        sub2.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:12px; padding-top:4px;")
        lay.addWidget(sub2)

        lay.addStretch()
        return wrap

    def _build_right(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(f"background:{COLOR_BG_SIDE};")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(80, 60, 80, 60)
        lay.setSpacing(14)
        lay.addStretch()

        title = QLabel("烟草异物检测系统登录")
        title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:22px; font-weight:bold;")
        lay.addWidget(title)
        lay.addSpacing(16)

        for label_text, attr, placeholder, is_pwd in [
            ("用户名", "user_input", "请输入用户名", False),
            ("密码", "pwd_input", "请输入密码", True),
        ]:
            lab = QLabel(label_text); lab.setStyleSheet(f"color:{COLOR_TEXT};")
            lay.addWidget(lab)
            inp = QLineEdit()
            inp.setPlaceholderText(placeholder)
            inp.setStyleSheet(_INPUT_QSS); inp.setMinimumHeight(42)
            if is_pwd:
                inp.setEchoMode(QLineEdit.Password)
            setattr(self, attr, inp)
            lay.addWidget(inp)

        # 验证码
        cap_lab = QLabel("验证码"); cap_lab.setStyleSheet(f"color:{COLOR_TEXT};")
        lay.addWidget(cap_lab)
        cap_row = QHBoxLayout()
        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("请输入验证码")
        self.captcha_input.setMaxLength(4)
        self.captcha_input.setStyleSheet(_INPUT_QSS)
        self.captcha_input.setMinimumHeight(46)
        self.captcha = CaptchaWidget()
        cap_row.addWidget(self.captcha_input, 1)
        cap_row.addWidget(self.captcha)
        lay.addLayout(cap_row)

        lay.addSpacing(14)

        self.login_btn = QPushButton("进入烟草异物检测系统")
        self.login_btn.setMinimumHeight(48)
        self.login_btn.setStyleSheet(_BTN_PRIMARY_QSS)
        self.login_btn.clicked.connect(self._do_login)
        lay.addWidget(self.login_btn)

        self.register_btn = QPushButton("注册账号")
        self.register_btn.setMinimumHeight(42)
        self.register_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BTN_PRIMARY}; border-radius:4px; font-size:13px; }}"
            f"QPushButton:hover {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
        )
        self.register_btn.clicked.connect(self._open_register)
        lay.addWidget(self.register_btn)

        lay.addStretch()

        # 回车直接登录
        self.user_input.returnPressed.connect(self._do_login)
        self.pwd_input.returnPressed.connect(self._do_login)
        self.captcha_input.returnPressed.connect(self._do_login)
        return wrap

    def _do_login(self) -> None:
        user = self.user_input.text().strip()
        pwd = self.pwd_input.text()
        cap = self.captcha_input.text()

        if not user or not pwd:
            QMessageBox.warning(self, "提示", "请输入用户名和密码")
            return
        if not self.captcha.verify(cap):
            QMessageBox.warning(self, "提示", "验证码错误")
            self.captcha.refresh()
            self.captcha_input.clear()
            return
        if not DbHelper.instance().verify_user(user, pwd):
            QMessageBox.warning(self, "提示", "用户名或密码错误")
            self.captcha.refresh()
            self.captcha_input.clear()
            return

        u = DbHelper.instance().get_user(user) or {}
        state.username = user
        state.avatar_path = u.get("avatar_path", "")
        log.info("用户登录: %s", user)
        self.login_success.emit(user)

    def _open_register(self) -> None:
        dlg = RegisterDialog(self)
        dlg.exec()


class RegisterDialog(QDialog):
    """注册账号对话框"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("注册新用户")
        self.setFixedSize(400, 420)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 24, 32, 24)
        lay.setSpacing(12)

        title = QLabel("注册新账号")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        lay.addWidget(title)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("用户名 (3-20 字符)")
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("密码 (≥6 位)")
        self.pwd_input.setEchoMode(QLineEdit.Password)
        self.pwd2_input = QLineEdit()
        self.pwd2_input.setPlaceholderText("确认密码")
        self.pwd2_input.setEchoMode(QLineEdit.Password)
        for w in (self.user_input, self.pwd_input, self.pwd2_input):
            w.setStyleSheet(_INPUT_QSS)
            w.setMinimumHeight(38)
            lay.addWidget(w)

        cap_row = QHBoxLayout()
        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("验证码")
        self.captcha_input.setStyleSheet(_INPUT_QSS)
        self.captcha_input.setMinimumHeight(44)
        self.captcha = CaptchaWidget()
        cap_row.addWidget(self.captcha_input, 1)
        cap_row.addWidget(self.captcha)
        lay.addLayout(cap_row)

        btn = QPushButton("注册")
        btn.setMinimumHeight(40)
        btn.setStyleSheet(_BTN_PRIMARY_QSS)
        btn.clicked.connect(self._do_register)
        lay.addWidget(btn)
        lay.addStretch()

    def _do_register(self) -> None:
        u = self.user_input.text().strip()
        p1 = self.pwd_input.text()
        p2 = self.pwd2_input.text()
        c = self.captcha_input.text()
        if len(u) < 3 or len(u) > 20:
            QMessageBox.warning(self, "提示", "用户名长度需在 3-20 之间")
            return
        if len(p1) < 6:
            QMessageBox.warning(self, "提示", "密码至少 6 位")
            return
        if p1 != p2:
            QMessageBox.warning(self, "提示", "两次密码不一致")
            return
        if not self.captcha.verify(c):
            QMessageBox.warning(self, "提示", "验证码错误")
            self.captcha.refresh()
            return
        if not DbHelper.instance().add_user(u, p1):
            QMessageBox.warning(self, "提示", "用户名已存在")
            return
        QMessageBox.information(self, "提示", "注册成功, 请登录")
        self.accept()


class EditProfileDialog(QDialog):
    """修改密码 + 上传头像"""

    saved = Signal()

    def __init__(self, username: str, parent=None) -> None:
        super().__init__(parent)
        self.username = username
        self.setWindowTitle("修改信息")
        self.setFixedSize(400, 360)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")
        self._avatar_path = ""
        self._build()

    def _build(self) -> None:
        lay = QFormLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(12)

        self.old_input = QLineEdit()
        self.old_input.setEchoMode(QLineEdit.Password)
        self.new_input = QLineEdit()
        self.new_input.setEchoMode(QLineEdit.Password)
        self.new2_input = QLineEdit()
        self.new2_input.setEchoMode(QLineEdit.Password)
        for w in (self.old_input, self.new_input, self.new2_input):
            w.setStyleSheet(_INPUT_QSS)
            w.setMinimumHeight(34)
        lay.addRow("旧密码:", self.old_input)
        lay.addRow("新密码:", self.new_input)
        lay.addRow("确认新密码:", self.new2_input)

        avatar_row = QHBoxLayout()
        self.avatar_label = QLabel("（未选择）")
        self.avatar_label.setStyleSheet(f"color:{COLOR_TEXT_DIM};")
        pick = QPushButton("选择头像")
        pick.clicked.connect(self._pick_avatar)
        avatar_row.addWidget(self.avatar_label, 1)
        avatar_row.addWidget(pick)
        lay.addRow("头像:", avatar_row)

        save = QPushButton("保存")
        save.setMinimumHeight(36)
        save.setStyleSheet(_BTN_PRIMARY_QSS)
        save.clicked.connect(self._save)
        lay.addRow("", save)

    def _pick_avatar(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择头像", "", "图片 (*.jpg *.png *.jpeg *.bmp)",
        )
        if fp:
            self._avatar_path = fp
            self.avatar_label.setText(fp.rsplit("/", 1)[-1])

    def _save(self) -> None:
        db = DbHelper.instance()
        changed = False
        if self.new_input.text():
            if self.new_input.text() != self.new2_input.text():
                QMessageBox.warning(self, "提示", "两次新密码不一致")
                return
            if not db.update_password(self.username, self.old_input.text(), self.new_input.text()):
                QMessageBox.warning(self, "提示", "旧密码错误或更新失败")
                return
            changed = True
        if self._avatar_path:
            db.update_avatar(self.username, self._avatar_path)
            state.avatar_path = self._avatar_path
            changed = True
        if changed:
            QMessageBox.information(self, "提示", "保存成功")
            self.saved.emit()
            self.accept()
        else:
            QMessageBox.information(self, "提示", "无任何变更")
