"""登录窗口 + 注册对话框 + 修改信息对话框"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
    QRadialGradient,
)
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
    """登录页圆形 logo — 科技 HUD 风:
    深色底 + 霓虹青蓝刻度环 + 细线烟叶轮廓 + 检测框四角 + 扫描线 + 命中节点。
    """

    _CYAN = QColor(46, 230, 255)
    _BLUE = QColor(45, 140, 254)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(220, 220)

    def paintEvent(self, _e) -> None:  # noqa: N802
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = min(w, h) / 2.0 - 6
        cyan, blue = self._CYAN, self._BLUE

        # ── 深色圆底(径向) ──
        bg = QRadialGradient(cx, cy * 0.85, r * 1.25)
        bg.setColorAt(0.0, QColor(22, 38, 60))
        bg.setColorAt(1.0, QColor(10, 17, 30))
        p.setPen(Qt.NoPen); p.setBrush(QBrush(bg))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # ── 外发光环 + 主环(青→蓝渐变) ──
        ring_grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
        ring_grad.setColorAt(0.0, cyan); ring_grad.setColorAt(1.0, blue)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(46, 230, 255, 45), 7))           # 外发光
        p.drawEllipse(QPointF(cx, cy), r - 4, r - 4)
        p.setPen(QPen(QBrush(ring_grad), 2.4))                # 主环
        p.drawEllipse(QPointF(cx, cy), r - 4, r - 4)

        # ── HUD 刻度(环上短线, 长短交替) ──
        p.save(); p.translate(cx, cy)
        for i in range(60):
            ang = math.radians(i * 6)
            long = (i % 5 == 0)
            r1 = r - 4
            r2 = r - (13 if long else 8)
            p.setPen(QPen(QColor(46, 230, 255, 150 if long else 70), 1.4 if long else 1.0))
            p.drawLine(QPointF(math.cos(ang) * r2, math.sin(ang) * r2),
                       QPointF(math.cos(ang) * r1, math.sin(ang) * r1))
        # 顶部一段高亮弧(像扫描进度)
        p.setPen(QPen(cyan, 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(int(-(r - 4)), int(-(r - 4)), int((r - 4) * 2), int((r - 4) * 2), 55 * 16, 70 * 16)
        p.restore()

        # ── 细线烟叶轮廓(倾斜, 半透明青色填充 + 描边) ──
        p.save()
        p.translate(cx - r * 0.04, cy)
        p.rotate(-20)
        ll, lw = r * 0.62, r * 0.32
        leaf = QPainterPath()
        leaf.moveTo(0, -ll)
        leaf.cubicTo(lw, -ll * 0.45, lw, ll * 0.5, 0, ll)
        leaf.cubicTo(-lw, ll * 0.5, -lw, -ll * 0.45, 0, -ll)
        fill = QLinearGradient(-lw, -ll, lw, ll)
        fill.setColorAt(0.0, QColor(46, 230, 255, 38))
        fill.setColorAt(1.0, QColor(45, 140, 254, 22))
        p.setBrush(QBrush(fill))
        p.setPen(QPen(cyan, 1.8))
        p.drawPath(leaf)
        # 主叶脉 + 斜向侧脉(细线)
        p.setPen(QPen(QColor(46, 230, 255, 170), 1.2))
        p.drawLine(QPointF(0, -ll * 0.9), QPointF(0, ll * 0.9))
        for t in (-0.4, -0.05, 0.3):
            y = ll * t; side = lw * (1 - abs(t)) * 0.8
            p.drawLine(QPointF(0, y), QPointF(side, y + ll * 0.14))
            p.drawLine(QPointF(0, y), QPointF(-side, y + ll * 0.14))
        p.restore()

        # ── 检测框四角(targeting brackets, 框住叶子) ──
        bs = r * 0.62          # 半边长
        arm = r * 0.16         # 角臂长
        p.setPen(QPen(cyan, 2.2, Qt.SolidLine, Qt.RoundCap))
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = cx + sx * bs, cy + sy * bs
            p.drawLine(QPointF(x, y), QPointF(x - sx * arm, y))
            p.drawLine(QPointF(x, y), QPointF(x, y - sy * arm))

        # ── 扫描线(横向, 中部, 两端渐隐) ──
        sy = cy - r * 0.12
        scan = QLinearGradient(cx - bs, sy, cx + bs, sy)
        scan.setColorAt(0.0, QColor(46, 230, 255, 0))
        scan.setColorAt(0.5, QColor(46, 230, 255, 220))
        scan.setColorAt(1.0, QColor(46, 230, 255, 0))
        p.setPen(QPen(QBrush(scan), 2.0))
        p.drawLine(QPointF(cx - bs, sy), QPointF(cx + bs, sy))

        # ── 命中节点(小方块 + 外环, 像被识别框中的目标点) ──
        for dx, dy in ((0.22, 0.30), (-0.28, 0.12)):
            nx, ny = cx + r * dx, cy + r * dy
            p.setPen(QPen(QColor(46, 230, 255, 120), 1.2)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(nx, ny), r * 0.06, r * 0.06)
            p.setPen(Qt.NoPen); p.setBrush(cyan)
            p.drawRect(int(nx - r * 0.018), int(ny - r * 0.018), int(r * 0.036), int(r * 0.036))


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

        # 去掉自助注册:账号统一由管理员开通(见角色分层方案)
        hint = QLabel("账号由管理员开通 · 忘记密码请联系管理员")
        hint.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:12px;")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

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
        if not u.get("active", 1):
            QMessageBox.warning(self, "提示", "账号已停用,请联系管理员")
            return
        state.username = user
        state.avatar_path = u.get("avatar_path", "")
        state.role = u.get("role", "worker")
        log.info("用户登录: %s (role=%s)", user, state.role)
        # 首次登录改密提示已彻底去掉(原实现反复弹, 体验差)
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
