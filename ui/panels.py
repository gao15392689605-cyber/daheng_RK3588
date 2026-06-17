"""独立面板控件: HistoryQueryWidget (历史查询弹窗) + ProfilePanel (个人中心)

从 widgets.py 拆出来, 保持 widgets.py < 500 行
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSlider,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from config import (
    COLOR_BG_MAIN, COLOR_BG_SIDE, COLOR_BORDER, COLOR_BTN_PRIMARY, COLOR_TEXT, COLOR_TEXT_DIM,
)
from core.app_state import state
from core.exporter import Exporter
from db.db_helper import DbHelper
from ui.widgets import make_round_pixmap


def ask_save_kind(parent, text: str, allow_image: bool = False) -> set[str] | None:
    """弹保存内容选择, 返回 {'image','detail','summary'} 的子集; 取消返回 None.

    allow_image=True  (照片/文件夹等静态图): 照片 / 明细 / 汇总 / 全部 / 取消
    allow_image=False (视频/摄像头/全程):    明细 / 汇总 / 全部 / 取消
    "全部" = 该场景全部可选内容. 所有保存/导出入口共用, 保证交互一致.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("保存内容选择")
    box.setText(text)
    btns: dict = {}
    if allow_image:
        btns[box.addButton("照片", QMessageBox.AcceptRole)] = {"image"}
    btns[box.addButton("明细", QMessageBox.AcceptRole)] = {"detail"}
    btns[box.addButton("汇总", QMessageBox.AcceptRole)] = {"summary"}
    all_set = {"detail", "summary"} | ({"image"} if allow_image else set())
    btns[box.addButton("全部", QMessageBox.AcceptRole)] = all_set
    box.addButton("取消", QMessageBox.RejectRole)
    box.exec()
    return btns.get(box.clickedButton())


class HistoryQueryWidget(QDialog):
    """历史检测记录查询弹窗 — 当前登录用户的审计日志"""

    def __init__(self, username: str, parent=None) -> None:
        super().__init__(parent)
        self.username = username
        # 工人=按批次; 技术员=按每次推理(逐次调试记录)
        self._is_worker = (state.role == "worker")
        self.setWindowTitle(f"历史检测记录 — {username}")
        self.resize(900, 520)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")
        self._build()
        self._reload()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        if self._is_worker:
            cap = "的批次检测记录（每个批次结束后存一条 · 按时间倒序）"
            headers = ["批次号", "品种", "开始时间", "结束时间"]
        else:
            cap = "的逐次检测记录（每推理一次记一条 · 按时间倒序）"
            headers = ["检测时间", "模式", "文件", "总检出", "异物分布", "状态"]
        title = QLabel(f"操作员【{self.username}】{cap}")
        title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:14px; font-weight:bold; padding:6px;")
        lay.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 关掉 alternating row, 避免某些 Qt 版本下 stylesheet 不生效导致白底
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # 双保险: stylesheet + QPalette 都设深色
        self.table.setStyleSheet(
            f"QTableWidget {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"gridline-color:{COLOR_BORDER}; }}"
            f"QTableWidget::item {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"padding:4px; }}"
            f"QTableWidget::item:selected {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
            f"QHeaderView::section {{ background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"padding:4px; border:none; border-right:1px solid {COLOR_BORDER}; }}"
        )
        pal = self.table.palette()
        pal.setColor(QPalette.Base, QColor(COLOR_BG_MAIN))
        pal.setColor(QPalette.AlternateBase, QColor(COLOR_BG_MAIN))
        pal.setColor(QPalette.Text, QColor(COLOR_TEXT))
        pal.setColor(QPalette.Highlight, QColor(COLOR_BTN_PRIMARY))
        pal.setColor(QPalette.HighlightedText, QColor("white"))
        self.table.setPalette(pal)
        lay.addWidget(self.table, 1)

        btn_bar = QHBoxLayout()
        btn_qss = (
            f"QPushButton {{ background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:6px 14px; }}"
            f"QPushButton:hover {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
        )
        refresh_btn = QPushButton("⟳ 刷新"); refresh_btn.setStyleSheet(btn_qss)
        refresh_btn.clicked.connect(self._reload)
        # 清空按钮 — 醒目红色
        clear_btn = QPushButton("🗑 清空记录")
        clear_btn.setStyleSheet(
            "QPushButton { background:#D9534F; color:white; border:none;"
            "border-radius:4px; padding:6px 14px; font-weight:bold; }"
            "QPushButton:hover { background:#E66662; }"
        )
        clear_btn.clicked.connect(self._on_clear)
        export_btn = QPushButton("⬇ 导出 CSV"); export_btn.setStyleSheet(btn_qss)
        export_btn.clicked.connect(self._on_export)
        close_btn = QPushButton("关闭"); close_btn.setStyleSheet(btn_qss)
        close_btn.clicked.connect(self.accept)
        btn_bar.addStretch()
        btn_bar.addWidget(clear_btn); btn_bar.addWidget(export_btn)
        btn_bar.addWidget(refresh_btn); btn_bar.addWidget(close_btn)
        lay.addLayout(btn_bar)

    def _on_export(self) -> None:
        import csv
        from pathlib import Path as _Path
        rows = getattr(self, "_export_rows", [])
        if not rows:
            QMessageBox.information(self, "提示", "暂无记录可导出"); return
        default = f"检测记录_{self.username}.csv"
        fp, _ = QFileDialog.getSaveFileName(self, "导出 CSV", str(_Path.home() / default), "CSV (*.csv)")
        if not fp:
            return
        try:
            with open(fp, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            QMessageBox.information(self, "已导出", f"已保存到\n{fp}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _on_clear(self) -> None:
        count = self.table.rowCount()
        if count == 0:
            QMessageBox.information(self, "提示", "暂无历史记录可清空")
            return
        # 第一道: 确认
        ret = QMessageBox.question(
            self, "确认清空",
            f"将永久删除【{self.username}】的全部 {count} 条历史记录, 此操作不可恢复.\n确认继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        # 第二道: 输入当前用户密码
        pwd, ok = QInputDialog.getText(
            self, "密码校验",
            f"请输入【{self.username}】的登录密码以继续:",
            QLineEdit.Password,
        )
        if not ok:
            return
        if not DbHelper.instance().verify_user(self.username, pwd):
            QMessageBox.warning(self, "密码错误", "密码不正确, 已取消清空")
            return
        db = DbHelper.instance()
        n_log = db.clear_logs(self.username)
        if self._is_worker:
            n_batch = db.delete_user_batches(self.username)
            msg = f"删除了 {n_batch} 个批次 / {n_log} 条检测明细"
        else:
            msg = f"删除了 {n_log} 条检测记录"
        QMessageBox.information(self, "已清空", msg)
        self._reload()

    _MODE_CN = {"photo": "照片", "video": "视频", "folder": "文件夹", "camera": "工业相机"}

    def _reload(self) -> None:
        import json
        db = DbHelper.instance()
        self.table.setRowCount(0)
        self._export_rows = []   # 供导出 CSV
        if self._is_worker:
            # 工人: 批次基本信息(明细在管理员批次追溯里)
            for b in db.list_batches(500, operator=self.username):
                d = {"批次号": b["batch_id"], "品种": b.get("variety") or "—",
                     "开始时间": str(b.get("start_time", "")),
                     "结束时间": str(b.get("end_time") or "进行中")}
                self._export_rows.append(d)
        else:
            # 技术员: 每次推理一条(调试记录)
            for r in db.query_logs(self.username, limit=500):
                try:
                    rs = json.loads(r.get("result_summary") or "{}")
                    dist = "、".join(f"{k}×{v}" for k, v in rs.items()) or "—"
                except Exception:
                    dist = "—"
                d = {"检测时间": str(r.get("start_time", "")),
                     "模式": self._MODE_CN.get(r.get("detect_mode", ""), r.get("detect_mode", "")),
                     "文件": r.get("file_path", "") or "—",
                     "总检出": str(r.get("total_targets", 0)), "异物分布": dist,
                     "状态": r.get("status", "") or "—"}
                self._export_rows.append(d)
        # 填表
        for d in self._export_rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for j, val in enumerate(d.values()):
                self.table.setItem(row, j, QTableWidgetItem(str(val)))


class CameraSettingsDialog(QDialog):
    """大恒相机亮度调节 — 曝光时间(us) + 增益(dB) 双滑块, 实时生效"""

    def __init__(self, camera_mgr, parent=None) -> None:
        super().__init__(parent)
        self.cm = camera_mgr
        self.setWindowTitle("相机亮度调节")
        self.resize(460, 260)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")
        self._build()

    def _build(self) -> None:
        if not self.cm or not self.cm.is_opened():
            lay = QVBoxLayout(self)
            tip = QLabel("⚠ 相机未连接\n\n请先到主界面左侧点击「摄像头」, 启动相机后再调节亮度")
            tip.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px; padding:30px;")
            tip.setAlignment(Qt.AlignCenter)
            lay.addWidget(tip)
            close = QPushButton("关闭"); close.clicked.connect(self.accept)
            close.setStyleSheet(
                f"background:{COLOR_BTN_PRIMARY}; color:white; padding:8px;"
                "border:none; border-radius:4px;"
            )
            lay.addWidget(close)
            return

        cur_exp = self.cm.get_exposure() or 10000.0
        cur_gain = self.cm.get_gain() or 2.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16); lay.setSpacing(12)

        title = QLabel(f"📷 {self.cm.info()}")
        title.setStyleSheet(f"color:{COLOR_BTN_PRIMARY}; font-size:13px; font-weight:bold;")
        lay.addWidget(title)

        # 曝光时间 0.5ms - 100ms 区间, 步长 500us, 显示整数
        self.exp_slider, self.exp_label = self._build_slider_row(
            500, 100000, int(cur_exp), 500,
            fmt=lambda v: str(v),
            on_change=self._on_exposure_changed,
        )
        lay.addLayout(self._wrap_row("曝光时间 (μs)", self.exp_slider, self.exp_label))

        # 增益 0-20 dB, 滑块用 0-200 (10 倍存储), 显示一位小数
        self.gain_slider, self.gain_label = self._build_slider_row(
            0, 200, int(cur_gain * 10), 10,
            fmt=lambda v: f"{v/10:.1f}",
            on_change=self._on_gain_changed,
        )
        lay.addLayout(self._wrap_row("增益 (dB)", self.gain_slider, self.gain_label))

        hint = QLabel("💡 曝光越大画面越亮(易拖影); 增益越大越亮但噪点多. 实时生效, 关闭即保留.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:11px; padding-top:6px;")
        lay.addWidget(hint)

        lay.addStretch()
        btn_row = QHBoxLayout()
        reset = QPushButton("恢复默认")
        reset.setStyleSheet(
            f"background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:6px 14px;"
        )
        reset.clicked.connect(self._reset_default)
        close = QPushButton("关闭")
        close.setStyleSheet(
            f"background:{COLOR_BTN_PRIMARY}; color:white; border:none;"
            "border-radius:4px; padding:6px 18px; font-weight:bold;"
        )
        close.clicked.connect(self.accept)
        btn_row.addWidget(reset); btn_row.addStretch(); btn_row.addWidget(close)
        lay.addLayout(btn_row)

    def _build_slider_row(self, lo, hi, init, step, fmt, on_change):
        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi); sl.setValue(init)
        sl.setSingleStep(step); sl.setPageStep(step * 4)
        lab = QLabel(fmt(init)); lab.setFixedWidth(70)
        # 拖动时按 step 对齐 (snap to step)
        def _on(v):
            snapped = round(v / step) * step
            if snapped != v:
                sl.blockSignals(True); sl.setValue(snapped); sl.blockSignals(False)
            lab.setText(fmt(snapped))
            on_change(snapped)
        sl.valueChanged.connect(_on)
        return sl, lab

    def _wrap_row(self, name: str, sl: QSlider, lab: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        n = QLabel(name); n.setFixedWidth(100); n.setStyleSheet(f"color:{COLOR_TEXT};")
        row.addWidget(n); row.addWidget(sl, 1); row.addWidget(lab)
        return row

    def _on_exposure_changed(self, v: int) -> None:
        self.cm.set_exposure(float(v))

    def _on_gain_changed(self, v: int) -> None:
        self.cm.set_gain(v / 10.0)

    def _reset_default(self) -> None:
        self.exp_slider.setValue(10000)
        self.gain_slider.setValue(20)  # 2.0 dB


class ProfilePanel(QWidget):
    """个人中心右侧面板"""

    history_clicked = Signal()
    edit_clicked = Signal()
    register_clicked = Signal()
    logout_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet(f"background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};")
        self._build()
        self.refresh()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 30, 20, 20); lay.setSpacing(14)

        self.avatar = QLabel()
        self.avatar.setFixedSize(110, 110)
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setStyleSheet(
            f"background:{COLOR_BG_MAIN}; border:2px solid {COLOR_BTN_PRIMARY};"
            f"border-radius:55px; color:{COLOR_BTN_PRIMARY};"
            f"font-size:42px; font-weight:bold;"
        )
        wrap = QHBoxLayout(); wrap.addStretch(); wrap.addWidget(self.avatar); wrap.addStretch()
        lay.addLayout(wrap)

        self.name_label = QLabel("—")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("font-size:18px; font-weight:bold;")
        lay.addWidget(self.name_label)

        sub = QLabel("操作员账号")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{COLOR_TEXT_DIM};")
        lay.addWidget(sub)

        lay.addSpacing(20)
        # 去掉「注册新用户」(账号统一由管理员开通) 和「退出登录」(已挪到顶栏明面)
        for text, sig in [
            ("修改信息", self.edit_clicked),
            ("历史检测记录", self.history_clicked),
        ]:
            btn = QPushButton(text); btn.setMinimumHeight(38)
            btn.setStyleSheet(
                f"background:{COLOR_BTN_PRIMARY}; color:white;"
                f"border-radius:4px; font-size:13px;"
            )
            btn.clicked.connect(sig); lay.addWidget(btn)

        lay.addStretch()

    def refresh(self) -> None:
        name = state.username or "—"
        self.name_label.setText(name)
        if state.avatar_path:
            pix = QPixmap(state.avatar_path)
            if not pix.isNull():
                self.avatar.setPixmap(make_round_pixmap(pix, 106))
                self.avatar.setText("")
                return
        self.avatar.setPixmap(QPixmap())
        self.avatar.setText(name[:1].upper() if name else "?")


class WorkerDashboard(QWidget):
    """工人看护页右侧大字看板 — 工人不需要坐标/置信度, 只要一眼看清生产状态。

    显示: 批次号 / 运行时长 / 本批次累计检出 / 严重异物次数 / 各严重类计数 / 状态。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet(f"background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};")
        self._build()

    def _big(self, color: str) -> QLabel:
        lb = QLabel("0")
        lb.setAlignment(Qt.AlignCenter)
        lb.setStyleSheet(f"color:{color}; font-size:46px; font-weight:bold;")
        return lb

    def _cap(self, text: str) -> QLabel:
        lb = QLabel(text)
        lb.setAlignment(Qt.AlignCenter)
        lb.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:13px;")
        return lb

    def _card(self, title: str, big: QLabel, lay) -> None:
        box = QVBoxLayout(); box.setSpacing(2)
        box.addWidget(self._cap(title)); box.addWidget(big)
        frame = QWidget()
        frame.setStyleSheet(
            f"background:{COLOR_BG_MAIN}; border:1px solid {COLOR_BORDER}; border-radius:6px;")
        frame.setLayout(box)
        lay.addWidget(frame)

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 18, 14, 18); lay.setSpacing(12)

        title = QLabel("生产看护")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:18px; font-weight:bold;")
        lay.addWidget(title)

        self.batch_lb = QLabel("批次: 未开始")
        self.batch_lb.setAlignment(Qt.AlignCenter)
        self.batch_lb.setStyleSheet("color:#FFD23F; font-size:15px; font-weight:bold;")
        lay.addWidget(self.batch_lb)

        self.time_lb = QLabel("已运行 00:00:00")
        self.time_lb.setAlignment(Qt.AlignCenter)
        self.time_lb.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:12px;")
        lay.addWidget(self.time_lb)

        self.total_big = self._big(COLOR_BTN_PRIMARY)
        self._card("本批次累计检出(个)", self.total_big, lay)

        self.severe_big = self._big("#E66662")
        self._card("严重异物报警(次)", self.severe_big, lay)

        lay.addWidget(self._cap("严重类别明细"))
        self.detail_lb = QLabel("—")
        self.detail_lb.setWordWrap(True)
        self.detail_lb.setAlignment(Qt.AlignTop)
        self.detail_lb.setFont(QFont("Microsoft YaHei", 11))
        self.detail_lb.setStyleSheet(
            f"color:{COLOR_TEXT}; background:{COLOR_BG_MAIN};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:8px; min-height:70px;")
        lay.addWidget(self.detail_lb, 1)

        self.hint_lb = QLabel("点「开始批次」开始看护")
        self.hint_lb.setWordWrap(True)
        self.hint_lb.setAlignment(Qt.AlignCenter)
        self.hint_lb.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:12px;")
        lay.addWidget(self.hint_lb)

    def set_batch(self, batch_id: str, running: bool) -> None:
        self.batch_lb.setText(f"批次: {batch_id}" if batch_id else "批次: 未开始")
        self.hint_lb.setText("检测中…发现严重异物会红屏报警" if running else "点「开始批次」开始看护")
        if not running:
            self.time_lb.setText("已运行 00:00:00")

    def update_counts(self, total: int, severe_count: int,
                      per_severe: dict[str, int], elapsed_sec: float) -> None:
        self.total_big.setText(str(total))
        self.severe_big.setText(str(severe_count))
        if per_severe:
            self.detail_lb.setText("\n".join(f"{k}：{v} 次" for k, v in per_severe.items()))
        else:
            self.detail_lb.setText("暂无严重异物")
        h = int(elapsed_sec // 3600); m = int((elapsed_sec % 3600) // 60); s = int(elapsed_sec % 60)
        self.time_lb.setText(f"已运行 {h:02d}:{m:02d}:{s:02d}")


class DetectionSummaryDialog(QDialog):
    """本次会话检测总计 — 各类"单帧最多同时出现"峰值 + 总计, 可保存 CSV

    口径: peak[类别] = 整段会话里该类在单帧中出现过的最大数量 (非逐帧累加).
    """

    _MODE_CN = {"photo": "照片", "video": "视频", "folder": "文件夹批量", "camera": "工业相机"}

    def __init__(self, mode: str, peak: dict[str, int], on_save=None, parent=None) -> None:
        super().__init__(parent)
        self._peak = {k: v for k, v in peak.items() if v > 0}
        self._mode_cn = self._MODE_CN.get(mode, mode)
        self._on_save_cb = on_save  # 主窗口的'本次保存'(带 汇总/明细/两者 选择); None 则只存汇总
        self.setWindowTitle("本次检测总计")
        self.resize(420, 460)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)

        title = QLabel(f"本次检测总计 — {self._mode_cn}")
        title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:16px; font-weight:bold;")
        lay.addWidget(title)

        sub = QLabel(f"共检测到异物 {sum(self._peak.values())} 个（按单帧最多同时出现计）")
        sub.setStyleSheet(f"color:{COLOR_BTN_PRIMARY}; font-size:13px;")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        if not self._peak:
            empty = QLabel("本次未检测到异物")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(f"color:{COLOR_TEXT_DIM}; padding:40px; font-size:14px;")
            lay.addWidget(empty, 1)
        else:
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["类别", "最多同时数量"])
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.NoSelection)
            table.setAlternatingRowColors(False)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            table.setStyleSheet(
                f"QTableWidget {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
                f"gridline-color:{COLOR_BORDER}; }}"
                f"QTableWidget::item {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT}; padding:6px; }}"
                f"QHeaderView::section {{ background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
                f"padding:6px; border:none; border-right:1px solid {COLOR_BORDER}; }}"
            )
            pal = table.palette()
            pal.setColor(QPalette.Base, QColor(COLOR_BG_MAIN))
            pal.setColor(QPalette.AlternateBase, QColor(COLOR_BG_MAIN))
            pal.setColor(QPalette.Text, QColor(COLOR_TEXT))
            table.setPalette(pal)
            rows = sorted(self._peak.items(), key=lambda kv: -kv[1])
            table.setRowCount(len(rows))
            for i, (cls, n) in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(cls))
                table.setItem(i, 1, QTableWidgetItem(str(n)))
            lay.addWidget(table, 1)

        btn_bar = QHBoxLayout()
        save_btn = QPushButton("💾 保存结果")
        save_btn.setMinimumHeight(36)
        save_btn.setStyleSheet(
            f"QPushButton {{ background:{COLOR_BTN_PRIMARY}; color:white; border:none;"
            f"border-radius:4px; padding:6px 16px; font-weight:bold; }}"
            f"QPushButton:hover {{ background:#3E9CFF; }}"
        )
        save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("关闭")
        close_btn.setMinimumHeight(36)
        close_btn.setStyleSheet(
            f"QPushButton {{ background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:6px 16px; }}"
            f"QPushButton:hover {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
        )
        close_btn.clicked.connect(self.accept)
        btn_bar.addWidget(save_btn)
        btn_bar.addStretch()
        btn_bar.addWidget(close_btn)
        lay.addLayout(btn_bar)

    def _on_save(self) -> None:
        # 交给主窗口统一的'本次保存'(带 仅汇总/仅明细/两者 选择, 与右侧导出一致)
        if self._on_save_cb is not None:
            self._on_save_cb()
            return
        # 兜底: 无回调时只存汇总
        if not self._peak:
            QMessageBox.information(self, "提示", "本次无检测结果可保存")
            return
        import time
        default = f"检测总计_{self._mode_cn}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        fp, _ = QFileDialog.getSaveFileName(self, "保存总计结果", default, "CSV (*.csv)")
        if not fp:
            return
        try:
            Exporter.write_summary_csv(fp, self._peak)
            QMessageBox.information(self, "已保存", f"总计结果已保存\n{fp}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
