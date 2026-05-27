"""程序入口 — 闪屏 → 模型加载 → 数据库初始化 → 登录窗 → 主窗口"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根可作为 import root (脚本运行场景)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from core.app_state import state
from db.db_helper import DbHelper
from ui.login_window import LoginWindow
from ui.main_window import MainWindow
from ui.splash_screen import SplashScreen
from utils.common import get_logger

log = get_logger("main")


def _bootstrap(splash: SplashScreen) -> bool:
    """初始化流程: 数据库 + 推理引擎"""
    splash.set_tip("正在初始化数据库...")
    try:
        DbHelper.instance()
    except Exception as e:
        log.error("数据库初始化失败: %s", e)
        QMessageBox.critical(None, "启动失败", f"数据库初始化失败:\n{e}")
        return False

    splash.set_tip("正在加载推理引擎...")
    try:
        from inference import BACKEND, DEVICE_NAME, load_model
        state.backend = BACKEND
        state.device_name = DEVICE_NAME
        load_model()
        log.info("推理引擎就绪: %s @ %s", BACKEND, DEVICE_NAME)
    except Exception as e:
        log.error("模型加载失败: %s", e)
        QMessageBox.warning(
            None, "模型加载警告",
            f"推理引擎初始化失败:\n{e}\n\n程序仍可启动, 但检测功能不可用",
        )
        # 不阻断启动 — 允许只看 UI / 历史日志

    splash.set_tip("准备就绪")
    return True


class _AppController:
    """串接登录 → 主窗口 → 注销返登录 的简单控制器"""

    def __init__(self) -> None:
        self.login: LoginWindow | None = None
        self.main: MainWindow | None = None

    def show_login(self) -> None:
        self.login = LoginWindow()
        self.login.login_success.connect(self._on_login_success)
        self.login.show()

    def _on_login_success(self, username: str) -> None:
        if self.login is not None:
            self.login.close()
            self.login = None
        self.main = MainWindow()
        self.main.init_for_user(username)
        self.main.logout_requested.connect(self._on_logout)
        self.main.show()

    def _on_logout(self) -> None:
        # 清空当前用户
        state.username = ""
        state.avatar_path = ""
        # 先打开登录窗 (保证不是"最后一个窗口"), 再关主窗 — 避免触发 quitOnLastWindowClosed
        self.show_login()
        if self.main is not None:
            self.main.close()
            self.main = None


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("烟草异物智能检测系统")

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    if not _bootstrap(splash):
        return 1

    controller = _AppController()
    QTimer.singleShot(400, lambda: (splash.close(), controller.show_login()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
