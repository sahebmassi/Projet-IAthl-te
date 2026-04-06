import os
import sys


def _configure_qt_platform() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("QT_QPA_PLATFORM"):
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            os.environ["QT_QPA_PLATFORM"] = "xcb"


_configure_qt_platform()

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
