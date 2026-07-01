"""A read-only, color-coded scrolling log panel.

Connect ``RobotService.log`` (or any ``(message, level)`` signal) to
``LogPanel.append``. Levels map to colors: info/success/warn/error.
"""

from __future__ import annotations

import html
from datetime import datetime

from PySide6 import QtWidgets

_LEVEL_COLORS = {
    "info": "#d0d0d0",
    "success": "#4caf50",
    "warn": "#ffb300",
    "error": "#ef5350",
    "header": "#42a5f5",
}


class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setStyleSheet(
            "QPlainTextEdit { background:#1a1a1a; font-family:monospace; }"
        )

    def append(self, message: str, level: str = "info") -> None:
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
        ts = datetime.now().strftime("%H:%M:%S")
        safe = html.escape(message)
        self.appendHtml(
            f'<span style="color:#666;">{ts}</span> '
            f'<span style="color:{color};">{safe}</span>'
        )
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
