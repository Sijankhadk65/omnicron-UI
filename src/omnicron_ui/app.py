"""Application entry point.

``main()`` boots the Qt application and shows the main window. This is the
console script target (``omnicron-ui``) and also what ``uv run omnicron-ui`` and
``uv run python -m omnicron_ui`` resolve to.
"""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from omnicron_ui.main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Omnicron")
    app.setApplicationDisplayName("Omnicron")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
