"""Locate and import the bundled Fairino SDK.

The Fairino ``Robot`` module is not on PyPI; it ships as ``Robot.py`` inside the
vendored ``fairino_sdk/`` tree at the repo root. This module wires that path onto
``sys.path`` once and re-exports the ``Robot`` module so callers can simply do::

    from omnicron_ui.robot.sdk import Robot
    rpc = Robot.RPC("192.168.58.2")

Keeping the path juggling in one place means the rest of the app never has to
know where the SDK lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo root = .../omnicron-UI (three parents up from this file:
# src/omnicron_ui/robot/sdk.py -> robot -> omnicron_ui -> src -> <root>)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SDK_DIR = _REPO_ROOT / "fairino_sdk" / "linux" / "fairino"


def _ensure_on_path() -> None:
    if not _SDK_DIR.is_dir():
        raise FileNotFoundError(
            f"Fairino SDK not found at {_SDK_DIR}. Expected the vendored SDK under "
            "fairino_sdk/linux/fairino/ containing Robot.py."
        )
    sdk_dir = str(_SDK_DIR)
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)


_ensure_on_path()

import Robot  # noqa: E402  (import after sys.path is set up)

__all__ = ["Robot"]
