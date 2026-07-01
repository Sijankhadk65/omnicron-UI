# Omnicron

PySide6 operator console for the **Fairino welding robot**.

## Setup

```bash
uv sync          # create the venv and install deps (first time)
```

## Run

```bash
uv run omnicron-ui          # console script
# or
uv run python -m omnicron_ui
```

## Layout

| Path | Role |
|------|------|
| `src/omnicron_ui/app.py` | Application entry point (`main()`) |
| `src/omnicron_ui/main_window.py` | `MainWindow` — the top-level window |
| `src/omnicron_ui/robot/sdk.py` | Loads the bundled Fairino `Robot` SDK |
| `fairino_sdk/` | Vendored Fairino Python SDK (not on PyPI) |

The Fairino `Robot` module is imported directly from `fairino_sdk/linux/fairino/`
via `omnicron_ui.robot.sdk`, so the rest of the app never touches `sys.path`.
