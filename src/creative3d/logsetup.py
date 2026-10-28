"""logsetup.py —— 日志初始化（滚动文件，写用户目录，永不因日志失败影响主流程）。"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from . import config

_LOGGER_NAME = "creative3d"
_initialized = False


def _safe_dir(path: Path) -> Path | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def setup(verbose: bool = False) -> logging.Logger:
    global _initialized
    logger = logging.getLogger(_LOGGER_NAME)
    if _initialized:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    target = _safe_dir(config.log_dir())
    if target is not None:
        try:
            fh = logging.handlers.RotatingFileHandler(
                target / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass

    # 打包后无控制台，stderr 可能为 None
    if verbose and sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    _initialized = True
    return logger


def get(name: str = "") -> logging.Logger:
    base = logging.getLogger(_LOGGER_NAME)
    return base.getChild(name) if name else base


def log_path() -> Path:
    return config.log_dir() / "app.log"
