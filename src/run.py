"""run.py —— 源码运行入口（打包后由 exe 直接调用 creative3d.app:main）。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from creative3d.app import main

if __name__ == "__main__":
    raise SystemExit(main())
