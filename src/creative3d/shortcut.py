"""shortcut.py —— 创建 / 修复桌面快捷方式。

为什么必须真的写 .lnk：
    Windows 资源管理器的「打开文件所在位置」读的是 .lnk 里的 TargetPath，
    只有把 TargetPath 写成 exe 的绝对路径，右键才会落到包体目录。
    用 .url 或 bat 包装都做不到。

实现走 PowerShell 的 WScript.Shell（系统自带、兼容性最好），
写完会回读校验，失败时给出明确原因。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config
from .errors import ShortcutError

APP_LINK_NAME = "3D 建模创意生成器"


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def desktop_dir() -> Path:
    """优先读注册表里的真实桌面（可能被重定向到 D:\\Desktop）。"""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        )
        with key:
            value, _ = winreg.QueryValueEx(key, "Desktop")
            path = Path(os.path.expandvars(value))
            if path.exists():
                return path
    except Exception:  # noqa: BLE001 - 注册表读取失败时退回环境变量
        pass
    env = os.environ.get("USERPROFILE")
    if env:
        return Path(env) / "Desktop"
    return Path.home() / "Desktop"


def shortcut_path(name: str = APP_LINK_NAME) -> Path:
    return desktop_dir() / f"{name}.lnk"


def launcher_path() -> Path:
    if config.is_frozen():
        return Path(sys.executable).resolve()
    return config.app_dir() / "run.py"


def _run_ps(script: str, timeout: int = 25) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError as exc:
        raise ShortcutError(
            "系统里找不到 PowerShell，无法创建快捷方式。",
            hint="可以手动把 exe 拖到桌面创建快捷方式。",
            detail=str(exc),
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ShortcutError(
            "创建快捷方式超时。",
            hint="请稍后重试；若持续失败，可手动把 exe 拖到桌面。",
            detail=str(exc),
        ) from exc


def create_desktop_shortcut(*, force: bool = False, name: str = APP_LINK_NAME) -> Path:
    target = launcher_path()
    link = shortcut_path(name)

    if link.exists() and not force:
        return link

    workdir = target.parent
    icon = target if target.suffix.lower() == ".exe" else config.resource_path("assets", "icon.ico")
    icon_expr = f"{_ps_quote(str(icon))},0" if str(icon).lower().endswith(".ico") else f"{_ps_quote(str(icon))},0"

    script = (
        f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(str(link))});"
        f"$s.TargetPath={_ps_quote(str(target))};"
        f"$s.WorkingDirectory={_ps_quote(str(workdir))};"
        f"$s.IconLocation={icon_expr};"
        f"$s.Description={_ps_quote('参数化 3D 创意生成器')};"
        "$s.Save();"
        f"if (Test-Path -LiteralPath {_ps_quote(str(link))}) {{ 'OK' }} else {{ 'FAIL' }}"
    )
    code, out = _run_ps(script)
    if code != 0 or "OK" not in out or not link.exists():
        raise ShortcutError(
            "快捷方式没有创建成功。",
            hint="可能是安全软件拦截了脚本执行；可手动把 exe 拖到桌面。",
            detail=out.strip()[:400],
        )
    return link


def read_shortcut(path: Path | None = None) -> dict[str, str]:
    link = path or shortcut_path()
    if not link.exists():
        return {}
    script = (
        f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(str(link))});"
        "'TARGET='+$s.TargetPath;'WORKDIR='+$s.WorkingDirectory"
    )
    code, out = _run_ps(script, timeout=15)
    data: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("TARGET="):
            data["target"] = line[7:].strip()
        elif line.startswith("WORKDIR="):
            data["workdir"] = line[8:].strip()
    return data
