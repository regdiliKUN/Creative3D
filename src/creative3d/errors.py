"""errors.py —— 统一异常体系。

目标：任何异常最终都能被翻译成「用户看得懂的一句话 + 可操作的建议」，
而不是把 Python traceback 直接甩到界面上。
"""

from __future__ import annotations


class AppError(Exception):
    """应用级异常基类。

    message : 给人看的一句话（不带技术细节）
    hint    : 可操作建议（可以为空）
    detail  : 技术细节，写日志用
    """

    code = "APP_ERROR"
    title = "操作失败"

    def __init__(self, message: str, *, hint: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail or message

    def user_text(self) -> str:
        if self.hint:
            return f"{self.message}\n\n建议：{self.hint}"
        return self.message


class ValidationError(AppError):
    code = "VALIDATION"
    title = "参数不合法"


class ImageError(AppError):
    code = "IMAGE"
    title = "图片无法使用"


class ExportError(AppError):
    code = "EXPORT"
    title = "导出失败"


class GenerationError(AppError):
    code = "GENERATE"
    title = "生成失败"


class ShortcutError(AppError):
    code = "SHORTCUT"
    title = "快捷方式创建失败"


class SettingsError(AppError):
    code = "SETTINGS"
    title = "设置读写失败"


# 常见系统错误 → 人话
_OS_HINTS = {
    13: "没有写入权限，请在设置里换一个导出目录（例如桌面）。",
    28: "磁盘空间不足，清理后再试。",
    112: "磁盘空间不足，清理后再试。",
    5: "文件被占用或权限不足，请关闭正在使用该文件的程序。",
    32: "文件正被其它程序占用，请稍后重试。",
    2: "找不到目标路径，可能目录已被移动或删除。",
}


def humanize_oserror(exc: OSError, *, action: str = "写入文件") -> str:
    """把 OSError 翻译成中文提示 + 建议。"""
    errno = getattr(exc, "errno", None)
    hint = _OS_HINTS.get(errno or -1, "")
    if not hint:
        hint = "若问题持续，请打开「帮助 → 查看日志」把日志发给开发者。"
    name = getattr(exc, "filename", "") or ""
    where = f"（{name}）" if name else ""
    return f"{action}时出错{where}：{exc.strerror or exc}"
