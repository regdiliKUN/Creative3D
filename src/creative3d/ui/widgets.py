"""ui.widgets —— 一套自绘深色控件（tk 原生控件在 Windows 上很难统一配色）。

包含：按钮 / 图标按钮 / 标签页 / 芯片 / 自定义滑杆 / 开关 / 滚动容器 /
提示气泡 / Toast / 进度条 / 区块标题 / 悬停提示。

原则：任何回调异常都在内部吞掉并记录，绝不让一次误触把界面带崩。
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable

from .. import config
from . import theme

T = config.THEME
log = logging.getLogger("creative3d.ui.widgets")


def _safe(fn: Callable | None, *args) -> None:
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("控件回调失败")


# --------------------------------------------------------------------------
# 悬停提示
# --------------------------------------------------------------------------


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, *, delay: int = 420, wrap: int = 260) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wrap = wrap
        self._after: str | None = None
        self._win: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        try:
            self._after = self.widget.after(self.delay, self._show)
        except tk.TclError:
            pass

    def _cancel(self) -> None:
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self) -> None:
        if self._win is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except tk.TclError:
            return
        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        win.wm_attributes("-topmost", True)
        frame = tk.Frame(win, bg=T.line, bd=0)
        frame.pack()
        tk.Label(
            frame, text=self.text, bg=T.panel_alt, fg=T.text_dim,
            font=theme.f(9), justify="left", wraplength=self.wrap, padx=10, pady=7,
        ).pack(padx=1, pady=1)
        win.update_idletasks()
        # 防止超出屏幕右/下边缘
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = win.winfo_width(), win.winfo_height()
        if x + w > sw - 8:
            x = sw - w - 8
        if y + h > sh - 8:
            y = self.widget.winfo_rooty() - h - 8
        win.wm_geometry(f"+{max(0, x)}+{max(0, y)}")
        self._win = win

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._win is not None:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None


# --------------------------------------------------------------------------
# 按钮
# --------------------------------------------------------------------------


class Button(tk.Label):
    """扁平按钮。kind: primary / secondary / ghost / danger"""

    _KINDS = {
        "primary": (T.accent, "#ffffff", T.accent_hover),
        "secondary": (T.panel_alt, T.text, T.line),
        "ghost": (T.panel, T.text_dim, T.panel_alt),
        "danger": ("#3a2230", "#ff9a9a", "#4a2838"),
        "success": ("#1c3a30", "#7ff0c4", "#24473b"),
    }

    def __init__(self, master, text: str, command: Callable | None = None, *,
                 kind: str = "secondary", icon: str = "", padx: int = 12, pady: int = 7,
                 font_size: int = 10, bold: bool = False, anchor: str = "center",
                 width: int | None = None) -> None:
        bg, fg, hover = self._KINDS.get(kind, self._KINDS["secondary"])
        self._bg, self._fg, self._hover = bg, fg, hover
        label = f"{icon} {text}".strip() if icon else text
        super().__init__(
            master, text=label, bg=bg, fg=fg, font=theme.f(font_size, bold),
            padx=padx, pady=pady, cursor="hand2", anchor=anchor,
            bd=0, highlightthickness=0, activebackground=hover, activeforeground=fg,
            justify="center",
        )
        if width:
            self.configure(width=width)
        self._command = command
        self._enabled = True
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<Button-1>", self._on_click, add="+")

    def _on_enter(self, _e=None):
        if self._enabled:
            self.configure(bg=self._hover)

    def _on_leave(self, _e=None):
        if self._enabled:
            self.configure(bg=self._bg)

    def _on_click(self, _e=None):
        if self._enabled:
            _safe(self._command)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self.configure(
            bg=self._bg if enabled else T.panel,
            fg=self._fg if enabled else T.text_mute,
            cursor="hand2" if enabled else "arrow",
        )

    def set_text(self, text: str) -> None:
        self.configure(text=text)


class IconButton(tk.Label):
    def __init__(self, master, glyph: str, command: Callable | None = None, *,
                 tip: str = "", size: int = 15, padx: int = 7, pady: int = 4) -> None:
        super().__init__(master, text=glyph, bg=T.panel, fg=T.text_dim,
                         font=theme.f(size), padx=padx, pady=pady, cursor="hand2",
                         bd=0, highlightthickness=0, activebackground=T.panel_alt)
        self._command = command
        self.bind("<Enter>", lambda e: self.configure(bg=T.panel_alt, fg=T.text), add="+")
        self.bind("<Leave>", lambda e: self.configure(bg=T.panel, fg=T.text_dim), add="+")
        self.bind("<Button-1>", lambda e: _safe(self._command), add="+")
        if tip:
            Tooltip(self, tip)


# --------------------------------------------------------------------------
# 芯片 / 分段控件
# --------------------------------------------------------------------------


class Chip(tk.Label):
    def __init__(self, master, text: str, command: Callable | None = None, *,
                 selected: bool = False, glyph: str = "", padx: int = 10, pady: int = 6,
                 font_size: int = 9, wraplength: int = 0, bg: str | None = None,
                 anchor: str = "center") -> None:
        super().__init__(
            master, text=f"{glyph} {text}".strip() if glyph else text,
            bg=T.accent_soft if selected else (bg or T.panel_alt),
            fg=T.accent if selected else T.text_dim,
            font=theme.f(font_size, selected), padx=padx, pady=pady,
            cursor="hand2", bd=0, highlightthickness=0, justify="center",
            anchor=anchor,
        )
        self._command = command
        self._selected = selected
        self._text = f"{glyph} {text}".strip() if glyph else text
        if wraplength:
            self.configure(wraplength=wraplength)
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<Button-1>", self._on_click, add="+")

    def _on_enter(self, _e=None):
        if not self._selected:
            self.configure(bg=T.line)

    def _on_leave(self, _e=None):
        if not self._selected:
            self.configure(bg=T.panel_alt)
            self.configure(fg=T.text_dim)

    def _on_click(self, _e=None):
        _safe(self._command)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.configure(
            bg=T.accent_soft if selected else T.panel_alt,
            fg=T.accent if selected else T.text_dim,
            font=theme.f(9, selected),
        )


class Segmented(tk.Frame):
    """两段式切换（文本 / 图片）。"""

    def __init__(self, master, labels: list[str], on_change: Callable[[int], None], *, bg: str | None = None) -> None:
        super().__init__(master, bg=bg or T.panel_alt, bd=0, highlightthickness=0)
        self._on_change = on_change
        self._buttons: list[tk.Label] = []
        self._index = 0
        for i, text in enumerate(labels):
            lbl = tk.Label(self, text=text, bg=bg or T.panel_alt, fg=T.text_dim,
                           font=theme.f(9), padx=14, pady=6, cursor="hand2")
            lbl.pack(side="left", padx=2, pady=2)
            lbl.bind("<Button-1>", lambda e, i=i: self.select(i))
            self._buttons.append(lbl)
        self.select(0, notify=False)

    def select(self, index: int, *, notify: bool = True) -> None:
        index = max(0, min(len(self._buttons) - 1, index))
        for i, lbl in enumerate(self._buttons):
            lbl.configure(
                bg=T.accent_soft if i == index else T.panel_alt,
                fg=T.accent if i == index else T.text_dim,
                font=theme.f(9, i == index),
            )
        changed = index != self._index
        self._index = index
        if notify and changed:
            _safe(self._on_change, index)

    @property
    def index(self) -> int:
        return self._index


# --------------------------------------------------------------------------
# 滑杆
# --------------------------------------------------------------------------


class Slider(tk.Canvas):
    def __init__(self, master, *, lo: float = 0, hi: float = 100, value: float = 50,
                 step: float = 1, command: Callable[[float], None] | None = None,
                 release: Callable[[float], None] | None = None,
                 height: int = 26, bg: str | None = None) -> None:
        super().__init__(master, height=height, bg=bg or T.panel, highlightthickness=0, bd=0,
                         cursor="hand2")
        self.lo, self.hi = float(lo), float(hi)
        self.step = float(step) if step else 0.0
        self.value = float(value)
        self.command = command
        self.release = release
        self._hover = False
        self._dragging = False
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<MouseWheel>", self._on_wheel)

    # ------------------------------------------------------------ 绘制
    def _fraction(self) -> float:
        span = self.hi - self.lo
        return 0.0 if span <= 0 else max(0.0, min(1.0, (self.value - self.lo) / span))

    def redraw(self) -> None:
        self.delete("all")
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        cy = h / 2
        pad = 9
        x0, x1 = pad, max(pad + 1, w - pad)
        theme.rounded_rect(self, x0, cy - 2, x1, cy + 2, r=2, fill=T.line, outline="")
        fx = x0 + (x1 - x0) * self._fraction()
        if fx > x0 + 1:
            theme.rounded_rect(self, x0, cy - 2, fx, cy + 2, r=2, fill=T.accent, outline="")
        r = 7 if (self._hover or self._dragging) else 6
        self.create_oval(fx - r, cy - r, fx + r, cy + r,
                         fill="#ffffff" if self._dragging else T.text,
                         outline=T.accent, width=2)

    def _on_enter(self, _e=None):
        self._hover = True
        self.redraw()

    def _on_leave(self, _e=None):
        self._hover = False
        self.redraw()

    # ------------------------------------------------------------ 交互
    def _value_at(self, x: float) -> float:
        w = max(1, self.winfo_width())
        pad = 9
        span = max(1, w - 2 * pad)
        frac = max(0.0, min(1.0, (x - pad) / span))
        value = self.lo + frac * (self.hi - self.lo)
        if self.step > 0:
            value = round(value / self.step) * self.step
        return max(self.lo, min(self.hi, value))

    def _on_press(self, event):
        self._dragging = True
        self.set_value(self._value_at(event.x), notify=True)
        return "break"

    def _on_drag(self, event):
        if not self._dragging:
            self._dragging = True
        self.set_value(self._value_at(event.x), notify=True)
        return "break"

    def _on_release(self, _event=None):
        if not self._dragging:
            return
        self._dragging = False
        self.redraw()
        _safe(self.release, self.value)

    def _on_wheel(self, event):
        delta = 1 if event.delta > 0 else -1
        step = self.step or 1
        self.set_value(self.value + delta * step * 2, notify=True)
        _safe(self.release, self.value)

    def set_value(self, value: float, *, notify: bool = False) -> None:
        value = max(self.lo, min(self.hi, float(value)))
        if abs(value - self.value) < 1e-9:
            self.redraw()
            return
        self.value = value
        self.redraw()
        if notify:
            _safe(self.command, value)


class Toggle(tk.Canvas):
    def __init__(self, master, *, value: bool = False, command: Callable[[bool], None] | None = None,
                 width: int = 40, height: int = 22) -> None:
        super().__init__(master, width=width, height=height, bg=T.panel,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._value = bool(value)
        self._command = command
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = int(self["width"])
        h = int(self["height"])
        on = self._value
        theme.rounded_rect(self, 1, 3, w - 1, h - 3, r=(h - 6) / 2,
                           fill=T.accent if on else T.line, outline="")
        cx = w - h / 2 - 1 if on else h / 2 + 1
        self.create_oval(cx - (h - 8) / 2, h / 2 - (h - 8) / 2,
                         cx + (h - 8) / 2, h / 2 + (h - 8) / 2,
                         fill="#ffffff", outline="")

    def _toggle(self, _e=None):
        self._value = not self._value
        self._draw()
        _safe(self._command, self._value)

    def set_value(self, value: bool, *, notify: bool = False) -> None:
        self._value = bool(value)
        self._draw()
        if notify:
            _safe(self._command, self._value)


# --------------------------------------------------------------------------
# 滚动容器
# --------------------------------------------------------------------------


class ScrollFrame(tk.Frame):
    """竖向滚动容器（tk 原生没有，自己搭一个；鼠标滚轮只作用于本控件）。

    两个关键细节：
      * scrollregion 的宽永远锁成画布宽，内容再宽也不会把界面横向拖歪；
      * 只在内容真的超出时才接管滚轮，否则会把事件吞掉、让外层收不到。
    """

    def __init__(self, master, *, bg: str | None = None) -> None:
        super().__init__(master, bg=bg or T.panel, bd=0, highlightthickness=0)
        self.canvas = tk.Canvas(self, bg=bg or T.panel, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg or T.panel)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        # 细长的滚动指示条，只在溢出时出现
        self.indicator = tk.Canvas(self, width=5, bg=bg or T.panel,
                                   highlightthickness=0, bd=0)
        self._overflow = False

    def _on_inner(self, _e=None) -> None:
        self._sync_scrollregion()

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self._sync_scrollregion()

    def _sync_scrollregion(self) -> None:
        try:
            width = self.canvas.winfo_width()
            height = max(self.inner.winfo_reqheight(), self.inner.winfo_height())
        except tk.TclError:
            return
        if width <= 1:
            return
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self._sync_indicator(height)

    def _sync_indicator(self, content_height: int) -> None:
        height = max(1, self.canvas.winfo_height())
        overflow = content_height > height + 2
        if overflow == self._overflow:
            if overflow:
                self._draw_thumb(content_height, height)
            return
        self._overflow = overflow
        if not overflow:
            self.indicator.pack_forget()
            return
        self.indicator.pack(side="right", fill="y", padx=(0, 3), pady=3)
        self._draw_thumb(content_height, height)

    def _draw_thumb(self, content_height: int, viewport: int) -> None:
        self.indicator.delete("all")
        h = max(6, self.indicator.winfo_height())
        thumb = max(16, int(h * viewport / max(content_height, 1)))
        first, _last = self.canvas.yview()
        top = first * (h - thumb)
        theme.rounded_rect(self.indicator, 0, top, 5, top + thumb, r=2.5,
                           fill=T.thumb, outline="")

    def _bind_wheel(self, _e=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_wheel_linux)

    def _unbind_wheel(self, _e=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel_linux(self, event) -> None:
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

    def _on_wheel(self, event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        # Windows 的高精度触控板 delta 是小数倍，按 120 归一后至少滚 1 行
        steps = delta / 120.0
        if -1 < steps < 1:
            steps = 1.0 if steps > 0 else -1.0
        self.canvas.yview_scroll(int(-steps), "units")
        self._sync_indicator(max(self.inner.winfo_reqheight(), self.inner.winfo_height()))

    def scroll_to_top(self) -> None:
        self.canvas.yview_moveto(0.0)

    def ensure_visible(self, widget: tk.Widget) -> None:
        """把某个子控件滚到可见范围（切换形态后自动定位到相关分区时用）。"""
        try:
            self.update_idletasks()
            top = widget.winfo_y()
            bottom = top + widget.winfo_height()
            height = max(1, self.canvas.winfo_height())
            first, last = self.canvas.yview()
            view_top = first * self.inner.winfo_height()
            view_bottom = view_top + height
        except tk.TclError:
            return
        if top < view_top:
            self.canvas.yview_moveto(max(0.0, (top - 8) / max(1, self.inner.winfo_height())))
        elif bottom > view_bottom:
            self.canvas.yview_moveto(max(0.0, (bottom - height + 8)
                                         / max(1, self.inner.winfo_height())))


# --------------------------------------------------------------------------
# 提示 / 进度
# --------------------------------------------------------------------------


class Toast(tk.Frame):
    """从底部浮出的轻提示，自动消失，不打断操作。"""

    def __init__(self, master) -> None:
        super().__init__(master, bg=T.panel_alt, bd=0, highlightthickness=1,
                         highlightbackground=T.line)
        self._label = tk.Label(self, text="", bg=T.panel_alt, fg=T.text,
                               font=theme.f(9), padx=16, pady=9, justify="left")
        self._label.pack()
        self._after: str | None = None
        self._queue: list[tuple[str, str, int]] = []

    def show(self, text: str, *, kind: str = "info", duration: int = 2600) -> None:
        color = {"ok": T.ok, "warn": T.warn, "err": T.err}.get(kind, T.text)
        self._queue.append((text, color, duration))
        if self._after is None:
            self._next()

    def _next(self) -> None:
        if not self._queue:
            self._after = None
            self.place_forget()
            return
        text, color, duration = self._queue.pop(0)
        self._label.configure(text=text, fg=color)
        self.place(relx=0.5, rely=0.94, anchor="s")
        self.lift()
        self._after = self.after(duration, self._next)

    def stop(self) -> None:
        if self._after:
            try:
                self.after_cancel(self._after)
            except tk.TclError:
                pass
        self._after = None
        self._queue.clear()
        self.place_forget()


class ProgressBar(tk.Canvas):
    def __init__(self, master, *, width: int = 180, height: int = 4, bg: str | None = None) -> None:
        super().__init__(master, width=width, height=height, bg=bg or T.panel,
                         highlightthickness=0, bd=0)
        self._value = 0.0
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        theme.rounded_rect(self, 0, 0, w, h, r=h / 2, fill=T.line, outline="")
        if self._value > 0:
            theme.rounded_rect(self, 0, 0, max(h, w * self._value), h, r=h / 2,
                               fill=T.accent, outline="")


def auto_wrap(label: tk.Label, padding: int, *, min_w: int = 80) -> tk.Label:
    """让 Label 的 wraplength 永远跟着实际宽度走，避免固定值在不同 DPI 下溢出。

    tk 的 wraplength 是像素，直接写死会在高倍屏 / 窗口缩放时把文字裁到栏外。
    """
    def _sync(_event=None) -> None:
        try:
            width = label.winfo_width()
        except tk.TclError:
            return
        if width <= 2:
            return
        want = max(min_w, width - padding)
        if abs(int(label.cget("wraplength")) - want) > 2:
            label.configure(wraplength=want)

    # 先给一个保守的初始 wraplength：wraplength=0 表示不换行，Label 的请求宽度
    # 会变成整行文字的宽度，把滚动容器撑宽几百像素。
    label.configure(wraplength=max(min_w, 240))
    label.bind("<Configure>", _sync, add="+")
    return label


class Section(tk.Frame):
    def __init__(self, master, title: str, *, subtitle: str = "", bg: str | None = None,
                 action: tuple[str, Callable] | None = None) -> None:
        super().__init__(master, bg=bg or T.panel)
        row = tk.Frame(self, bg=bg or T.panel)
        row.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(row, text=title, bg=bg or T.panel, fg=T.text, font=theme.f(10, True)).pack(side="left")
        if action:
            text, cmd = action
            lbl = tk.Label(row, text=text, bg=bg or T.panel, fg=T.text_mute,
                           font=theme.f(8), cursor="hand2")
            lbl.pack(side="right")
            lbl.bind("<Button-1>", lambda e: _safe(cmd))
            lbl.bind("<Enter>", lambda e: lbl.configure(fg=T.accent))
            lbl.bind("<Leave>", lambda e: lbl.configure(fg=T.text_mute))
        if subtitle:
            lbl = tk.Label(self, text=subtitle, bg=bg or T.panel, fg=T.text_mute,
                           font=theme.f(8), anchor="w", justify="left")
            auto_wrap(lbl, 4)
            lbl.pack(fill="x", padx=16, pady=(0, 4))


def hr(master, *, bg: str | None = None, padx: int = 16, pady: int = 8) -> None:
    line = tk.Frame(master, bg=T.line_soft, height=1)
    line.pack(fill="x", padx=padx, pady=pady)


class LinkLabel(tk.Label):
    def __init__(self, master, text: str, command: Callable | None = None, *, bg: str | None = None,
                 color: str | None = None, size: int = 9) -> None:
        super().__init__(master, text=text, bg=bg or T.panel, fg=color or T.accent,
                         font=(config.ui_font_family(), size, "underline"), cursor="hand2")
        self.bind("<Button-1>", lambda e: _safe(command))
