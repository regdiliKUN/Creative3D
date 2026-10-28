"""jobs.py —— 后台任务（线程池 + 主线程回调 + 可取消）。

约定：
* 所有耗时操作（高精度网格、导出、图片分析）都在这里跑，界面永不卡死；
* 结果通过 Tk 的 after 轮询回到主线程，避免跨线程操作控件；
* 每个任务带 token，界面切换参数时旧任务可以「过期作废」，不会回滚画面。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import AppError

log = logging.getLogger("creative3d.jobs")


class Cancelled(Exception):
    """任务被主动取消。"""


@dataclass
class Token:
    name: str = ""
    cancelled: bool = field(default=False)
    started: float = field(default_factory=time.perf_counter)

    def cancel(self) -> None:
        self.cancelled = True

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled()


class Jobs:
    """轻量任务调度器。root 必须是有 after 方法的 Tk 对象。"""

    def __init__(self, root, *, workers: int = 3, poll_ms: int = 24) -> None:
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="c3d")
        self.results: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.poll_ms = poll_ms
        self._running = 0
        self._closed = False
        self._pump()

    # ------------------------------------------------------------------ 内部
    def _pump(self) -> None:
        if self._closed:
            return
        try:
            while True:
                fn = self.results.get_nowait()
                try:
                    fn()
                except Exception:  # noqa: BLE001 - 回调异常不应打断主循环
                    log.exception("任务回调失败")
        except queue.Empty:
            pass
        try:
            self.root.after(self.poll_ms, self._pump)
        except Exception:  # noqa: BLE001 - 窗口销毁时会失败
            self._closed = True

    def _post(self, fn: Callable[[], None]) -> None:
        self.results.put(fn)

    def post(self, fn: Callable[[], None]) -> None:
        """供工作线程调用：把一段代码排到主线程执行（线程安全）。"""
        self.results.put(fn)

    # ------------------------------------------------------------------ 提交
    def submit(
        self,
        fn: Callable[..., Any],
        *args,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finally: Callable[[], None] | None = None,
        name: str = "task",
        **kwargs,
    ) -> Token:
        token = Token(name=name)
        self._running += 1

        def worker() -> None:
            try:
                token.raise_if_cancelled()
                result = fn(*args, token=token, **kwargs)
                if token.cancelled:
                    return
                if on_done:
                    self._post(lambda: on_done(result))
            except Cancelled:
                log.debug("任务被取消：%s", name)
            except AppError as exc:
                log.warning("任务 %s 失败：%s", name, exc.detail)
                if on_error:
                    self._post(lambda exc=exc: on_error(exc))
            except Exception as exc:  # noqa: BLE001
                log.error("任务 %s 异常：%s\n%s", name, exc, traceback.format_exc())
                if on_error:
                    self._post(lambda exc=exc: on_error(exc))
            finally:
                self._running -= 1
                if on_finally:
                    self._post(on_finally)

        self.pool.submit(worker)
        return token

    @property
    def busy(self) -> bool:
        return self._running > 0

    def shutdown(self) -> None:
        self._closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)
