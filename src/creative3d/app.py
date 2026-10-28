"""app.py —— 程序入口：DPI、单实例、日志、命令行自检、启动主窗口。

命令行：
    （无参数）        启动图形界面
    --selftest DIR   无界面自检：生成全部形态、校验封闭性、导出样例、写报告
    --make-shortcut  仅创建桌面快捷方式后退出
    --version        打印版本
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import traceback
from pathlib import Path

from . import config, logsetup


def enable_dpi() -> None:
    """声明 DPI 感知，避免高分屏上界面糊成一片。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _single_instance_lock():
    """创建命名互斥体；已有实例时把它的窗口拉到前台并返回 None。"""
    if sys.platform != "win32":
        return object()
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, "Local\\Creative3D_v2_singleton")
        if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            _focus_existing()
            return None
        return handle
    except Exception:  # noqa: BLE001
        return object()


def _focus_existing() -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, f"{config.APP_NAME} v{config.APP_VERSION}")
        if hwnd:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
    except Exception:  # noqa: BLE001
        pass


def _set_app_id() -> None:
    """让任务栏使用自己的图标分组，而不是 python.exe。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"{config.APP_SHORT}.{config.APP_VERSION}"
        )
    except Exception:  # noqa: BLE001
        pass


def run_selftest(out_dir: Path) -> int:
    """无界面自检：几何 + 导出 + 渲染全链路，结果写 selftest-report.txt。"""
    from . import exporter, generators, ideas, registry, render
    from .generators import GenContext
    from .imaging import analyze
    from PIL import Image

    log = logsetup.get("selftest")
    out_dir = out_dir if out_dir.is_absolute() else Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Creative3D 自检报告",
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"版本: {config.APP_VERSION}",
        f"解释器: {sys.version.split()[0]}  frozen={config.is_frozen()}",
        f"程序目录: {config.app_dir()}",
        "",
    ]
    failures = 0

    # 造一张测试图，顺便验证多模态链路
    test_img = Image.new("RGB", (320, 320), (18, 22, 30))
    for y in range(320):
        for x in range(320):
            if (x - 160) ** 2 + (y - 160) ** 2 < 110 ** 2:
                test_img.putpixel((x, y), (90 + (x % 60), 150 + (y % 40), 210))
    feats = analyze(test_img, name="selftest.png")
    lines.append(f"图像分析: {feats.summary()} / 主色 {feats.colors[:3]}")

    for spec in registry.FORMS:
        started = time.perf_counter()
        try:
            mesh = generators.build(spec.id, GenContext(seed=1234, res=70, a=55, b=50, image=feats))
            mesh.normalize(config.MODEL_TARGET_MM)
            ok, boundary, nonmanifold = mesh.check_watertight()
            problems = mesh.validate()
            idea = ideas.generate_idea(spec.id, 1234, res=70, params={"a": 55, "b": 50})
            img = render.render_scene(
                mesh, render.Camera(), 900, 700, config.PALETTES[spec.default_palette]
            )
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            img.image.save(sample_dir / f"{spec.id}.png")
            mesh_path = sample_dir / f"{spec.id}.stl"
            mesh_path.write_bytes(mesh.stl_bytes(spec.id))
            elapsed = (time.perf_counter() - started) * 1000
            healthy = ok and not problems
            failures += 0 if healthy else 1
            lines.append(
                f"[{'OK' if healthy else 'FAIL'}] {spec.id:8s} 面 {mesh.n_faces:6d} "
                f"封闭 {ok} 边界 {boundary} 非流形 {nonmanifold} "
                f"体积 {mesh.signed_volume():.3f} 耗时 {elapsed:6.0f}ms 文案《{idea.title}》"
            )
            if problems:
                lines.append("        问题：" + "；".join(problems))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            lines.append(f"[FAIL] {spec.id:8s} 异常：{type(exc).__name__}: {exc}")
            lines.append(traceback.format_exc())

    # 导出全链路
    try:
        mesh = generators.build("vase", GenContext(seed=7, res=60, a=45, b=45))
        mesh.normalize(config.MODEL_TARGET_MM)
        idea = ideas.generate_idea("vase", 7, res=60, params={"a": 45, "b": 45})
        result = exporter.export_bundle(
            mesh,
            form_id="vase",
            seed=7,
            res=60,
            params={"a": 45, "b": 45},
            palette_id="frost",
            palette=config.PALETTES["frost"],
            idea=idea,
            image_info=None,
            out_dir=out_dir / "bundle",
            options=exporter.ExportOptions(stl=True, obj=True, png=True, card=True),
        )
        lines.append(f"[OK] 导出套装：{', '.join(p.name for p in result.files)}（{result.seconds:.1f}s）")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        lines.append(f"[FAIL] 导出套装：{type(exc).__name__}: {exc}")

    lines += ["", f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}"]
    report = out_dir / "selftest-report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已写入：{report}")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=config.APP_SHORT, description=config.APP_NAME)
    parser.add_argument("--selftest", metavar="DIR", nargs="?", const=".selftest",
                        help="无界面自检并输出报告")
    parser.add_argument("--make-shortcut", action="store_true", help="仅创建桌面快捷方式")
    parser.add_argument("--version", action="store_true", help="打印版本")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    if args.version:
        print(f"{config.APP_NAME} v{config.APP_VERSION}")
        return 0

    enable_dpi()
    logsetup.setup(verbose=args.verbose)

    if args.make_shortcut:
        from . import shortcut

        try:
            path = shortcut.create_desktop_shortcut(force=True)
            print(f"已创建：{path}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"创建失败：{exc}")
            return 1

    if args.selftest is not None:
        return run_selftest(Path(args.selftest))

    lock = _single_instance_lock()
    if lock is None:
        return 0
    _set_app_id()

    # 延迟导入，保证 --selftest 在无显示环境下也能跑
    from .settings import Settings
    from .ui.mainwindow import MainWindow

    log = logsetup.get("startup")
    log.info("启动 %s v%s (frozen=%s)", config.APP_NAME, config.APP_VERSION, config.is_frozen())
    settings = Settings()

    try:
        window = MainWindow(settings)
    except Exception:
        log.exception("主窗口创建失败")
        try:
            import tkinter.messagebox as mb

            mb.showerror("启动失败", "界面初始化失败，详情见日志：\n" + str(logsetup.log_path()))
        except Exception:  # noqa: BLE001
            print(traceback.format_exc(), file=sys.stderr)
        return 1

    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
