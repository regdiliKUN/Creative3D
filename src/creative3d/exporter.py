"""exporter.py —— 导出 STL / 3MF / OBJ / PNG / 创意卡 / 打印体检报告。

要点：
* 所有写入都是原子的（临时文件 + 替换），中途失败不会留下半截文件；
* 文件重名自动加序号，不会覆盖用户已有作品；
* 每个步骤都会回调进度，界面可以显示「正在渲染预览图 2/5」；
* 磁盘满 / 没权限等异常统一转成 ExportError，附中文建议。
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, ideas, printcheck, registry
from .errors import ExportError, humanize_oserror
from .mesh import Mesh, atomic_write_bytes, atomic_write_text, safe_filename
from .render import Camera, render_scene

ProgressFn = Callable[[str, float], None]

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" \
Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


@dataclass
class ExportOptions:
    stl: bool = True
    obj: bool = False
    three_mf: bool = False
    png: bool = True
    card: bool = True
    report: bool = False
    png_width: int = 1400
    png_height: int = 1050
    supersample: int = 2
    transparent: bool = False
    wire: bool = False


@dataclass
class ExportResult:
    files: list[Path] = field(default_factory=list)
    folder: Path | None = None
    seconds: float = 0.0
    basename: str = ""

    def summary(self) -> str:
        kinds = "、".join(p.suffix.lstrip(".").upper() for p in self.files)
        return f"已导出 {kinds}（{self.seconds:.1f}s）"


def _unique_base(folder: Path, base: str, suffixes: list[str]) -> str:
    candidate = base
    index = 2
    while any((folder / f"{candidate}{s}").exists() for s in suffixes):
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def three_mf_bytes(mesh: Mesh, name: str = "creative3d") -> bytes:
    """写一个最小但合规的 3MF（单对象、毫米单位）。

    3MF 比 STL 多存了「单位」和「对象名」，切片软件导入时不用再猜尺寸。
    """
    verts = mesh.verts
    faces = mesh.faces + 1  # 3MF 索引从 1 开始
    v_parts = [f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in verts.tolist()]
    f_parts = [f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces.tolist()]
    safe_name = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="zh-CN" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        f' <metadata name="Title">{safe_name}</metadata>\n'
        f' <metadata name="Designer">Creative3D</metadata>\n'
        ' <resources>\n'
        f'  <object id="1" type="model" name="{safe_name}">\n'
        '   <mesh>\n'
        f'    <vertices>{"".join(v_parts)}</vertices>\n'
        f'    <triangles>{"".join(f_parts)}</triangles>\n'
        '   </mesh>\n'
        '  </object>\n'
        ' </resources>\n'
        ' <build><item objectid="1"/></build>\n'
        '</model>\n'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


def export_bundle(
    mesh: Mesh,
    *,
    form_id: str,
    seed: int,
    res: float,
    params: dict,
    palette_id: str,
    palette,
    idea: ideas.Idea,
    image_info: dict | None,
    out_dir: Path,
    options: ExportOptions,
    camera: Camera | None = None,
    concept_id: str | None = None,
    size_mm: float = 0.0,
    quality_label: str = "",
    print_report: printcheck.PrintReport | None = None,
    on_progress: ProgressFn | None = None,
    token=None,
) -> ExportResult:
    started = time.perf_counter()

    def tick(stage: str, frac: float) -> None:
        if token is not None:
            token.raise_if_cancelled()
        if on_progress:
            on_progress(stage, frac)

    if mesh is None or mesh.is_empty():
        raise ExportError("没有可导出的模型。", hint="先选一个形态并等预览出现。")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise ExportError(
            humanize_oserror(exc, action="准备导出目录"),
            hint="在「设置 → 导出目录」里换一个可写的位置（例如桌面）。",
            detail=str(exc),
        ) from exc

    suffixes = []
    if options.stl:
        suffixes.append(".stl")
    if options.three_mf:
        suffixes.append(".3mf")
    if options.obj:
        suffixes.append(".obj")
    if options.png:
        suffixes.append(".png")
    if options.card:
        suffixes.append(".md")
    if options.report:
        suffixes.append(".txt")

    label = registry.get(form_id).label
    base = safe_filename(f"{idea.title}_{label}_{seed}", fallback="creative3d")
    base = _unique_base(out_dir, base, suffixes)

    steps = max(1, len(suffixes))
    done = 0
    result = ExportResult(folder=out_dir, basename=base)

    def step(stage: str) -> None:
        nonlocal done
        tick(stage, done / steps)
        done += 1

    report = print_report
    if report is None and (options.report or options.card):
        tick("检查可打印性", 0.02)
        report = printcheck.analyze(mesh)

    try:
        if options.stl:
            step("导出 STL 模型")
            atomic_write_bytes(out_dir / f"{base}.stl", mesh.stl_bytes(base))

        if options.three_mf:
            step("导出 3MF 模型")
            atomic_write_bytes(out_dir / f"{base}.3mf", three_mf_bytes(mesh, base))

        if options.obj:
            step("导出 OBJ 模型")
            atomic_write_text(
                out_dir / f"{base}.obj", mesh.obj_text(base, palette=palette_id), encoding="utf-8"
            )

        if options.png:
            step("渲染预览图")
            width = max(320, min(4096, int(options.png_width)))
            height = max(240, min(4096, int(options.png_height)))
            result_img = render_scene(
                mesh,
                camera or Camera(),
                width,
                height,
                palette,
                supersample=max(1, min(3, int(options.supersample))),
                transparent=options.transparent,
                wire=options.wire,
                shadow=not options.transparent,
            )
            png_path = out_dir / f"{base}.png"
            tmp = png_path.with_suffix(".png.tmp")
            result_img.image.save(tmp, format="PNG", optimize=True)
            tmp.replace(png_path)

        if options.card:
            step("生成创意卡")
            text = ideas.idea_card_text(
                idea,
                form_id=form_id,
                seed=seed,
                res=res,
                params=params,
                palette_id=palette_id,
                image_info=image_info,
                concept_id=concept_id,
                size_mm=size_mm,
                quality_label=quality_label,
                report=report,
            )
            atomic_write_text(out_dir / f"{base}.md", text, encoding="utf-8")

        if options.report and report is not None:
            step("写打印体检报告")
            atomic_write_text(
                out_dir / f"{base}.txt", printcheck.report_text(report, title=idea.title),
                encoding="utf-8",
            )

        tick("完成", 1.0)
    except OSError as exc:
        raise ExportError(
            humanize_oserror(exc, action="写入导出文件"),
            hint="检查磁盘空间，或在「设置」里换一个导出目录。",
            detail=str(exc),
        ) from exc

    for suffix in suffixes:
        path = out_dir / f"{base}{suffix}"
        if path.exists():
            result.files.append(path)
    result.seconds = time.perf_counter() - started
    return result


def open_folder(path: Path) -> None:
    """在资源管理器中定位文件或文件夹（跨平台兜底）。"""
    import os
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            if path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:  # noqa: BLE001 - 打开失败不值得中断流程
        pass
