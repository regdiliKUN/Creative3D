"""tools/contact_sheet.py —— 概念库目视验收：批量生成 + 拼成一张总览图。

用法：
    python tools/contact_sheet.py out.png [grid] [--quality draft]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from creative3d import concepts, config, quality, render, sdfmesh  # noqa: E402


def build_one(cid: str, q, seed: int = 2026, **style):
    bp = concepts.make_params(seed, salt=cid, **style)
    fn = concepts.build(cid, bp)
    return sdfmesh.mesh_from_sdf(fn, q)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sheet.png")
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    q = quality.get("draft")
    if "--quality" in sys.argv:
        q = quality.get(sys.argv[sys.argv.index("--quality") + 1])
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))

    ids = [c.id for c in concepts.CONCEPTS if not only or c.id in only]
    cell = 300
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 22)), (12, 16, 23))
    draw = ImageDraw.Draw(sheet)

    bad = []
    for i, cid in enumerate(ids):
        t = time.perf_counter()
        try:
            mesh = build_one(cid, q)
            ok, b, nm = mesh.check_watertight()
            vol = mesh.signed_volume()
            img = render.render_scene(
                mesh, render.Camera(), cell, cell,
                config.PALETTES[concepts.get(cid).category == "生物" and "moss" or "frost"],
                supersample=1, shadow=True,
            ).image
            if not ok:
                bad.append((cid, b, nm))
            label = f"{cid} {mesh.n_faces // 1000}k {time.perf_counter()-t:.1f}s"
            if not ok:
                label += f" !{b}/{nm}"
            if vol <= 0:
                label += " INV"
                bad.append((cid, "inv", 0))
        except Exception as exc:  # noqa: BLE001
            bad.append((cid, type(exc).__name__, str(exc)[:40]))
            img = Image.new("RGB", (cell, cell), (60, 20, 20))
            label = f"{cid} FAIL"
        x = (i % cols) * cell
        y = (i // cols) * (cell + 22)
        sheet.paste(img, (x, y))
        draw.text((x + 6, y + cell + 4), label, fill=(200, 210, 225))

    sheet.save(out)
    print(f"已写入 {out}")
    if bad:
        print("问题清单：")
        for item in bad:
            print("  ", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
