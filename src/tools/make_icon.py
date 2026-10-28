"""tools/make_icon.py —— 用程序自己的渲染器生成应用图标（.ico / .png）。

好处是图标与实际作品同源：换个形态或配色，图标也跟着换，不会出现
「图标和软件风格不一致」的问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw  # noqa: E402

from creative3d import config, generators, render  # noqa: E402
from creative3d.generators import GenContext  # noqa: E402


def build_icon(size: int = 512) -> Image.Image:
    palette = config.PALETTES["frost"]
    mesh = generators.build("vase", GenContext(seed=11, res=70, a=52, b=34))
    mesh.normalize(config.MODEL_TARGET_MM)

    render_result = render.render_scene(
        mesh,
        render.Camera(yaw=-0.5, pitch=0.20, zoom=1.10),
        size,
        size,
        palette,
        supersample=2,
        transparent=True,
        shadow=False,
    )
    model = render_result.image.convert("RGBA")

    # 深色圆角底盘，让图标在浅色任务栏上也立得住
    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(plate)
    inset = int(size * 0.03)
    draw.rounded_rectangle(
        [inset, inset, size - inset, size - inset],
        radius=int(size * 0.22),
        fill=(13, 17, 24, 255),
        outline=(38, 48, 64, 255),
        width=max(2, size // 128),
    )
    # 顶部一抹冷色光晕
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse(
        [size * 0.1, -size * 0.35, size * 0.9, size * 0.55],
        fill=(77, 156, 255, 40),
    )
    glow = glow.filter(Image.ImageFilter.GaussianBlur(size / 12)) if hasattr(Image, "ImageFilter") else glow

    from PIL import ImageFilter

    glow = glow.filter(ImageFilter.GaussianBlur(size / 12))
    plate.alpha_composite(glow)

    # 模型缩放到 82% 居中
    inner = int(size * 0.76)
    model = model.resize((inner, inner), Image.LANCZOS)
    offset = ((size - inner) // 2, (size - inner) // 2)
    plate.alpha_composite(model, offset)
    return plate


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    master = build_icon(512)
    master.save(out_dir / "icon.png")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    master.save(out_dir / "icon.ico", sizes=sizes)
    print(f"图标已生成：{out_dir / 'icon.ico'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
