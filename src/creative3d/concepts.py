"""concepts.py —— 概念模型库：把一句话变成一具 SDF 实体。

和 v2 的「8 个固定形态」相比，这里是一套可组合的形体语汇：
每个概念都是若干 SDF 图元的装配（身体 + 头 + 四肢 + 尾巴 + 附件），
再叠上全局的风格修饰（圆润 / 尖锐 / 粗糙 / 中空 / 扭曲）。

为什么能「任意提示词」：
    * 命中了概念 → 走对应配方，参数与细节由提示词里的风格词驱动；
    * 没命中 → 走「抽象构成」兜底，用提示词哈希决定一组图元的堆叠方式，
      所以任何输入都会得到一个确定、可复现、可打印的形体，而不是报错。
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import sdf as S

TAU = 2.0 * math.pi


# --------------------------------------------------------------------------
# 建模参数
# --------------------------------------------------------------------------


@dataclass
class BuildParams:
    """一次生成所需的全部输入（全部与「毫米」解耦，统一用局部单位）。"""

    seed: int = 0
    a: float = 0.5           # 参数 A（含义随概念变化）
    b: float = 0.5           # 参数 B
    detail: float = 0.5      # 细节密度 0..1
    organic: float = 0.5     # 圆润程度 0..1
    sharp: float = 0.0       # 刻面 / 棱角 0..1
    rough: float = 0.0       # 表面粗糙 0..1
    twist: float = 0.0       # 扭曲 0..1
    hollow: bool = False     # 中空
    wall: float = 0.022      # 壁厚（局部单位）
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # 常用换算 -------------------------------------------------------------
    def k(self, base: float = 0.05) -> float:
        """平滑并集系数：越圆润，过渡越大。"""
        return base * (0.45 + self.organic * 1.55)

    def jitter(self, amp: float) -> float:
        return float(self.rng.uniform(-amp, amp))

    def pick(self, options: Iterable):
        opts = list(options)
        return opts[int(self.rng.integers(0, len(opts)))]


def make_params(seed: int, *, a: float = 0.5, b: float = 0.5, detail: float = 0.5,
                organic: float = 0.5, sharp: float = 0.0, rough: float = 0.0,
                twist: float = 0.0, hollow: bool = False, wall: float = 0.022,
                salt: str = "") -> BuildParams:
    mix = (int(seed) * 2654435761 + zlib.crc32(salt.encode("utf-8"))) & 0xFFFFFFFF
    return BuildParams(
        seed=int(seed), a=float(a), b=float(b), detail=float(detail),
        organic=float(organic), sharp=float(sharp), rough=float(rough),
        twist=float(twist), hollow=bool(hollow), wall=float(max(wall, 0.004)),
        rng=np.random.default_rng(mix),
    )


# --------------------------------------------------------------------------
# 通用构件
# --------------------------------------------------------------------------


def _limb(a, b, r0: float, r1: float | None = None):
    return S.tapered_path([a, b], r0, r0 if r1 is None else r1)


def _legs(positions, z_top: float, r: float, spread: float = 0.0):
    parts = []
    for x, y in positions:
        parts.append(_limb((x * (1 + spread), y * (1 + spread), 0.0), (x, y, z_top), r, r * 0.86))
    return parts


def _eyes(x: float, y: float, z: float, r: float = 0.035):
    return [S.at(S.sphere(r), -x, y, z), S.at(S.sphere(r), x, y, z)]


def _ear_cone(x: float, z: float, r: float, h: float, tilt: float = 0.22):
    c = S.cone(r, 0.014, h)
    return S.at(S.rotx(c, -tilt), x, 0.0, z + h * 0.5)


def _hemisphere(r: float, z_cut: float = 0.0):
    """半球顶（球与半空间求交）。"""
    return S.inter(S.sphere(r), S.plane((0, 0, -1), -z_cut))


def _disc(r: float, h: float):
    return S.cyl(r, h)


# --------------------------------------------------------------------------
# 器物
# --------------------------------------------------------------------------


def _vase(bp: BuildParams):
    waist = 0.24 + bp.a * 0.22
    flare = 0.20 + bp.b * 0.16
    h = 1.0
    # 平滑曲线采样：折线剖面会在侧壁上留下横向色带
    prof = []
    n = 20
    for i in range(n + 1):
        t = i / n
        bulge = math.exp(-(((t - 0.30) / 0.34) ** 2))
        r = waist * (0.72 + 0.62 * bulge + 0.10 * t)
        lip = max(0.0, (t - 0.88) / 0.12)
        r += flare * 0.55 * lip * lip
        prof.append((t * h, max(r, 0.05)))
    body = S.hollow_lathe(prof, bp.wall, open_top=True) if bp.hollow else S.lathe(prof)
    if bp.twist > 0.05:
        body = S.twist(body, bp.twist * 2.2, height=h)
    lobes = int(5 + round(bp.b * 8))
    if bp.detail > 0.4:
        body = S.flute(body, lobes, amp=0.018 + bp.b * 0.05,
                       fade_from=h * 0.80, fade_to=h * 0.95)
    return body


def _mug(bp: BuildParams):
    h = 0.72 + bp.a * 0.42
    r = 0.30 + bp.b * 0.10
    prof = [(0.0, r * 0.94), (h * 0.05, r), (h * 0.97, r * 1.03), (h, r * 1.03)]
    body = S.hollow_lathe(prof, bp.wall, open_top=True) if bp.hollow else S.lathe(prof)
    handle = S.at(S.rotx(S.torus(0.20 + h * 0.06, 0.048), math.pi * 0.5), r * 0.86, 0.0, h * 0.56)
    out = S.smooth_union(0.02, body, handle)
    out = S.sub(out, S.at(S.cyl(r * 0.30, 0.1), 0, 0, 0.02))
    return out


def _bowl(bp: BuildParams):
    r = 0.40 + bp.a * 0.14
    depth = 0.40 + bp.b * 0.30
    prof = [(0.0, r * 0.34), (0.02, r * 0.60), (0.08, r * 0.86), (0.22, r * 1.0), (depth, r * 0.98)]
    return S.hollow_lathe(prof, bp.wall, open_top=True) if bp.hollow else S.lathe(prof)


def _bottle(bp: BuildParams):
    h = 1.0
    body_r = 0.24 + bp.a * 0.10
    neck_r = 0.085 + bp.b * 0.05
    prof = [
        (0.0, body_r * 0.90),
        (h * 0.05, body_r),
        (h * 0.34, body_r * 1.06),
        (h * 0.50, body_r * 0.96),
        (h * 0.64, body_r * 0.62),
        (h * 0.74, neck_r * 1.1),
        (h * 0.94, neck_r),
        (h, neck_r * 1.16),
    ]
    body = S.hollow_lathe(prof, bp.wall, open_top=True) if bp.hollow else S.lathe(prof)
    ring = S.at(S.torus(neck_r * 1.10, 0.024), 0, 0, h * 0.90)
    return S.smooth_union(0.012, body, ring)


def _planter(bp: BuildParams):
    top = 0.40 + bp.a * 0.12
    bottom = top * (0.66 - bp.b * 0.16)
    h = 0.62 + bp.a * 0.22
    prof = [(0.0, bottom), (h * 0.08, bottom * 1.02), (h * 0.9, top), (h, top * 1.04)]
    body = S.hollow_lathe(prof, bp.wall * 1.3, open_top=True)
    rim = S.at(S.torus(top * 0.98, 0.026), 0, 0, h * 0.96)
    saucer = S.at(S.cyl(top * 0.98, 0.05), 0, 0, -0.03)
    return S.smooth_union(0.02, body, rim, saucer)


def _teapot(bp: BuildParams):
    r = 0.34 + bp.a * 0.08
    body = S.at(S.ellipsoid(r, r, r * 0.82), 0, 0, r * 0.92)
    if bp.hollow:
        inner = S.at(S.ellipsoid(r - bp.wall, r - bp.wall, r * 0.82 - bp.wall), 0, 0, r * 0.92)
        body = S.sub(body, inner)
    spout = S.tapered_path(
        [(r * 0.7, 0, r * 0.9), (r * 1.25, 0, r * 1.15), (r * 1.5, 0, r * 1.5)],
        r * 0.20, r * 0.10,
    )
    handle = S.at(S.rotx(S.torus(r * 0.72, r * 0.10), math.pi * 0.5), -r * 1.02, 0, r * 1.05)
    lid = S.at(_hemisphere(r * 0.62), 0, 0, r * 1.62)
    knob = S.at(S.sphere(r * 0.13), 0, 0, r * 1.78)
    pot = S.smooth_union(bp.k(0.05), body, spout, handle)
    pot = S.sub(pot, S.at(S.cyl(r * 0.5, 0.06), 0, 0, r * 1.70))
    return S.smooth_union(0.02, pot, lid, knob)


def _candle(bp: BuildParams):
    h = 0.30 + bp.a * 0.34
    base = S.lathe([(0.0, 0.30), (0.03, 0.34), (0.06, 0.20), (0.09, 0.14)])
    stem = S.at(S.lathe([(0.0, 0.075), (h * 0.5, 0.055), (h, 0.075)]), 0, 0, 0.05)
    cup = S.at(S.lathe([(0.0, 0.09), (0.05, 0.17), (0.08, 0.19)]), 0, 0, h + 0.03)
    wax = S.at(S.cyl(0.135, 0.26), 0, 0, h + 0.20)
    flame = S.at(S.ellipsoid(0.035, 0.035, 0.075), 0, 0, h + 0.38)
    out = S.smooth_union(0.03, base, stem, cup, wax)
    if bp.b > 0.5:
        out = S.smooth_union(0.02, out, flame)
    return out


def _lampshade(bp: BuildParams):
    top = 0.20 + bp.b * 0.10
    bottom = 0.40 + bp.a * 0.14
    h = 0.62 + bp.a * 0.16
    w = max(bp.wall * 1.4, 0.028)
    outer = S.lathe([(0.0, bottom), (h, top)])
    # 内芯用「斜壁 + 垂直上延」：口沿处壁厚恒为 w，不会收成刀刃
    inner = S.lathe([(-0.08, bottom - w), (h, top - w), (h + 0.10, top - w)])
    shade = S.sub(outer, inner)
    ring = S.at(S.torus(bottom * 0.99, 0.022), 0, 0, 0.012)
    return S.smooth_union(0.012, shade, ring)


def _penholder(bp: BuildParams):
    sides = int(4 + round(bp.b * 5))
    r = 0.28 + bp.a * 0.08
    h = 0.78
    w = max(bp.wall * 1.5, 0.026)
    outer = S.at(S.hexprism(r, h, sides=sides, round_r=0.010), 0, 0, h * 0.5)
    inner = S.at(S.hexprism(max(r - w, 0.05), h, sides=sides, round_r=0.008), 0, 0, h * 0.5 + w)
    body = S.sub(outer, inner)
    base = S.at(S.hexprism(r * 1.06, 0.05, sides=sides), 0, 0, 0.025)
    return S.smooth_union(0.012, body, base)


def _tray(bp: BuildParams):
    w = 0.46 + bp.a * 0.14
    d = 0.30 + bp.b * 0.10
    h = 0.10
    outer = S.rbox(w, d, h, 0.05)
    inner = S.rbox(w - 0.05, d - 0.05, h * 1.2, 0.045)
    body = S.at(S.sub(outer, S.at(inner, 0, 0, h + 0.055)), 0, 0, h)
    handles = [
        S.at(S.rotx(S.torus(0.10, 0.026), math.pi * 0.5), sx * (w + 0.03), 0, h * 1.15)
        for sx in (-1, 1)
    ]
    return S.smooth_union(0.02, body, *handles)


def _plate(bp: BuildParams):
    r = 0.44 + bp.a * 0.12
    prof = [(0.0, r * 0.55), (0.015, r * 0.80), (0.05, r * 1.0), (0.09, r * 0.98)]
    body = S.lathe(prof)
    rim = S.at(S.torus(r * 0.94, 0.018 + bp.b * 0.02), 0, 0, 0.085)
    return S.smooth_union(0.02, body, rim)


# --------------------------------------------------------------------------
# 家具
# --------------------------------------------------------------------------


def _chair(bp: BuildParams):
    seat_h = 0.46 + bp.a * 0.14
    w = 0.34 + bp.b * 0.10
    seat = S.at(S.rbox(w, w * 0.92, 0.035, 0.025), 0, 0, seat_h)
    back = S.at(S.rbox(w, 0.035, 0.30 + bp.b * 0.12, 0.03), 0, w * 0.9, seat_h + 0.30)
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            legs.append(_limb((sx * w * 0.82, sy * w * 0.80, 0.0),
                              (sx * w * 0.86, sy * w * 0.86, seat_h), 0.028, 0.024))
    for sx in (-1, 1):
        legs.append(_limb((sx * w * 0.86, w * 0.86, seat_h),
                          (sx * w * 0.86, w * 0.90, seat_h + 0.44), 0.024, 0.022))
    return S.smooth_union(bp.k(0.03), seat, back, *legs)


def _table(bp: BuildParams):
    w = 0.50 + bp.a * 0.16
    d = 0.34 + bp.b * 0.12
    h = 0.66
    top = S.at(S.rbox(w, d, 0.032, 0.022), 0, 0, h)
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            legs.append(_limb((sx * (w - 0.07), sy * (d - 0.06), 0.0),
                              (sx * (w - 0.09), sy * (d - 0.08), h), 0.032, 0.028))
    apron = S.at(S.rbox(w * 0.9, d * 0.9, 0.035, 0.01), 0, 0, h - 0.06)
    apron = S.sub(apron, S.at(S.rbox(w * 0.86, d * 0.86, 0.06, 0.01), 0, 0, h - 0.06))
    return S.smooth_union(bp.k(0.02), top, apron, *legs)


def _stool(bp: BuildParams):
    h = 0.44 + bp.a * 0.12
    r = 0.26 + bp.b * 0.08
    seat = S.at(S.lathe([(0.0, r * 0.92), (0.03, r), (0.06, r * 0.96)]), 0, 0, h - 0.03)
    legs = []
    for i in range(3):
        ang = TAU * i / 3
        legs.append(_limb((math.cos(ang) * r * 0.80, math.sin(ang) * r * 0.80, 0.0),
                          (math.cos(ang) * r * 0.55, math.sin(ang) * r * 0.55, h - 0.03),
                          0.030, 0.024))
    ring = S.at(S.torus(r * 0.66, 0.014), 0, 0, h * 0.34)
    return S.smooth_union(bp.k(0.02), seat, ring, *legs)


def _shelf(bp: BuildParams):
    w = 0.42 + bp.a * 0.10
    h = 0.90
    d = 0.22 + bp.b * 0.10
    parts = [
        S.at(S.rbox(0.022, d, h * 0.5, 0.008), -w, 0, h * 0.5),
        S.at(S.rbox(0.022, d, h * 0.5, 0.008), w, 0, h * 0.5),
        S.at(S.rbox(w, 0.014, h * 0.5, 0.006), 0, d - 0.014, h * 0.5),
    ]
    levels = 4
    for i in range(levels):
        z = 0.04 + (h - 0.06) * i / (levels - 1)
        parts.append(S.at(S.rbox(w, d, 0.014, 0.006), 0, 0, z))
    return S.smooth_union(0.008, *parts)


def _lamp(bp: BuildParams):
    base_r = 0.22 + bp.a * 0.08
    h = 0.52 + bp.b * 0.16
    base = S.at(S.lathe([(0.0, base_r), (0.03, base_r * 1.02), (0.06, base_r * 0.86)]), 0, 0, 0.0)
    stem = S.at(S.cyl(0.022, h), 0, 0, h * 0.5 + 0.05)
    shade_top = 0.16
    shade_bot = 0.30
    shade_h = 0.34
    w = max(bp.wall * 1.5, 0.028)
    outer = S.at(S.lathe([(0.0, shade_bot), (shade_h, shade_top)]), 0, 0, h + 0.02)
    inner = S.at(S.lathe([(-0.08, shade_bot - w), (shade_h, shade_top - w),
                          (shade_h + 0.10, shade_top - w)]), 0, 0, h + 0.02)
    shade = S.sub(outer, inner)
    bulb = S.at(S.sphere(0.075), 0, 0, h + 0.10)
    return S.smooth_union(0.015, base, stem, shade, bulb)


# --------------------------------------------------------------------------
# 建筑
# --------------------------------------------------------------------------


def _house(bp: BuildParams):
    w = 0.40 + bp.a * 0.10
    d = 0.34 + bp.b * 0.10
    h = 0.42
    body = S.at(S.box(w, d, h * 0.5), 0, 0, h * 0.5)
    roof_h = 0.26 + bp.b * 0.12
    tri = S.poly2([(-w * 1.12, 0.0), (w * 1.12, 0.0), (0.0, roof_h)])
    roof = S.at(S.rotx(S.extrude(tri, d * 1.12), math.pi * 0.5), 0, 0, h)
    chimney = S.at(S.box(0.05, 0.05, 0.12), w * 0.55, d * 0.45, h + roof_h * 0.55)
    out = S.union(body, roof, chimney)
    door = S.at(S.box(0.085, 0.05, 0.15), 0, -d - 0.01, 0.15)
    win = []
    for sx in (-1, 1):
        win.append(S.at(S.box(0.07, 0.05, 0.07), sx * w * 0.58, -d - 0.01, h * 0.62))
    side = S.at(S.box(0.05, 0.07, 0.07), w + 0.01, 0, h * 0.62)
    out = S.sub(out, door, side, *win)
    return out


def _tower(bp: BuildParams):
    h = 0.95 + bp.a * 0.35
    r = 0.26 + bp.b * 0.10
    prof = [(0.0, r * 1.15), (h * 0.08, r * 1.05), (h * 0.55, r * 0.82), (h * 0.92, r * 0.72)]
    body = S.lathe(prof)
    if bp.hollow:
        body = S.sub(body, S.at(S.cyl(r * 0.5, h * 1.2), 0, 0, h * 0.6))
    merlons = S.ring_of(S.box(0.055, 0.040, 0.048), 8, r * 0.82)
    merlons = S.at(merlons, 0, 0, h * 0.95)
    roof = S.at(S.cone(r * 0.86, 0.004, r * 1.25), 0, 0, h + r * 0.62)
    band = S.at(S.torus(r * 0.78, 0.020), 0, 0, h * 0.55)
    return S.smooth_union(0.015, body, merlons, roof, band)


def _castle(bp: BuildParams):
    w = 0.30 + bp.a * 0.08
    h = 0.56 + bp.b * 0.18
    keep = S.at(S.box(w, w * 0.8, h * 0.5), 0, 0, h * 0.5)
    parts = [keep]
    for sx in (-1, 1):
        for sy in (-1, 1):
            tr = w * 0.30
            parts.append(S.at(S.cyl(tr, h * 1.25), sx * w, sy * w * 0.8, h * 0.62))
            parts.append(S.at(S.cone(tr * 1.15, 0.004, tr * 1.3), sx * w, sy * w * 0.8, h * 1.24 + tr * 0.62))
    out = S.smooth_union(bp.k(0.02), *parts)
    gate = S.union(S.at(S.box(0.10, 0.12, 0.14), 0, -w * 0.8 - 0.02, 0.14),
                   S.at(S.roty(S.cyl(0.10, 0.3), math.pi * 0.5), 0, -w * 0.8, 0.28))
    return S.sub(out, gate)


def _arch(bp: BuildParams):
    w = 0.40 + bp.a * 0.10
    h = 0.70 + bp.b * 0.30
    t = 0.10
    outer = S.poly2([(-w, 0.0), (w, 0.0), (w, h), (-w, h)])
    hole_w = w - t
    hole = S.union2(
        S.at2(S.box2(hole_w, (h - w) * 0.5 + 0.02), 0.0, (h - w) * 0.5),
        S.at2(S.circle2(hole_w), 0.0, h - w),
    )
    shape = S.sub2(outer, hole)
    # extrude 的 2D 平面是 xy、沿 z 挤出；竖起来需要绕 x 转 90°（y → z）
    return S.rotx(S.extrude(shape, 0.13, round_r=0.012), math.pi * 0.5)


def _bridge(bp: BuildParams):
    span = 0.60 + bp.a * 0.20
    h = 0.36 + bp.b * 0.14
    deck = S.at(S.rbox(span, 0.13, 0.035, 0.012), 0, 0, h)
    rail = []
    for sy in (-1, 1):
        rail.append(S.at(S.rbox(span, 0.014, 0.05, 0.008), 0, sy * 0.13, h + 0.06))
    piers = []
    for sx in (-1, 1):
        piers.append(S.at(S.rbox(0.07, 0.12, h * 0.5, 0.012), sx * span * 0.72, 0, h * 0.5))
    # 拱洞沿 y 方向贯穿：圆柱轴转到 y，圆心放在桥面下方，顶到桥底为止
    arch_r = span * 0.52
    arch = S.at(S.rotx(S.cyl(arch_r, 0.34), math.pi * 0.5), 0, 0, h - 0.05 - arch_r)
    out = S.union(deck, *rail, *piers)
    return S.sub(out, arch)


def _stairs(bp: BuildParams):
    n = int(6 + round(bp.a * 6))
    rise = 0.09
    run = 0.13 + bp.b * 0.05
    st = S.steps(n, rise, run, 0.34)
    st = S.at(st, 0, -n * run * 0.5, 0.0)
    walls = []
    for sx in (-1, 1):
        walls.append(S.at(S.rbox(0.022, n * run * 0.5, n * rise * 0.5, 0.008),
                          sx * 0.17, 0, n * rise * 0.5))
    return S.smooth_union(0.01, st, *walls)


def _cupola(bp: BuildParams):
    r = 0.44 + bp.a * 0.12
    base_h = 0.14 + bp.b * 0.12
    base = S.at(S.cyl(r * 1.02, base_h), 0, 0, base_h * 0.5)
    cap = S.at(_hemisphere(r), 0, 0, base_h)
    ring = S.at(S.torus(r * 1.0, 0.026), 0, 0, base_h)
    out = S.smooth_union(0.015, base, cap, ring)
    oculus = S.at(S.cyl(r * 0.16, r * 1.2), 0, 0, base_h + r * 0.9)
    return S.sub(out, oculus)


# --------------------------------------------------------------------------
# 生物
# --------------------------------------------------------------------------


def _cat(bp: BuildParams):
    k = bp.k(0.05)
    body = S.at(S.ellipsoid(0.26, 0.24, 0.30), 0, 0.03, 0.34)
    chest = S.at(S.ellipsoid(0.20, 0.18, 0.20), 0, -0.06, 0.58)
    head = S.at(S.sphere(0.185), 0, -0.10, 0.82)
    muzzle = S.at(S.ellipsoid(0.085, 0.075, 0.06), 0, -0.26, 0.78)
    ears = [_ear_cone(sx * 0.095, 0.88, 0.062, 0.13) for sx in (-1, 1)]
    front = [_limb((sx * 0.11, -0.14, 0.0), (sx * 0.11, -0.16, 0.44), 0.045, 0.042) for sx in (-1, 1)]
    haunch = [S.at(S.ellipsoid(0.10, 0.13, 0.14), sx * 0.20, 0.06, 0.16) for sx in (-1, 1)]
    tail_pts = [(0.0, 0.24, 0.30), (0.10, 0.34, 0.34), (0.22, 0.40, 0.44), (0.30, 0.40, 0.58)]
    tail = S.tapered_path(tail_pts, 0.035, 0.018)
    out = S.smooth_union(k, body, chest, head, muzzle, *ears, *front, *haunch, tail)
    if bp.detail > 0.4:
        out = S.smooth_union(0.02, out, *_eyes(0.075, -0.24, 0.86, 0.024))
    return out


def _dog(bp: BuildParams):
    k = bp.k(0.05)
    body = S.at(S.ellipsoid(0.21, 0.32, 0.20), 0, 0.02, 0.52)
    chest = S.at(S.ellipsoid(0.20, 0.16, 0.18), 0, -0.24, 0.50)
    head = S.at(S.sphere(0.155), 0, -0.40, 0.66)
    snout = S.at(S.ellipsoid(0.075, 0.11, 0.065), 0, -0.55, 0.61)
    ears = [S.at(S.ellipsoid(0.05, 0.035, 0.10), sx * 0.13, -0.38, 0.68) for sx in (-1, 1)]
    legs = [_limb((sx * 0.15, sy * 0.22, 0.0), (sx * 0.15, sy * 0.22, 0.40), 0.042, 0.036)
            for sx in (-1, 1) for sy in (-1, 1)]
    tail = S.tapered_path([(0, 0.32, 0.60), (0, 0.42, 0.74), (0, 0.44, 0.86)], 0.035, 0.02)
    out = S.smooth_union(k, body, chest, head, snout, *ears, *legs, tail)
    if bp.detail > 0.4:
        out = S.smooth_union(0.02, out, *_eyes(0.07, -0.50, 0.70, 0.022))
    return out


def _rabbit(bp: BuildParams):
    k = bp.k(0.05)
    ear_h = 0.30 + bp.a * 0.18
    body = S.at(S.ellipsoid(0.24, 0.28, 0.26), 0, 0.06, 0.36)
    head = S.at(S.sphere(0.17), 0, -0.16, 0.66)
    ears = []
    for sx in (-1, 1):
        c = S.cone(0.055, 0.02, ear_h)
        ears.append(S.at(S.rotx(c, -0.18), sx * 0.075, -0.14, 0.74 + ear_h * 0.5))
    hind = [S.at(S.ellipsoid(0.10, 0.14, 0.13), sx * 0.18, 0.14, 0.16) for sx in (-1, 1)]
    front = [_limb((sx * 0.09, -0.16, 0.0), (sx * 0.09, -0.18, 0.36), 0.038, 0.034) for sx in (-1, 1)]
    tail = S.at(S.sphere(0.055), 0, 0.30, 0.42)
    out = S.smooth_union(k, body, head, *ears, *hind, *front, tail)
    if bp.detail > 0.4:
        out = S.smooth_union(0.02, out, *_eyes(0.07, -0.28, 0.70, 0.022))
    return out


def _bear(bp: BuildParams):
    k = bp.k(0.07)
    body = S.at(S.ellipsoid(0.28, 0.24, 0.28), 0, 0.04, 0.40)
    head = S.at(S.sphere(0.20), 0, -0.16, 0.70)
    muzzle = S.at(S.ellipsoid(0.095, 0.09, 0.075), 0, -0.32, 0.65)
    ears = [S.at(S.sphere(0.062), sx * 0.14, -0.12, 0.85) for sx in (-1, 1)]
    arms = [_limb((sx * 0.24, -0.10, 0.16), (sx * 0.30, -0.16, 0.50), 0.075, 0.06) for sx in (-1, 1)]
    legs = [_limb((sx * 0.16, 0.06, 0.0), (sx * 0.18, 0.06, 0.24), 0.08, 0.07) for sx in (-1, 1)]
    out = S.smooth_union(k, body, head, muzzle, *ears, *arms, *legs)
    if bp.detail > 0.4:
        out = S.smooth_union(0.02, out, *_eyes(0.075, -0.28, 0.76, 0.022))
    return out


def _bird(bp: BuildParams):
    k = bp.k(0.05)
    body = S.at(S.ellipsoid(0.20, 0.26, 0.19), 0, 0.02, 0.44)
    head = S.at(S.sphere(0.13), 0, -0.22, 0.62)
    beak = S.at(S.roty(S.cone(0.045, 0.004, 0.12), -math.pi * 0.5), 0, -0.34, 0.60)
    tail = S.at(S.ellipsoid(0.10, 0.11, 0.014), 0, 0.28, 0.47)
    wings = [S.at(S.ellipsoid(0.055, 0.16, 0.11), sx * 0.19, 0.02, 0.48) for sx in (-1, 1)]
    legs = [_limb((sx * 0.07, 0.02, 0.0), (sx * 0.07, 0.02, 0.28), 0.020, 0.017) for sx in (-1, 1)]
    out = S.smooth_union(k, body, head, beak, tail, *wings, *legs)
    if bp.detail > 0.4:
        out = S.smooth_union(0.015, out, *_eyes(0.062, -0.30, 0.65, 0.018))
    return out


def _fish(bp: BuildParams):
    k = bp.k(0.04)
    body = S.at(S.ellipsoid(0.13, 0.34, 0.24), 0, 0, 0.34)
    # 尾鳍：上下两片扁椭球张成 V 形
    fins = []
    for s in (-1, 1):
        f = S.rotx(S.ellipsoid(0.012, 0.11, 0.075), s * 0.55)
        fins.append(S.at(f, 0, 0.40, 0.34 + s * 0.07))
    dorsal = S.at(S.ellipsoid(0.012, 0.10, 0.06), 0, -0.02, 0.56)
    eyes = _eyes(0.10, -0.22, 0.40, 0.028)
    out = S.smooth_union(k, body, dorsal, *fins, *eyes)
    return S.sub(out, S.at(S.cyl(0.030, 0.16), 0, -0.36, 0.33))


def _dinosaur(bp: BuildParams):
    k = bp.k(0.05)
    neck_len = 0.28 + bp.a * 0.20
    body = S.at(S.ellipsoid(0.20, 0.28, 0.22), 0, 0.06, 0.50)
    neck = S.tapered_path([(0, -0.16, 0.58), (0, -0.20, 0.74), (0, -0.24, 0.74 + neck_len)],
                          0.075, 0.05)
    head = S.at(S.ellipsoid(0.075, 0.11, 0.075), 0, -0.26, 0.76 + neck_len)
    tail = S.tapered_path([(0, 0.28, 0.52), (0, 0.46, 0.44), (0, 0.62, 0.30)], 0.09, 0.02)
    legs = []
    for sx in (-1, 1):
        legs.append(_limb((sx * 0.16, -0.02, 0.0), (sx * 0.14, -0.02, 0.34), 0.058, 0.05))
        legs.append(_limb((sx * 0.15, 0.20, 0.0), (sx * 0.14, 0.20, 0.30), 0.05, 0.045))
    out = S.smooth_union(k, body, neck, head, tail, *legs)
    if bp.detail > 0.4:
        out = S.smooth_union(0.015, out, *_eyes(0.055, -0.34, 0.78 + neck_len, 0.017))
    return out


def _octopus(bp: BuildParams):
    k = bp.k(0.05)
    head = S.at(S.ellipsoid(0.30, 0.30, 0.28), 0, 0, 0.62)
    eyes = _eyes(0.14, -0.24, 0.66, 0.045)
    arms = []
    n = int(6 + round(bp.a * 3))
    for i in range(n):
        ang = TAU * i / n + 0.2
        pts = []
        for j in range(4):
            t = j / 3.0
            rad = 0.22 + t * 0.42
            pts.append((math.cos(ang) * rad, math.sin(ang) * rad, 0.46 - t * 0.42 - 0.05 * math.sin(t * 3)))
        arms.append(S.tapered_path(pts, 0.072, 0.026))
    return S.smooth_union(k, head, *eyes, *arms)


def _turtle(bp: BuildParams):
    k = bp.k(0.04)
    shell = S.at(S.ellipsoid(0.33, 0.40, 0.26), 0, 0, 0.26)
    # 切平壳底，露出平底方便打印
    shell = S.sub(shell, S.at(S.box(0.9, 1.0, 0.30), 0, 0, -0.20))
    head = S.at(S.ellipsoid(0.10, 0.13, 0.09), 0, -0.44, 0.20)
    legs = [S.at(S.ellipsoid(0.075, 0.10, 0.055), sx * 0.27, sy * 0.22, 0.08)
            for sx in (-1, 1) for sy in (-1, 1)]
    tail = S.tapered_path([(0, 0.36, 0.14), (0, 0.46, 0.10)], 0.032, 0.010)
    ridges = []
    if bp.detail > 0.35:
        ridges = [S.at(S.torus(0.22 - i * 0.07, 0.012), 0, 0, 0.30 + i * 0.055) for i in range(2)]
    return S.smooth_union(k, shell, head, *legs, tail, *ridges)


def _duck(bp: BuildParams):
    k = bp.k(0.05)
    body = S.at(S.ellipsoid(0.20, 0.28, 0.18), 0, 0.06, 0.34)
    neck = S.tapered_path([(0, -0.16, 0.40), (0, -0.20, 0.56), (0, -0.18, 0.64)], 0.075, 0.06)
    head = S.at(S.sphere(0.115), 0, -0.18, 0.70)
    beak = S.at(S.roty(S.cone(0.05, 0.035, 0.11), -math.pi * 0.5), 0, -0.28, 0.68)
    tail = S.at(S.ellipsoid(0.08, 0.09, 0.012), 0, 0.30, 0.40)
    wings = [S.at(S.ellipsoid(0.05, 0.13, 0.09), sx * 0.18, 0.06, 0.36) for sx in (-1, 1)]
    out = S.smooth_union(k, body, neck, head, beak, tail, *wings)
    if bp.detail > 0.4:
        out = S.smooth_union(0.015, out, *_eyes(0.06, -0.26, 0.74, 0.018))
    return out


# --------------------------------------------------------------------------
# 机械
# --------------------------------------------------------------------------


def _robot(bp: BuildParams):
    k = bp.k(0.025)
    tw = 0.24 + bp.a * 0.06
    torso = S.at(S.rbox(tw, 0.15, 0.26, 0.035), 0, 0, 0.66)
    head = S.at(S.rbox(0.17, 0.14, 0.13, 0.03), 0, 0, 1.06)
    neck = S.at(S.cyl(0.055, 0.10), 0, 0, 0.94)
    antenna = S.at(S.cyl(0.012, 0.16), 0, 0, 1.24)
    tip = S.at(S.sphere(0.032), 0, 0, 1.34)
    arms = []
    for sx in (-1, 1):
        arms.append(S.at(S.sphere(0.075), sx * (tw + 0.04), 0, 0.86))
        arms.append(_limb((sx * (tw + 0.05), 0, 0.84), (sx * (tw + 0.07), 0.02, 0.46), 0.05, 0.045))
    legs = []
    for sx in (-1, 1):
        legs.append(_limb((sx * 0.12, 0, 0.40), (sx * 0.13, 0, 0.10), 0.06, 0.055))
        legs.append(S.at(S.rbox(0.085, 0.12, 0.035, 0.02), sx * 0.13, -0.02, 0.035))
    out = S.smooth_union(k, torso, head, neck, antenna, tip, *arms, *legs)
    eyes = [S.at(S.sphere(0.032), sx * 0.07, -0.135, 1.08) for sx in (-1, 1)]
    out = S.sub(out, *eyes)
    if bp.detail > 0.4:
        vents = [S.at(S.rbox(0.10, 0.02, 0.012, 0.004), 0, -0.15, 0.60 + i * 0.055) for i in range(3)]
        out = S.sub(out, *vents)
    return out


def _rocket(bp: BuildParams):
    h = 1.10 + bp.a * 0.30
    r = 0.17 + bp.b * 0.05
    prof = [(0.0, r * 1.10), (h * 0.06, r), (h * 0.66, r), (h * 0.78, r * 0.82), (h, 0.006)]
    body = S.lathe(prof)
    fins = []
    fin_shape = S.poly2([(0.0, 0.0), (r * 1.55, 0.0), (0.0, h * 0.30)])
    for i in range(3):
        a = TAU * i / 3
        fin = S.rotx(S.extrude(fin_shape, 0.018, round_r=0.006), math.pi * 0.5)
        fin = S.at(fin, 0, 0, h * 0.06)
        fins.append(S.rotz(fin, a))
    nozzle = S.at(S.cone(r * 0.92, r * 0.55, r * 0.5), 0, 0, -r * 0.2)
    window = S.at(S.torus(r * 0.42, 0.026), 0, -r * 0.90, h * 0.60)
    window = S.roty(window, math.pi * 0.5)
    window = S.rotx(window, math.pi * 0.5)
    out = S.smooth_union(bp.k(0.02), body, nozzle, *fins)
    out = S.sub(out, S.at(S.sphere(r * 0.34), 0, -r * 0.92, h * 0.60))
    return out


def _car(bp: BuildParams):
    L = 0.52 + bp.a * 0.10
    W = 0.28 + bp.b * 0.05
    body = S.at(S.rbox(W, L, 0.11, 0.05), 0, 0, 0.20)
    cabin = S.at(S.rbox(W * 0.86, L * 0.50, 0.10, 0.045), 0, 0.02, 0.36)
    wheels = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            w = S.roty(S.cyl(0.105, 0.07), math.pi * 0.5)
            wheels.append(S.at(w, sx * (W + 0.01), sy * L * 0.62, 0.105))
    out = S.smooth_union(bp.k(0.02), body, cabin, *wheels)
    glass = [
        S.at(S.rbox(W * 0.70, L * 0.22, 0.075, 0.03), 0, 0.02, 0.40),
    ]
    out = S.sub(out, *glass)
    return out


def _plane(bp: BuildParams):
    L = 0.52 + bp.a * 0.14
    fuse = S.at(S.ellipsoid(0.09, L, 0.09), 0, 0, 0.30)
    nose = S.at(S.roty(S.cone(0.09, 0.004, 0.16), -math.pi * 0.5), 0, -L - 0.06, 0.30)
    # 机翼：2D 平面是 xy、厚度沿 z —— 水平翼正好不用转
    wing_shape = S.poly2([(-0.46, 0.0), (0.46, 0.0), (0.40, 0.14), (-0.40, 0.14)])
    wing = S.at(S.extrude(wing_shape, 0.014, round_r=0.006), 0, -0.04, 0.28)
    tail_h = S.at(S.extrude(S.poly2([(-0.16, 0), (0.16, 0), (0.13, 0.09), (-0.13, 0.09)]), 0.010),
                  0, L * 0.80, 0.30)
    # 垂尾：2D 的 x 当作高度，roty 后变成竖直
    fin_shape = S.poly2([(0.0, 0.0), (0.20, 0.0), (0.02, 0.17)])
    fin = S.at(S.roty(S.extrude(fin_shape, 0.010), math.pi * 0.5), 0, L * 0.78, 0.34)
    out = S.smooth_union(bp.k(0.015), fuse, nose, wing, tail_h, fin)
    return out


def _gear(bp: BuildParams):
    teeth = int(8 + round(bp.a * 10))
    r = 0.36 + bp.b * 0.10
    body = S.cyl(r, 0.14)
    tooth = S.box(0.055, 0.075, 0.065)
    ring = S.ring_of(tooth, teeth, r + 0.03)
    out = S.union(body, ring)
    hole = S.cyl(r * 0.26, 0.4)
    out = S.sub(out, hole)
    if bp.detail > 0.35:
        spokes = S.ring_of(S.cyl(r * 0.10, 0.4), 4, r * 0.56)
        out = S.sub(out, *[S.at(spokes, 0, 0, 0)])
    return out


def _screw(bp: BuildParams):
    h = 0.70 + bp.a * 0.30
    r = 0.075 + bp.b * 0.02
    shaft = S.at(S.cyl(r, h), 0, 0, h * 0.5)
    tip = S.at(S.cone(r, 0.004, r * 1.8), 0, 0, -r * 0.9)
    head = S.at(S.cyl(r * 2.5, 0.075), 0, 0, h + 0.04)
    turns = 7
    pts = S.polar_path(lambda t: r * 1.05, lambda t: 0.05 + t * (h - 0.05), turns, turns * 12)
    thread = S.tapered_path(pts, r * 0.30, r * 0.30)
    out = S.smooth_union(0.006, shaft, tip, head, thread)
    slot = S.at(S.box(r * 2.0, 0.016, 0.05), 0, 0, h + 0.06)
    out = S.sub(out, slot)
    return out


def _sword(bp: BuildParams):
    blade_len = 0.90 + bp.a * 0.30
    blade_w = 0.055 + bp.b * 0.025
    # 剑身沿 z 竖起：扁盒 + 两个斜平面切出剑尖
    blade = S.at(S.box(blade_w, 0.016, blade_len * 0.5), 0, 0, blade_len * 0.5)
    off = blade_len / math.sqrt(2.0)
    blade = S.inter(blade, S.plane((1, 0, 1), off), S.plane((-1, 0, 1), off))
    guard = S.at(S.rbox(blade_w * 3.2, 0.030, 0.024, 0.010), 0, 0, 0.0)
    grip = S.at(S.cyl(0.030, 0.22), 0, 0, -0.14)
    pommel = S.at(S.sphere(0.042), 0, 0, -0.27)
    out = S.smooth_union(0.012, blade, guard, grip, pommel)
    return S.at(out, 0, 0, 0.30)


def _shield(bp: BuildParams):
    w = 0.34 + bp.a * 0.08
    h = 0.80 + bp.b * 0.20
    pts = [
        (-w, h * 0.5), (w, h * 0.5), (w * 1.02, h * 0.1),
        (w * 0.62, -h * 0.32), (0.0, -h * 0.5), (-w * 0.62, -h * 0.32), (-w * 1.02, h * 0.1),
    ]
    shape = S.poly2(pts)
    body = S.extrude(shape, 0.032, round_r=0.012)
    boss = S.at(S.ellipsoid(w * 0.42, w * 0.42, 0.075), 0, 0, 0.0)
    out = S.smooth_union(0.02, body, boss)
    border = S.sub(S.extrude(shape, 0.044, round_r=0.012), S.extrude(shape, 0.032, round_r=0.012))
    border = S.inter(border, S.plane((0, 0, -1), 0.0))
    out = S.smooth_union(0.01, out, border)
    # 立起来：2D 的 y（高度）转成 3D 的 z
    out = S.rotx(out, math.pi * 0.5)
    return S.at(out, 0, 0, h * 0.5)


# --------------------------------------------------------------------------
# 食物
# --------------------------------------------------------------------------


def _mushroom(bp: BuildParams):
    cap_r = 0.30 + bp.a * 0.12
    stem_h = 0.32 + bp.b * 0.18
    stem = S.at(S.lathe([(0.0, 0.11), (stem_h * 0.4, 0.085), (stem_h, 0.10)]), 0, 0, 0)
    cap = S.at(S.ellipsoid(cap_r, cap_r, cap_r * 0.74), 0, 0, stem_h + cap_r * 0.10)
    # 切掉伞盖下半：box 上表面落在 stem_h - 0.05*cap_r，留下一个圆顶
    cap = S.sub(cap, S.at(S.box(cap_r * 1.6, cap_r * 1.6, cap_r), 0, 0, stem_h - cap_r * 1.05))
    out = S.smooth_union(bp.k(0.04), stem, cap)
    if bp.detail > 0.4:
        gills = S.ring_of(S.box(0.014, cap_r * 0.55, 0.02), 11, cap_r * 0.45)
        out = S.sub(out, S.at(gills, 0, 0, stem_h - 0.01))
    return out


def _donut(bp: BuildParams):
    R = 0.34 + bp.a * 0.10
    r = 0.135 + bp.b * 0.04
    body = S.torus(R, r)
    glaze = S.offset(S.torus(R, r * 1.03), 0.012)
    glaze = S.inter(glaze, S.plane((0, 0, -1), -0.01))
    out = S.smooth_union(0.008, body, glaze)
    if bp.detail > 0.4:
        sprinkles = []
        for i in range(14):
            ang = TAU * i / 14 + 0.3
            rad = R + (0.05 if i % 2 else -0.05)
            s = S.capsule((-0.022, 0, 0), (0.022, 0, 0), 0.008)
            s = S.rotz(s, ang * 2.0)
            s = S.rotx(s, math.pi * 0.5)
            sprinkles.append(S.at(s, math.cos(ang) * rad, math.sin(ang) * rad, r * 0.85))
        out = S.smooth_union(0.004, out, *sprinkles)
    return out


def _cake(bp: BuildParams):
    r = 0.34 + bp.a * 0.08
    h = 0.34 + bp.b * 0.16
    base = S.at(S.cyl(r, h * 0.55), 0, 0, h * 0.275)
    top = S.at(S.cyl(r * 0.84, h * 0.40), 0, 0, h * 0.75)
    frost = S.at(S.torus(r * 0.92, 0.030), 0, 0, h * 0.55)
    frost2 = S.at(S.torus(r * 0.70, 0.030), 0, 0, h * 0.95)
    cherry = S.at(S.sphere(0.055), 0, 0, h * 0.98 + 0.05)
    out = S.smooth_union(0.02, base, top, frost, frost2, cherry)
    if bp.detail > 0.4:
        drips = S.ring_of(S.capsule((0, 0, 0), (0, 0, -0.08), 0.022), 9, r * 0.95)
        out = S.smooth_union(0.01, out, S.at(drips, 0, 0, h * 0.55))
    return out


def _icecream(bp: BuildParams):
    h = 0.46 + bp.a * 0.14
    r = 0.14 + bp.b * 0.03
    cone = S.at(S.cone(r, 0.004, h), 0, 0, h * 0.5)
    scoop1 = S.at(S.sphere(r * 1.18), 0, 0, h + r * 0.72)
    scoop2 = S.at(S.sphere(r * 0.94), r * 0.45, -r * 0.30, h + r * 1.42)
    scoop3 = S.at(S.sphere(r * 0.86), -r * 0.42, r * 0.34, h + r * 1.28)
    out = S.smooth_union(bp.k(0.03), cone, scoop1, scoop2, scoop3)
    if bp.detail > 0.45:
        drip = S.ring_of(S.capsule((0, 0, 0), (0, 0, -0.05), 0.018), 7, r * 0.92)
        out = S.smooth_union(0.01, out, S.at(drip, 0, 0, h + r * 0.30))
    return out


def _apple(bp: BuildParams):
    r = 0.30 + bp.a * 0.08
    squash = 0.88 + bp.b * 0.16
    body = S.at(S.ellipsoid(r, r, r * squash), 0, 0, r * squash)
    body = S.sub(body, S.at(S.sphere(r * 0.42), 0, 0, r * squash * 1.92))
    stem = S.at(S.cyl(0.016, 0.12), 0, 0, r * squash * 1.80)
    stem = S.rotx(stem, 0.22)
    leaf = S.at(S.ellipsoid(0.075, 0.035, 0.012), 0.07, 0, r * squash * 1.90)
    leaf = S.rotz(leaf, -0.5)
    return S.smooth_union(bp.k(0.02), body, stem, leaf)


# --------------------------------------------------------------------------
# 抽象
# --------------------------------------------------------------------------


def _heart(bp: BuildParams):
    size = 0.52 + bp.a * 0.12
    depth = 0.10 + bp.b * 0.06
    body = S.rotx(S.extrude(S.heart2(size), depth, round_r=0.03), math.pi * 0.5)
    return S.at(body, 0, 0, size * 0.85)


def _star(bp: BuildParams):
    points = int(5 + round(bp.a * 2)) * 2 - 1
    r_out = 0.48 + bp.b * 0.10
    r_in = r_out * (0.42 + bp.a * 0.10)
    body = S.rotx(S.extrude(S.star2(r_out, r_in, (points + 1) // 2), 0.075, round_r=0.02),
                  math.pi * 0.5)
    return S.at(body, 0, 0, r_out * 0.9)


def _skull(bp: BuildParams):
    r = 0.32 + bp.a * 0.08
    cranium = S.at(S.ellipsoid(r, r * 1.06, r * 1.02), 0, 0, r * 1.15)
    jaw = S.at(S.rbox(r * 0.66, r * 0.62, r * 0.34, r * 0.16), 0, -r * 0.22, r * 0.42)
    out = S.smooth_union(bp.k(0.03), cranium, jaw)
    eyes = [S.at(S.sphere(r * 0.30), sx * r * 0.42, -r * 0.80, r * 1.22) for sx in (-1, 1)]
    nose = S.at(S.rotx(S.cone(r * 0.15, 0.006, r * 0.34), math.pi), 0, -r * 0.92, r * 0.92)
    out = S.sub(out, *eyes, nose)
    if bp.detail > 0.35:
        teeth = [S.at(S.box(r * 0.038, r * 0.06, r * 0.10), (i - 3) * r * 0.135, -r * 0.74, r * 0.34)
                 for i in range(7)]
        out = S.sub(out, *teeth)
    return out


def _dice(bp: BuildParams):
    s = 0.30 + bp.a * 0.06
    body = S.rbox(s, s, s, 0.035 + bp.b * 0.04)
    pips = []
    for axis in range(3):
        for sign in (-1, 1):
            for (u, v) in ((0, 0), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)):
                if (u == 0 and v == 0 and sign > 0):
                    continue
                p = [0.0, 0.0, 0.0]
                p[axis] = sign * s
                p[(axis + 1) % 3] = u * s * 0.5
                p[(axis + 2) % 3] = v * s * 0.5
                pips.append(S.at(S.sphere(s * 0.11), *p))
    return S.sub(body, *pips)


def _chess(bp: BuildParams):
    h = 0.90 + bp.a * 0.30
    r = 0.20 + bp.b * 0.05
    prof = [
        (0.0, r * 1.5), (0.03, r * 1.5), (0.06, r * 1.15), (0.10, r * 0.95),
        (h * 0.30, r * 0.55), (h * 0.48, r * 0.42), (h * 0.56, r * 0.72),
        (h * 0.70, r * 0.62), (h * 0.80, r * 0.78), (h, r * 0.86),
    ]
    body = S.lathe(prof)
    knob = S.at(S.sphere(r * 0.72), 0, 0, h + r * 0.30)
    return S.smooth_union(0.02, body, knob)


def _top(bp: BuildParams):
    h = 0.46 + bp.a * 0.16
    r = 0.34 + bp.b * 0.10
    prof = [(0.0, 0.012), (h * 0.34, r * 0.62), (h * 0.72, r), (h * 0.86, r * 0.94), (h, r * 0.42)]
    body = S.lathe(prof)
    handle = S.at(S.cyl(0.045, h * 0.5), 0, 0, h + h * 0.24)
    return S.smooth_union(0.02, body, handle)


def _hourglass(bp: BuildParams):
    h = 0.80 + bp.a * 0.24
    r = 0.26 + bp.b * 0.06
    waist = r * 0.20
    prof = [(0.0, r), (h * 0.42, waist), (h * 0.50, waist * 0.86), (h * 0.58, waist), (h, r)]
    body = S.lathe(prof)
    if bp.hollow:
        body = S.sub(body, S.lathe([(0.02, r - bp.wall), (h * 0.44, max(waist - bp.wall, 0.01)),
                                    (h * 0.56, max(waist - bp.wall, 0.01)), (h - 0.02, r - bp.wall)]))
    caps = [S.at(S.cyl(r * 1.20, 0.05), 0, 0, z) for z in (0.0, h)]
    pillars = S.ring_of(S.cyl(0.024, h), 3, r * 1.22)
    return S.smooth_union(0.015, body, *caps, pillars)


def _tree(bp: BuildParams):
    h = 0.52 + bp.a * 0.18
    trunk = S.lathe([(0.0, 0.095), (h * 0.55, 0.070), (h * 1.05, 0.045)])
    branches = []
    for i in range(3):
        a = TAU * i / 3 + 0.4
        branches.append(S.tapered_path(
            [(0, 0, h * 0.70), (math.cos(a) * 0.16, math.sin(a) * 0.16, h * 0.95),
             (math.cos(a) * 0.28, math.sin(a) * 0.28, h * 1.18)], 0.040, 0.022))
    foliage_r = 0.23 + bp.b * 0.09
    blobs = []
    n = int(8 + round(bp.detail * 7))
    for i in range(n):
        a = TAU * i / n * 2.7
        rr = foliage_r * (0.30 + 0.55 * ((i * 37) % 10) / 10)
        zz = h * 1.14 + foliage_r * 0.55 * ((i * 13) % 7) / 7
        blobs.append(S.at(S.sphere(foliage_r * (0.58 + 0.30 * ((i * 17) % 5) / 5)),
                          math.cos(a) * rr, math.sin(a) * rr, zz))
    canopy = S.smooth_union(0.085, *blobs)
    return S.smooth_union(bp.k(0.06), trunk, *branches, canopy)


def _coral(bp: BuildParams):
    base = S.at(S.cyl(0.20 + bp.a * 0.06, 0.10), 0, 0, 0.05)
    branches = []
    n = int(4 + round(bp.b * 4))
    for i in range(n):
        a = TAU * i / n + 0.3
        pts = [(0, 0, 0.06)]
        rad = 0.10
        z = 0.10
        for j in range(3):
            rad += 0.09 + 0.03 * math.sin(i * 3 + j)
            z += 0.16 + 0.03 * math.cos(i * 2 + j)
            pts.append((math.cos(a + j * 0.4) * rad, math.sin(a + j * 0.4) * rad, z))
        branches.append(S.tapered_path(pts, 0.048, 0.018))
        # 分叉
        for s in (-0.5, 0.5):
            p0 = pts[1]
            p1 = (p0[0] + math.cos(a + s) * 0.16, p0[1] + math.sin(a + s) * 0.16, p0[2] + 0.16)
            branches.append(S.tapered_path([p0, p1], 0.030, 0.016))
    return S.smooth_union(bp.k(0.03), base, *branches)


def _crystal(bp: BuildParams):
    core = S.lathe([(0.0, 0.30 + bp.a * 0.10), (0.10, 0.34), (0.9, 0.16), (1.0, 0.004)])
    shards = []
    n = int(5 + round(bp.b * 6))
    for i in range(n):
        a = TAU * i / n
        rr = 0.16 + 0.10 * ((i * 7) % 5) / 5
        hh = 0.34 + 0.42 * ((i * 11) % 7) / 7
        shard = S.lathe([(0.0, 0.075), (hh * 0.82, 0.055), (hh, 0.004)])
        shard = S.rotx(shard, -0.28 - 0.2 * ((i * 3) % 4) / 4)
        shards.append(S.at(S.rotz(shard, a), math.cos(a) * rr, math.sin(a) * rr, 0.02))
    out = S.smooth_union(0.01, core, *shards)
    if bp.sharp > 0.2:
        out = S.facet(out, int(4 + bp.sharp * 10), seed=bp.seed, depth=0.10 + bp.sharp * 0.12)
    return out


def _knot(bp: BuildParams):
    p = 2 + int(round(bp.a * 4))
    q = 3 + int(round(bp.b * 5))
    while math.gcd(p, q) != 1:
        q += 1
    n = 90
    pts = []
    for i in range(n + 1):
        t = TAU * i / n
        r = 2.0 + math.cos(q * t)
        pts.append((r * math.cos(p * t) * 0.30, r * math.sin(p * t) * 0.30, math.sin(q * t) * 0.26))
    return S.tube(pts, 0.052, 0.052)


def _terrain(bp: BuildParams):
    from . import noise as _noise

    amp = 0.10 + bp.a * 0.42
    freq = 1.8 + bp.b * 2.2
    seed = int(bp.seed)

    def height(x, y):
        pts = np.stack([x * freq, y * freq, np.full(x.shape[0], seed * 0.01)], axis=1)
        # fbm 的实际区间只有约 0.3–0.7，先拉满量程再乘振幅，否则地形太平
        raw = _noise.fbm(pts, seed=seed, octaves=4)
        return np.maximum((raw - 0.5) * 2.0, 0.0) * amp

    def field(q):
        h = height(q[:, 0], q[:, 1])
        # 高度场本身不是距离场：除以坡度因子，把它拉回「近似距离」，
        # 否则牛顿投影会过冲，等值面在陡坡上会退化
        e = 0.012
        hx = (height(q[:, 0] + e, q[:, 1]) - h) / e
        hy = (height(q[:, 0], q[:, 1] + e) - h) / e
        return (q[:, 2] - h) / np.sqrt(1.0 + hx * hx + hy * hy)

    block = S.at(S.rbox(0.55, 0.55, 0.30, 0.02), 0, 0, 0.30 - 0.02)
    return S.inter(block, field)


def _seashell(bp: BuildParams):
    turns = 2.6 + bp.b * 2.8
    growth = 0.85 + bp.a * 0.85
    n = 80
    pts = []
    radii = []
    for i in range(n):
        u = i / (n - 1)
        th = turns * TAU * u
        r = 0.055 * math.exp(growth * u)
        pts.append((r * math.cos(th), r * math.sin(th), 0.02 + u * 0.62))
        radii.append(0.020 + 0.075 * u ** 1.6)
    # 逐段变径：用 tube 的一次求值实现
    return S.tube(pts, radii[0], radii[-1], radii=radii)


def _abstract(bp: BuildParams):
    """兜底配方：任何提示词都能得到一个确定的抽象构成。"""
    parts = []
    base = S.lathe([(0.0, 0.30), (0.06, 0.34), (0.10, 0.22), (0.14, 0.16)])
    parts.append(base)
    z = 0.14
    count = int(3 + round(bp.detail * 4))
    for i in range(count):
        kind = int(bp.rng.integers(0, 5))
        scale = float(bp.rng.uniform(0.10, 0.22)) * (1.0 - i * 0.06)
        z += scale * float(bp.rng.uniform(0.9, 1.5))
        if kind == 0:
            piece = S.sphere(scale)
        elif kind == 1:
            piece = S.ellipsoid(scale, scale * 1.5, scale * 0.7)
        elif kind == 2:
            piece = S.box(scale, scale * 0.8, scale * 0.6)
        elif kind == 3:
            piece = S.torus(scale, scale * 0.34)
            piece = S.rotx(piece, float(bp.rng.uniform(0, math.pi)))
        else:
            piece = S.hexprism(scale, scale * 1.4, sides=int(bp.rng.integers(3, 8)))
        off = float(bp.rng.uniform(-0.08, 0.08))
        parts.append(S.at(piece, off, off * 0.7, z))
    body = S.smooth_union(bp.k(0.06), *parts)
    if bp.twist > 0.1:
        body = S.twist(body, bp.twist * 3.0, height=z)
    return body


# --------------------------------------------------------------------------
# 概念登记表
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    category: str
    note: str
    keywords: tuple[str, ...]
    build: Callable[[BuildParams], object]
    pa: tuple[str, str] = ("比例", "整体比例")
    pb: tuple[str, str] = ("细节", "细节强度")
    hint: str = ""
    hollow_default: bool = False


CONCEPTS: tuple[Concept, ...] = (
    # ---------------- 器物
    Concept("vase", "花瓶", "器物", "旋转体 · 收腰曲线",
            ("花瓶", "花器", "瓶", "vase", "vessel", "urn"),
            _vase, ("收腰程度", "瓶身中段收得多紧"), ("瓣数", "表面竖棱的数量"),
            "壁厚 1.6mm 起；想插花把瓶口内径放大。", True),
    Concept("mug", "马克杯", "器物", "杯身 + 环形把手",
            ("杯子", "马克杯", "杯", "mug", "cup", "coffee"),
            _mug, ("杯身高度", "越高越修长"), ("杯口半径", "越大越能装"),
            "把手与杯身连接处加 0.4mm 圆角更结实。", True),
    Concept("bowl", "碗", "器物", "浅口旋转体",
            ("碗", "钵", "盆", "bowl", "basin"),
            _bowl, ("碗口半径", "越大越浅"), ("碗深", "越深越像汤碗"),
            "倒扣打印可免支撑；内壁建议 0.16mm 层高。", True),
    Concept("bottle", "瓶子", "器物", "细颈旋转体",
            ("瓶子", "壶", "水瓶", "bottle", "flask"),
            _bottle, ("瓶身粗细", "越大越敦实"), ("瓶颈半径", "越粗越稳"),
            "竖立打印最省支撑；瓶口可后加螺纹。", True),
    Concept("planter", "花盆", "器物", "锥形盆 + 接水盘",
            ("花盆", "盆栽", "花器", "planter", "pot", "flowerpot"),
            _planter, ("盆口半径", "越大越宽"), ("盆底收缩", "越小越难脱模"),
            "底部建议开 Ø6mm 排水孔（可在切片软件里打）。", True),
    Concept("teapot", "茶壶", "器物", "球身 + 壶嘴 + 提梁",
            ("茶壶", "壶", "teapot", "kettle"),
            _teapot, ("壶身半径", "越大越圆"), ("壶嘴粗细", "越粗出水越快"),
            "壶嘴内孔建议 ≥Ø6mm，否则容易挂水。", True),
    Concept("candle", "烛台", "器物", "底盘 + 立柱 + 烛杯",
            ("烛台", "蜡烛", "香薰", "candle", "candlestick"),
            _candle, ("立柱高度", "越高越挺拔"), ("是否带火焰", "大于 50 加火焰造型"),
            "烛杯内径按蜡烛实际尺寸留 0.3mm 余量。"),
    Concept("lampshade", "灯罩", "器物", "锥形薄壁罩",
            ("灯罩", "灯", "lampshade", "shade"),
            _lampshade, ("下口半径", "越大光越散"), ("上口半径", "越小越聚光"),
            "薄壁灯罩建议 0.6mm 喷嘴 + 0.12mm 层高，透光更均匀。", True),
    Concept("penholder", "笔筒", "器物", "多边形筒身",
            ("笔筒", "笔架", "pen", "penholder", "pencil"),
            _penholder, ("筒身半径", "越大越能装"), ("边数", "3 棱到 8 棱"),
            "底面加大 1mm 边沿可提升站立稳定性。", True),
    Concept("tray", "托盘", "器物", "圆角浅盘 + 提手",
            ("托盘", "盘子", "收纳", "tray", "dish", "plate"),
            _tray, ("长度", "越长越能放"), ("宽度", "越宽越稳"),
            "平放打印，不需要支撑。", True),
    # ---------------- 家具
    Concept("chair", "椅子", "家具", "座面 + 靠背 + 四腿",
            ("椅子", "凳", "座椅", "chair", "seat"),
            _chair, ("座面高度", "越高越像吧椅"), ("座面宽度", "越宽越舒适"),
            "1:12 模型可直接打印；实尺需拆分并加销钉。"),
    Concept("table", "桌子", "家具", "桌面 + 四条腿",
            ("桌子", "台", "table", "desk"),
            _table, ("桌面长度", "越长越像餐桌"), ("桌面宽度", "越宽越稳"),
            "细腿建议横放打印，层纹方向更抗弯。"),
    Concept("stool", "圆凳", "家具", "圆面 + 三腿",
            ("凳子", "圆凳", "stool"),
            _stool, ("凳面高度", "越高越难站稳"), ("凳面半径", "越大越稳"),
            "三腿结构最省材料，重心低更稳。"),
    Concept("shelf", "书架", "家具", "层板 + 侧板",
            ("书架", "置物架", "架子", "shelf", "rack"),
            _shelf, ("宽度", "越宽越能放"), ("深度", "越深越稳"),
            "层板厚 ≥2mm 才不会因自重下垂。"),
    Concept("lamp", "台灯", "家具", "底座 + 灯杆 + 灯罩",
            ("台灯", "落地灯", "lamp", "light"),
            _lamp, ("底座半径", "越大越稳"), ("灯杆高度", "越高越像落地灯"),
            "灯罩与灯泡之间留 8mm 散热间隙。", True),
    # ---------------- 建筑
    Concept("house", "房子", "建筑", "坡屋顶 + 门窗",
            ("房子", "小屋", "住宅", "house", "home", "cabin"),
            _house, ("面宽", "越宽越扁"), ("进深", "越深越像别墅"),
            "屋顶悬挑处需要支撑；也可拆成两件粘合。"),
    Concept("tower", "塔", "建筑", "收分塔身 + 城垛 + 尖顶",
            ("塔", "塔楼", "钟楼", "tower", "spire", "turret"),
            _tower, ("塔高", "越高越挺拔"), ("塔身半径", "越粗越敦实"),
            "高塔建议分段打印后对接。"),
    Concept("castle", "城堡", "建筑", "主体 + 四角塔",
            ("城堡", "要塞", "castle", "fort"),
            _castle, ("主楼宽度", "越宽越气派"), ("主楼高度", "越高越像要塞"),
            "角塔与主楼交接处会自动圆角过渡，不易断。"),
    Concept("arch", "拱门", "建筑", "拱券 + 门洞",
            ("拱门", "拱", "门", "arch", "gateway"),
            _arch, ("门宽", "越宽越舒展"), ("门高", "越高越修长"),
            "拱券顶部需要支撑；侧放打印更省支撑。"),
    Concept("bridge", "拱桥", "建筑", "桥面 + 桥墩 + 拱",
            ("桥", "拱桥", "bridge", "viaduct"),
            _bridge, ("跨度", "越长越壮观"), ("桥面高度", "越高拱越明显"),
            "拱洞下方必须加支撑，或整体侧放打印。"),
    Concept("stairs", "楼梯", "建筑", "踏步 + 侧墙",
            ("楼梯", "台阶", "阶梯", "stairs", "steps"),
            _stairs, ("台阶数", "越多越陡长"), ("踏步进深", "越深越好走"),
            "按 45° 侧放打印，踏步面朝上最省支撑。"),
    Concept("dome", "穹顶", "建筑", "半球顶 + 基座环",
            ("穹顶", "圆顶", "dome", "cupola"),
            _cupola, ("穹顶半径", "越大越宏伟"), ("基座高度", "越高越稳"),
            "穹顶内表面需要支撑；可开启「中空」减重。"),
    # ---------------- 生物
    Concept("cat", "猫", "生物", "坐姿 + 尖耳 + 卷尾",
            ("猫", "猫咪", "小猫", "cat", "kitten", "kitty"),
            _cat, ("体型比例", "越胖越可爱"), ("耳朵高度", "越尖越精神"),
            "尾巴较细，建议树脂打印或放大到 120mm。"),
    Concept("dog", "狗", "生物", "站姿 + 长吻 + 垂耳",
            ("狗", "小狗", "犬", "dog", "puppy", "hound"),
            _dog, ("体长比例", "越长越像腊肠"), ("头部大小", "越大越卡通"),
            "四条腿同时着地，站立打印无需支撑。"),
    Concept("rabbit", "兔子", "生物", "长耳 + 圆身",
            ("兔子", "兔", "rabbit", "bunny", "hare"),
            _rabbit, ("耳朵长度", "越长越萌"), ("身体圆度", "越圆越可爱"),
            "长耳建议横放打印，避免层间断裂。"),
    Concept("bear", "熊", "生物", "敦实体型 + 圆耳",
            ("熊", "小熊", "bear", "teddy"),
            _bear, ("体型宽窄", "越宽越憨"), ("头部大小", "越大越卡通"),
            "整体厚度均匀，几乎不需要支撑。"),
    Concept("bird", "小鸟", "生物", "身体 + 喙 + 尾羽",
            ("鸟", "小鸟", "雀", "bird", "sparrow", "robin"),
            _bird, ("身体长度", "越长越像雀"), ("尾巴大小", "越大越飘逸"),
            "细腿易断，建议开启「底座」或放大尺寸。"),
    Concept("fish", "鱼", "生物", "纺锤体 + 尾鳍",
            ("鱼", "小鱼", "金鱼", "fish", "goldfish"),
            _fish, ("身体长度", "越长越修长"), ("鱼鳍大小", "越大越飘逸"),
            "侧躺打印最省支撑，鳍面朝上。"),
    Concept("dinosaur", "恐龙", "生物", "长颈 + 长尾 + 四足",
            ("恐龙", "龙", "dinosaur", "dino", "rex", "brachiosaurus"),
            _dinosaur, ("脖颈长度", "越长越像腕龙"), ("尾巴粗细", "越粗越稳"),
            "长颈长尾是悬臂结构，建议加厚或缩小尺寸。"),
    Concept("octopus", "章鱼", "生物", "圆头 + 八腕",
            ("章鱼", "八爪鱼", "octopus", "squid"),
            _octopus, ("腕足数量", "6–9 条"), ("头部大小", "越大越呆萌"),
            "腕足尖端很细，建议树脂打印。"),
    Concept("turtle", "乌龟", "生物", "龟壳 + 四足",
            ("乌龟", "龟", "turtle", "tortoise"),
            _turtle, ("龟壳半径", "越大越圆"), ("壳顶高度", "越高越鼓"),
            "四足平底，站立打印即可。"),
    Concept("duck", "小鸭", "生物", "圆身 + 扁喙",
            ("鸭子", "鸭", "小鸭", "duck", "duckling"),
            _duck, ("身体长度", "越长越像鸭"), ("头部比例", "越大越卡通"),
            "扁喙悬空较大，建议加支撑。"),
    # ---------------- 机械
    Concept("robot", "机器人", "机械", "方块躯干 + 关节四肢",
            ("机器人", "机甲", "robot", "mech", "android"),
            _robot, ("躯干宽度", "越宽越壮"), ("头部大小", "越大越卡通"),
            "分件打印后用 Ø3mm 销钉连接，可做可动模型。"),
    Concept("rocket", "火箭", "机械", "锥头 + 尾翼",
            ("火箭", "飞船", "航天", "rocket", "spaceship", "missile"),
            _rocket, ("箭体长度", "越长越修长"), ("箭体半径", "越粗越敦实"),
            "尾翼是薄片，建议侧放打印提高强度。"),
    Concept("car", "汽车", "机械", "车身 + 座舱 + 四轮",
            ("汽车", "车", "小汽车", "car", "automobile", "vehicle"),
            _car, ("车长", "越长越像轿车"), ("车宽", "越宽越稳"),
            "轮子与车身一体，若要做可转动请分件打印。"),
    Concept("plane", "飞机", "机械", "机身 + 机翼 + 尾翼",
            ("飞机", "客机", "plane", "airplane", "aircraft", "jet"),
            _plane, ("机身长度", "越长越像客机"), ("翼展比例", "越大越像滑翔机"),
            "机翼较薄，建议竖立打印（机头朝上）。"),
    Concept("gear", "齿轮", "机械", "轮齿 + 中心孔",
            ("齿轮", "齿", "gear", "cog", "sprocket"),
            _gear, ("齿数", "8–18 齿"), ("齿轮半径", "越大越像飞轮"),
            "齿隙建议 0.15mm；配合件按 0.2mm 间隙打印。"),
    Concept("screw", "螺丝", "机械", "螺旋牙 + 十字槽",
            ("螺丝", "螺钉", "螺栓", "screw", "bolt"),
            _screw, ("螺纹长度", "越长越像机螺丝"), ("螺距粗细", "越粗越吃木"),
            "螺纹建议 0.1mm 层高，横放打印精度最高。"),
    Concept("sword", "剑", "机械", "剑身 + 护手 + 剑柄",
            ("剑", "刀", "长剑", "sword", "blade", "katana"),
            _sword, ("剑身长度", "越长越像双手剑"), ("剑身宽度", "越宽越像阔剑"),
            "剑身平放打印，刃口朝上，避免层间断裂。"),
    Concept("shield", "盾牌", "机械", "盾面 + 中心凸起",
            ("盾牌", "盾", "徽章", "shield", "buckler"),
            _shield, ("盾面宽度", "越宽越像塔盾"), ("盾面高度", "越高越修长"),
            "平放打印，盾面朝上最省支撑。"),
    # ---------------- 食物
    Concept("mushroom", "蘑菇", "食物", "伞盖 + 菌柄",
            ("蘑菇", "菌", "香菇", "mushroom", "fungus"),
            _mushroom, ("伞盖半径", "越大越像香菇"), ("菌柄高度", "越高越纤细"),
            "伞盖下方需要支撑；也可倒置打印。"),
    Concept("donut", "甜甜圈", "食物", "圆环 + 糖霜",
            ("甜甜圈", "面包圈", "donut", "doughnut"),
            _donut, ("圆环半径", "越大越扁"), ("环管粗细", "越粗越松软"),
            "平放打印，糖霜朝上。"),
    Concept("cake", "蛋糕", "食物", "双层蛋糕 + 樱桃",
            ("蛋糕", "甜点", "cake", "dessert"),
            _cake, ("蛋糕半径", "越大越像生日蛋糕"), ("蛋糕高度", "越高越多层"),
            "奶油边缘是悬垂，建议加支撑。"),
    Concept("icecream", "冰淇淋", "食物", "蛋筒 + 三球",
            ("冰淇淋", "雪糕", "甜筒", "icecream", "ice cream", "gelato"),
            _icecream, ("蛋筒高度", "越高越尖"), ("球体大小", "越大越满"),
            "倒置（蛋筒朝上）打印可完全免支撑。"),
    Concept("apple", "苹果", "食物", "球体 + 果柄 + 叶子",
            ("苹果", "水果", "apple", "fruit"),
            _apple, ("果实半径", "越大越圆"), ("果实扁度", "越扁越像苹果"),
            "果柄较细，建议 0.12mm 层高。"),
    # ---------------- 抽象
    Concept("heart", "心形", "抽象", "经典心形 + 圆角",
            ("心形", "爱心", "心", "heart", "love"),
            _heart, ("心形大小", "越大越饱满"), ("厚度", "越厚越像摆件"),
            "平躺打印，尖端朝上需要少量支撑。"),
    Concept("star", "星星", "抽象", "多角星 + 圆角",
            ("星星", "星", "五角星", "star", "pentagram"),
            _star, ("角数", "5–7 角"), ("星角大小", "越大越尖"),
            "平躺打印最省支撑。"),
    Concept("skull", "骷髅", "抽象", "头骨 + 眼窝 + 牙",
            ("骷髅", "头骨", "骨", "skull", "cranium"),
            _skull, ("头骨半径", "越大越圆"), ("下颌宽度", "越宽越方"),
            "眼窝是深腔，建议开启「中空」并加排水孔。"),
    Concept("dice", "骰子", "抽象", "圆角立方 + 点数",
            ("骰子", "色子", "dice", "die"),
            _dice, ("骰子大小", "越大越像摆件"), ("圆角半径", "越大越圆润"),
            "点数凹坑深度 ≤1mm，避免透光。"),
    Concept("chess", "国际象棋", "抽象", "经典棋子剖面",
            ("棋子", "国际象棋", "chess", "pawn", "chessman"),
            _chess, ("棋子高度", "越高越修长"), ("底座半径", "越大越稳"),
            "底座直径 ≥1.4 倍身高才稳。"),
    Concept("top", "陀螺", "抽象", "旋转体 + 手柄",
            ("陀螺", "转", "top", "spinner", "spinning top"),
            _top, ("陀螺高度", "越高越难转"), ("陀螺半径", "越大越稳"),
            "平底打印后打磨底面，转得更久。"),
    Concept("hourglass", "沙漏", "抽象", "双锥腔 + 三柱",
            ("沙漏", "沙", "hourglass", "sandglass"),
            _hourglass, ("沙漏高度", "越高越久"), ("瓶体半径", "越大越能装"),
            "腔体需中空 + 顶部开孔，建议树脂打印。", True),
    Concept("tree", "树", "抽象", "树干 + 树冠",
            ("树", "树木", "森林", "tree", "plant", "bonsai"),
            _tree, ("树干高度", "越高越像乔木"), ("树冠半径", "越大越茂密"),
            "树冠是悬垂结构，需要大量支撑。"),
    Concept("coral", "珊瑚", "抽象", "分枝结构",
            ("珊瑚", "礁", "coral", "reef"),
            _coral, ("枝干数量", "越多越繁密"), ("分枝长度", "越长越飘逸"),
            "枝干细长，建议树脂打印并加粗 20%。"),
    Concept("knot", "环面纽结", "抽象", "数学纽结 · 管状扫掠",
            ("纽结", "结", "拓扑", "knot", "torus", "loop", "topology"),
            _knot, ("绕轴数 p", "绕大圆缠绕的圈数"), ("绕管数 q", "绕小圆缠绕的圈数"),
            "管径较细，建议加底座或放大到 120mm 打印。"),
    Concept("terrain", "地形切片", "抽象", "分形高度场 · 封闭地块",
            ("地形", "地块", "山", "地貌", "等高线", "terrain", "landscape", "mountain", "map"),
            _terrain, ("起伏幅度", "山峰与谷底的高差"), ("地形频率", "越大山脊越细碎"),
            "平放打印最省支撑。"),
    Concept("seashell", "海螺", "抽象", "对数螺旋 · 渐扩管体",
            ("海螺", "贝壳", "螺", "shell", "conch", "nautilus", "snail"),
            _seashell, ("生长速率", "螺线张开的快慢"), ("盘旋圈数", "越多越紧致"),
            "尖端起翘处需要支撑，也可竖立打印。"),
    Concept("crystal", "晶体", "抽象", "棱柱晶簇",
            ("晶体", "水晶", "矿石", "宝石", "crystal", "gem", "mineral"),
            _crystal, ("主晶高度", "越高越修长"), ("晶柱数量", "越多越繁密"),
            "尖锐处建议 0.1mm 层高 + 哑光透明漆。"),
    Concept("abstract", "抽象构成", "抽象", "兜底配方 · 任意提示词",
            ("抽象", "雕塑", "构成", "abstract", "sculpture", "totem"),
            _abstract, ("比例", "整体比例"), ("层数", "构成层次"),
            "没有匹配到具体概念时，会用提示词哈希生成独特的抽象构成。"),
)

CONCEPT_MAP: dict[str, Concept] = {c.id: c for c in CONCEPTS}
CONCEPT_IDS: tuple[str, ...] = tuple(c.id for c in CONCEPTS)
CATEGORIES: tuple[str, ...] = ("器物", "家具", "建筑", "生物", "机械", "食物", "抽象")

DEFAULT_CONCEPT = "vase"

CATEGORY_PALETTE = {
    "器物": "frost",
    "家具": "amber",
    "建筑": "steel",
    "生物": "moss",
    "机械": "steel",
    "食物": "coral",
    "抽象": "amethyst",
}


def palette_for(concept_id: str) -> str:
    return CATEGORY_PALETTE.get(get(concept_id).category, "frost")


def get(concept_id: str) -> Concept:
    return CONCEPT_MAP.get(concept_id) or CONCEPT_MAP[DEFAULT_CONCEPT]


def by_category(category: str) -> tuple[Concept, ...]:
    return tuple(c for c in CONCEPTS if c.category == category)


def keyword_index() -> list[tuple[str, str]]:
    """(关键词, 概念 id)，长词优先，避免「小猫」被「猫」抢先。"""
    pairs: list[tuple[str, str]] = []
    for c in CONCEPTS:
        for kw in c.keywords:
            pairs.append((kw.lower(), c.id))
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
    return pairs


def match(text: str) -> tuple[str | None, str | None]:
    """在文本里找最匹配的概念（最长关键词优先）。"""
    low = (text or "").lower()
    best: tuple[int, str, str] | None = None
    for kw, cid in keyword_index():
        if kw and kw in low:
            if best is None or len(kw) > best[0]:
                best = (len(kw), cid, kw)
    if best:
        return best[1], best[2]
    return None, None


def fallback_for(text: str) -> str:
    """没命中关键词时：确定性地落回「抽象构成」。"""
    return "abstract"


# --------------------------------------------------------------------------
# 生成入口
# --------------------------------------------------------------------------


def build(concept_id: str, params: BuildParams, *, apply_style: bool = True):
    """生成 SDF 实体（含全局风格修饰）。"""
    concept = get(concept_id)
    fn = concept.build(params)
    if not apply_style:
        return fn

    # 全局风格修饰：所有概念共享同一套「旋钮」，避免每个配方各写一套
    if params.twist > 0.08 and concept_id not in ("vase", "abstract"):
        fn = S.twist(fn, params.twist * 1.6, height=1.1)
    if params.sharp > 0.25 and concept.category not in ("机械", "建筑"):
        fn = S.facet(fn, int(3 + params.sharp * 9), seed=params.seed, depth=0.08 + params.sharp * 0.10)
    if params.rough > 0.05:
        fn = S.displace(fn, 0.006 + params.rough * 0.020, freq=5.0, seed=params.seed, octaves=2)
    # 保证「换种子换造型」：极小的噪声位移，肉眼几乎不可见，但足以改变网格
    fn = S.displace(fn, 0.0022, freq=2.2, seed=params.seed + 991, octaves=2)
    return fn
