"""tests/test_core.py —— 核心逻辑回归测试（无界面，可直接跑）。

覆盖：网格自洽性 / 生成器确定性与参数响应 / 导出文件合法性 / 提示词解析 /
图像分析 / 会话撤销重做 / 设置持久化 / 创意文案确定性 / 注册表一致性。

运行：  python tests/test_core.py         （或 python -m unittest discover tests）
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from creative3d import config, exporter, generators, ideas, imaging, nlp, registry  # noqa: E402
from creative3d.errors import AppError  # noqa: E402
from creative3d.generators import GenContext  # noqa: E402
from creative3d.mesh import Mesh, safe_filename  # noqa: E402
from creative3d.session import Session  # noqa: E402
from creative3d.settings import Settings  # noqa: E402

ALL_FORMS = [f.id for f in registry.FORMS]


def ctx(seed: int = 7, res: int = 40, a: float = 50.0, b: float = 50.0) -> GenContext:
    return GenContext(seed=seed, res=res, a=a, b=b)


# ---------------------------------------------------------------------------
class TestMesh(unittest.TestCase):
    def test_empty_mesh(self):
        m = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        self.assertTrue(m.is_empty())
        self.assertEqual(m.n_faces, 0)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            Mesh(np.zeros((3, 2), np.float32), np.zeros((1, 3), np.int32))

    def test_normalize_puts_floor_at_zero(self):
        m = generators.build("vase", ctx()).normalize(100)
        self.assertAlmostEqual(float(m.verts[:, 2].min()), 0.0, places=3)
        lo, hi = m.bounds()
        self.assertAlmostEqual(float((hi - lo).max()), 100.0, places=2)

    def test_vertex_normals_unit_length(self):
        m = generators.build("shell", ctx(res=30))
        lengths = np.linalg.norm(m.vertex_normals(), axis=1)
        used = np.unique(m.faces)
        self.assertTrue(np.allclose(lengths[used], 1.0, atol=1e-3))

    def test_stl_bytes_length_matches_face_count(self):
        m = generators.build("vase", ctx(res=30))
        data = m.stl_bytes("unit")
        self.assertEqual(len(data), 84 + m.n_faces * 50)

    def test_obj_text_has_geometry(self):
        m = generators.build("vase", ctx(res=30))
        text = m.obj_text("unit", palette="frost")
        self.assertTrue(text.count("\nf ") > 10)
        self.assertTrue(text.count("\nv ") > 10)

    def test_validate_returns_list(self):
        self.assertIsInstance(generators.build("vase", ctx(res=30)).validate(), list)

    def test_safe_filename(self):
        name = safe_filename('a/b\\c:d*e?f"g<h>i|j')
        for ch in '/\\:*?"<>|':
            self.assertNotIn(ch, name)
        self.assertTrue(safe_filename("   "))


# ---------------------------------------------------------------------------
class TestGenerators(unittest.TestCase):
    def test_every_form_produces_non_empty_mesh(self):
        for fid in ALL_FORMS:
            for res in (5, 40, 100):
                with self.subTest(form=fid, res=res):
                    m = generators.build(fid, ctx(res=res))
                    self.assertFalse(m.is_empty(), f"{fid}@{res} 生成了空网格")
                    self.assertGreater(m.n_faces, 8)

    def test_no_nan_or_inf(self):
        for fid in ALL_FORMS:
            m = generators.build(fid, ctx(res=60))
            with self.subTest(form=fid):
                self.assertTrue(np.isfinite(m.verts).all(), f"{fid} 顶点含 NaN/Inf")

    def test_vertices_finite_range(self):
        for fid in ALL_FORMS:
            m = generators.build(fid, ctx(res=45)).normalize(100)
            with self.subTest(form=fid):
                self.assertLess(np.abs(m.verts).max(), 500.0)

    def test_deterministic_same_seed(self):
        for fid in ALL_FORMS:
            a = generators.build(fid, ctx(seed=123, res=50))
            b = generators.build(fid, ctx(seed=123, res=50))
            with self.subTest(form=fid):
                np.testing.assert_array_equal(a.verts, b.verts)
                np.testing.assert_array_equal(a.faces, b.faces)

    def test_different_seed_changes_shape(self):
        for fid in ALL_FORMS:
            a = generators.build(fid, ctx(seed=1, res=50))
            b = generators.build(fid, ctx(seed=999, res=50))
            with self.subTest(form=fid):
                self.assertFalse(np.allclose(a.verts, b.verts), f"{fid} 换种子没变化")

    def test_watertight(self):
        """所有形态都必须是封闭流形，否则切片软件会打出破面。"""
        for fid in ALL_FORMS:
            m = generators.build(fid, ctx(res=40))
            ok, boundary, nonmanifold = m.check_watertight()
            with self.subTest(form=fid):
                self.assertTrue(ok, f"{fid} 非封闭：边界边 {boundary}，非流形 {nonmanifold}")

    def test_outward_normals(self):
        for fid in ALL_FORMS:
            m = generators.build(fid, ctx(res=40))
            with self.subTest(form=fid):
                self.assertGreater(m.signed_volume(), 0.0, f"{fid} 法线朝内")

    def test_resolution_increases_face_count(self):
        """密度滑杆必须真的改变面数（不能有任何形态对它免疫）。"""
        for fid in ALL_FORMS:
            lo = generators.build(fid, ctx(res=10)).n_faces
            hi = generators.build(fid, ctx(res=90)).n_faces
            with self.subTest(form=fid):
                self.assertGreater(hi, lo, f"{fid} 提高密度后面数没增加")

    def test_params_change_shape(self):
        """参数必须真的改变几何（顶点数不同也算改变，所以不能直接 allclose）。"""
        for fid in ALL_FORMS:
            a = generators.build(fid, ctx(res=45, a=15, b=50))
            b = generators.build(fid, ctx(res=45, a=85, b=50))
            with self.subTest(form=fid):
                same = a.verts.shape == b.verts.shape and np.allclose(a.verts, b.verts)
                self.assertFalse(same, f"{fid} 参数 A 无效")

    def test_extreme_params_do_not_crash(self):
        for fid in ALL_FORMS:
            for a, b in ((0.0, 0.0), (100.0, 100.0), (0.0, 100.0), (100.0, 0.0)):
                with self.subTest(form=fid, a=a, b=b):
                    m = generators.build(fid, ctx(res=25, a=a, b=b))
                    self.assertFalse(m.is_empty())
                    self.assertTrue(np.isfinite(m.verts).all())

    def test_unknown_form_raises_app_error(self):
        with self.assertRaises(AppError):
            generators.build("不存在的形态", ctx())

    def test_face_indices_in_range(self):
        for fid in ALL_FORMS:
            m = generators.build(fid, ctx(res=55))
            with self.subTest(form=fid):
                self.assertGreaterEqual(int(m.faces.min()), 0)
                self.assertLess(int(m.faces.max()), len(m.verts))

    def test_resolution_capped(self):
        m = generators.build("knot", ctx(res=100))
        self.assertLess(m.n_faces, 400_000)

    def test_build_speed(self):
        """高端机器上单个形态不应卡住界面。"""
        import time

        t = time.perf_counter()
        for fid in ALL_FORMS:
            generators.build(fid, ctx(res=70))
        self.assertLess(time.perf_counter() - t, 8.0)


# ---------------------------------------------------------------------------
class TestExporter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="c3d_exp_")
        self.dir = Path(self.tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bundle(self, **overrides) -> exporter.ExportResult:
        opts = exporter.ExportOptions(stl=True, obj=True, png=True, card=True,
                                      png_width=320, png_height=240)
        mesh = generators.build("vase", ctx(res=30)).normalize(100)
        idea = ideas.generate_idea("vase", 7, res=30, params={"a": 50, "b": 50})
        kwargs = dict(
            mesh=mesh, form_id="vase", seed=7, res=30.0, params={"a": 50, "b": 50},
            palette_id="frost", palette=config.PALETTES["frost"], idea=idea,
            image_info=None, out_dir=self.dir, options=opts,
        )
        kwargs.update(overrides)
        return exporter.export_bundle(**kwargs)

    def test_bundle_creates_all_selected(self):
        result = self._bundle()
        self.assertEqual(len(result.files), 4)
        for f in result.files:
            self.assertTrue(f.exists(), f"{f} 没写出来")
            self.assertGreater(f.stat().st_size, 0)

    def test_bundle_respects_disabled_formats(self):
        opts = exporter.ExportOptions(stl=True, obj=False, png=False, card=False)
        result = self._bundle(options=opts)
        self.assertEqual(len(result.files), 1)
        self.assertTrue(result.files[0].name.endswith(".stl"))

    def test_png_has_requested_size(self):
        from PIL import Image as _I

        self._bundle()
        png = next(self.dir.glob("*.png"))
        with _I.open(png) as img:
            self.assertEqual(img.size, (320, 240))

    def test_obj_bundle_writes_geometry(self):
        self._bundle()
        obj = next(self.dir.glob("*.obj"))
        text = obj.read_text(encoding="utf-8")
        self.assertTrue(text.count("\nf ") > 10)

    def test_card_markdown_contains_title(self):
        self._bundle()
        md = next(self.dir.glob("*.md"))
        self.assertIn("#", md.read_text(encoding="utf-8"))

    def test_empty_mesh_raises_typed_error(self):
        empty = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        with self.assertRaises(AppError):
            self._bundle(mesh=empty)

    def test_unwritable_dir_raises_typed_error(self):
        with self.assertRaises(AppError):
            self._bundle(out_dir=Path("Z:/这个盘不存在/nope"))

    def test_unique_names_do_not_overwrite(self):
        self._bundle()
        self._bundle()
        stls = list(self.dir.glob("*.stl"))
        self.assertGreaterEqual(len(stls), 2)

    def test_summary_text(self):
        self.assertIsInstance(self._bundle().summary(), str)

    def test_cancellation_token_stops_export(self):
        from creative3d.jobs import Token

        token = Token(name="test")
        token.cancel()
        with self.assertRaises(Exception):
            self._bundle(token=token)


# ---------------------------------------------------------------------------
class TestNLP(unittest.TestCase):
    def test_form_keywords(self):
        cases = {
            "做个花瓶": "vase",
            "晶体簇": "crystal",
            "扭结": "knot",
            "螺旋塔": "tower",
            "地形": "terrain",
            "海螺": "shell",
            "浮雕": "relief",
            "用这张图的轮廓": "profile",
        }
        for text, expect in cases.items():
            with self.subTest(text=text):
                self.assertEqual(nlp.parse(text).form_id, expect)

    def test_seed_extraction(self):
        for text in ("种子 2026", "seed=42", "seed 8888", "随机种子 123", "编号 77"):
            with self.subTest(text=text):
                self.assertIsNotNone(nlp.parse(text).seed)

    def test_seed_value(self):
        self.assertEqual(nlp.parse("seed=42").seed, 42)
        self.assertEqual(nlp.parse("种子 2026").seed, 2026)

    def test_resolution_digits(self):
        self.assertEqual(nlp.parse("细节 80").res, 80.0)

    def test_resolution_chinese_number(self):
        self.assertEqual(nlp.parse("细节八十").res, 80.0)
        self.assertEqual(nlp.parse("密度三十五").res, 35.0)

    def test_resolution_keywords(self):
        hi = nlp.parse("细节拉满 高精度").res
        lo = nlp.parse("低模 极简").res
        self.assertIsNotNone(hi)
        self.assertIsNotNone(lo)
        self.assertGreater(hi, lo)

    def test_resolution_clamped(self):
        self.assertLessEqual(nlp.parse("细节 999").res, 100.0)

    def test_palette_keywords(self):
        for word, pid in (("冷色", "frost"), ("紫晶", "amethyst"), ("琥珀", "amber"),
                          ("苔绿", "moss"), ("金属", "steel"), ("珊瑚", "coral")):
            with self.subTest(word=word):
                self.assertEqual(nlp.parse(word).palette, pid)

    def test_style_words_move_params(self):
        sharp = nlp.parse("尖锐")
        self.assertIn("b", sharp.param_delta)
        self.assertGreater(sharp.param_delta["b"], 0)

    def test_random_keyword(self):
        self.assertTrue(nlp.parse("随便来一个").random_seed)

    def test_empty_prompt_is_safe(self):
        r = nlp.parse("")
        self.assertFalse(r.understood)
        self.assertIsInstance(r.summary(), str)

    def test_garbage_prompt_is_safe(self):
        r = nlp.parse("!!!@@@   烫烫烫")
        self.assertIsInstance(r.summary(), str)

    def test_whitespace_and_fullwidth(self):
        self.assertEqual(nlp.parse("　花瓶　").form_id, "vase")

    def test_summary_is_human_readable(self):
        text = nlp.parse("冷色的尖锐晶簇，种子 2026，细节 80").summary()
        self.assertIsInstance(text, str)
        self.assertTrue(text)

    def test_apply_to_clamps(self):
        merged = nlp.apply_to("vase", {"a": 95.0, "b": 10.0}, nlp.parse("尖锐 扭曲"))
        for key in ("a", "b"):
            p = registry.get("vase").param(key)
            self.assertGreaterEqual(merged[key], p.lo)
            self.assertLessEqual(merged[key], p.hi)

    def test_keywords_help(self):
        text = nlp.keywords_help()
        self.assertIsInstance(text, str)
        self.assertIn("种子", text)


# ---------------------------------------------------------------------------
class TestImaging(unittest.TestCase):
    @staticmethod
    def _sample() -> Image.Image:
        img = Image.new("RGB", (240, 240), (250, 250, 250))
        d = ImageDraw.Draw(img)
        d.ellipse([40, 40, 200, 200], fill=(10, 20, 30))
        d.rectangle([90, 90, 150, 150], fill=(200, 40, 40))
        return img

    def test_analyze_returns_features(self):
        feats = imaging.analyze(self._sample())
        self.assertGreater(len(feats.colors), 0)
        self.assertGreater(feats.heightmap(16).size, 0)
        self.assertGreater(feats.width, 0)

    def test_height_map_range(self):
        feats = imaging.analyze(self._sample())
        h = feats.heightmap(32)
        self.assertEqual(h.shape, (32, 32))
        self.assertGreaterEqual(float(h.min()), 0.0)
        self.assertLessEqual(float(h.max()), 1.0)

    def test_radial_profile(self):
        feats = imaging.analyze(self._sample())
        profile = feats.radial_profile(24)
        self.assertEqual(len(profile), 24)
        self.assertTrue(np.isfinite(profile).all())

    def test_blank_image_does_not_crash(self):
        feats = imaging.analyze(Image.new("RGB", (64, 64), (128, 128, 128)))
        self.assertGreaterEqual(float(feats.heightmap(16).min()), 0.0)

    def test_colors_are_hex(self):
        for c in imaging.analyze(self._sample()).colors:
            self.assertTrue(c.startswith("#"))
            self.assertEqual(len(c), 7)

    def test_contrast_and_symmetry_in_range(self):
        feats = imaging.analyze(self._sample())
        for value in (feats.contrast, feats.symmetry, feats.edge_density):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_summary_text(self):
        self.assertIsInstance(imaging.analyze(self._sample()).summary(), str)

    def test_palette_from_features(self):
        pal = imaging.palette_from_features(imaging.analyze(self._sample()))
        self.assertEqual(len(pal.hex("base")), 7)

    def test_thumbnail(self):
        thumb = imaging.make_thumbnail(self._sample(), 64)
        self.assertLessEqual(max(thumb.size), 64)

    def test_unsupported_file_raises_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "x.txt"
            bad.write_text("not an image", encoding="utf-8")
            with self.assertRaises(AppError):
                imaging.analyze_path(bad)

    def test_missing_file_raises_typed_error(self):
        with self.assertRaises(AppError):
            imaging.analyze_path(Path("Z:/不存在/图.png"))

    def test_tiny_image_rejected(self):
        for size in ((3, 3), (10, 200)):
            with self.subTest(size=size):
                with self.assertRaises(AppError):
                    imaging.analyze(Image.new("RGB", size, (0, 0, 0)))


# ---------------------------------------------------------------------------
class TestIdeas(unittest.TestCase):
    def test_deterministic(self):
        a = ideas.generate_idea("vase", 2026, res=60)
        b = ideas.generate_idea("vase", 2026, res=60)
        self.assertEqual(a.title, b.title)
        self.assertEqual(a.story, b.story)

    def test_seed_changes_content(self):
        self.assertNotEqual(ideas.generate_idea("vase", 1, res=60).title,
                            ideas.generate_idea("vase", 2, res=60).title)

    def test_all_fields_filled(self):
        for fid in ALL_FORMS:
            idea = ideas.generate_idea(fid, 5, res=40)
            with self.subTest(form=fid):
                for field in ("title", "story", "material", "usage", "tip"):
                    self.assertTrue(getattr(idea, field).strip(), f"{fid}.{field} 为空")
                self.assertTrue(idea.tags)

    def test_tags_are_short(self):
        for tag in ideas.generate_idea("knot", 9, res=40).tags:
            self.assertLessEqual(len(tag), 8)

    def test_card_text_contains_title(self):
        idea = ideas.generate_idea("vase", 3, res=40)
        text = ideas.idea_card_text(idea, form_id="vase", seed=3, res=40,
                                    params={"a": 50, "b": 50}, palette_id="frost")
        self.assertIn(idea.title, text)

    def test_to_dict_is_serialisable(self):
        import json

        json.dumps(ideas.generate_idea("vase", 1).to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
class TestRegistry(unittest.TestCase):
    def test_unique_ids(self):
        ids = [f.id for f in registry.FORMS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_form_has_two_params(self):
        for spec in registry.FORMS:
            with self.subTest(form=spec.id):
                self.assertIsNotNone(spec.param("a"))
                self.assertIsNotNone(spec.param("b"))

    def test_every_form_has_presets(self):
        for spec in registry.FORMS:
            with self.subTest(form=spec.id):
                self.assertTrue(spec.presets)

    def test_presets_within_param_range(self):
        for spec in registry.FORMS:
            pa, pb = spec.param("a"), spec.param("b")
            for p in spec.presets:
                with self.subTest(form=spec.id, preset=p.name):
                    self.assertGreaterEqual(p.a, pa.lo)
                    self.assertLessEqual(p.a, pa.hi)
                    self.assertGreaterEqual(p.b, pb.lo)
                    self.assertLessEqual(p.b, pb.hi)

    def test_default_palette_exists(self):
        for spec in registry.FORMS:
            with self.subTest(form=spec.id):
                self.assertIn(spec.default_palette, config.PALETTES)

    def test_param_ranges_valid(self):
        for spec in registry.FORMS:
            for key in ("a", "b"):
                p = spec.param(key)
                with self.subTest(form=spec.id, param=key):
                    self.assertLess(p.lo, p.hi)
                    self.assertGreaterEqual(p.default, p.lo)
                    self.assertLessEqual(p.default, p.hi)

    def test_keyword_index_covers_all_forms(self):
        covered = {fid for _kw, fid in registry.keyword_index()}
        for fid in ALL_FORMS:
            self.assertIn(fid, covered)

    def test_keywords_not_empty(self):
        for spec in registry.FORMS:
            with self.subTest(form=spec.id):
                self.assertTrue(spec.keywords)

    def test_clamp_params(self):
        clamped = registry.clamp_params("vase", {"a": -50, "b": 500})
        pa, pb = registry.get("vase").param("a"), registry.get("vase").param("b")
        self.assertGreaterEqual(clamped["a"], pa.lo)
        self.assertLessEqual(clamped["b"], pb.hi)

    def test_get_unknown_form_falls_back(self):
        self.assertIsNotNone(registry.get("绝对没有"))


# ---------------------------------------------------------------------------
class TestSession(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="c3d_sess_")
        self.settings = Settings(Path(self.tmp) / "s.json")
        self.s = Session(self.settings)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_initial_state_valid(self):
        self.assertIn(self.s.form, ALL_FORMS)
        self.assertIsNotNone(self.s.spec)

    def test_undo_redo(self):
        before = self.s.form
        target = "crystal" if before != "crystal" else "vase"
        self.s.set_form(target)
        self.assertEqual(self.s.form, target)
        self.assertTrue(self.s.undo())
        self.assertEqual(self.s.form, before)
        self.assertTrue(self.s.redo())
        self.assertEqual(self.s.form, target)

    def test_undo_past_beginning_is_safe(self):
        for _ in range(20):
            self.s.undo()
        self.assertIn(self.s.form, ALL_FORMS)

    def test_redo_past_end_is_safe(self):
        for _ in range(20):
            self.s.redo()
        self.assertIn(self.s.form, ALL_FORMS)

    def test_can_undo_redo_flags(self):
        self.assertFalse(self.s.can_undo())
        self.s.set_form("tower")
        self.assertTrue(self.s.can_undo())

    def test_param_clamped(self):
        p = self.s.spec.param("a")
        self.s.set_param("a", -9999)
        self.assertGreaterEqual(self.s.params["a"], p.lo)
        self.s.set_param("a", 9999)
        self.assertLessEqual(self.s.params["a"], p.hi)

    def test_seed_normalised(self):
        self.s.set_seed(-5)
        self.assertGreaterEqual(self.s.seed, 0)
        self.s.set_seed(10 ** 12)
        self.assertLessEqual(self.s.seed, 999999)

    def test_res_clamped(self):
        self.s.set_res(-100)
        self.assertGreaterEqual(self.s.res, 0)
        self.s.set_res(1e6)
        self.assertLessEqual(self.s.res, 100)

    def test_unknown_palette_falls_back(self):
        self.s.set_palette("一定不存在的配色")
        self.assertIn(self.s.palette_id, config.PALETTES)

    def test_unknown_form_is_ignored(self):
        before = self.s.form
        self.s.set_form("不存在")
        self.assertEqual(self.s.form, before)

    def test_mesh_key_changes_with_state(self):
        a = self.s.snapshot().key()
        self.s.set_param("a", 12.0)
        self.assertNotEqual(a, self.s.snapshot().key())

    def test_subscribers_notified(self):
        got: list = []
        self.s.subscribe("state", lambda **kw: got.append(kw))
        self.s.set_form("tower")
        self.assertTrue(got)

    def test_broken_subscriber_does_not_break_session(self):
        self.s.subscribe("state", lambda **kw: 1 / 0)
        self.s.set_form("tower")
        self.assertEqual(self.s.form, "tower")

    def test_generate_mesh_works(self):
        mesh = self.s.generate_mesh()
        self.assertFalse(mesh.is_empty())

    def test_generate_mesh_cached(self):
        a = self.s.generate_mesh()
        b = self.s.generate_mesh()
        self.assertIs(a, b)

    def test_apply_prompt_result(self):
        self.s.apply_prompt_result(nlp.parse("晶簇 种子 42 细节 70"))
        self.assertEqual(self.s.form, "crystal")
        self.assertEqual(self.s.seed, 42)

    def test_randomize_changes_seed(self):
        before = self.s.seed
        for _ in range(6):
            self.s.randomize(form=False)
            if self.s.seed != before:
                break
        self.assertNotEqual(self.s.seed, before)

    def test_palette_object(self):
        self.assertIsNotNone(self.s.palette())
        self.assertTrue(self.s.palette_label())

    def test_export_options_defaults(self):
        self.assertIsInstance(self.s.export_options_defaults(), dict)


# ---------------------------------------------------------------------------
class TestSettings(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            a = Settings(path)
            a.set("form", "tower", save=True)
            a.set("res", 77, save=True)
            b = Settings(path)
            self.assertEqual(b.get("form"), "tower")
            self.assertEqual(b.get("res"), 77)

    def test_corrupted_file_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{这不是合法 JSON", encoding="utf-8")
            s = Settings(path)
            self.assertIsNotNone(s.get("form"))
            self.assertTrue(path.with_suffix(".bad.json").exists())

    def test_unknown_key_returns_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(Path(tmp) / "s.json")
            self.assertEqual(s.get("绝对没有的键", "兜底"), "兜底")

    def test_schema_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text('{"schema": 1, "form": "shell"}', encoding="utf-8")
            s = Settings(path)
            self.assertEqual(s.get("form"), "shell")
            self.assertIn("palette", s.data)

    def test_write_failure_is_silent(self):
        s = Settings(Path("Z:/不存在/s.json"))
        s.set("a", 1)
        self.assertFalse(s.save())

    def test_recent_exports_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(Path(tmp) / "s.json")
            for i in range(50):
                s.add_recent_export({"path": f"p{i}", "name": f"n{i}"})
            self.assertLessEqual(len(s.get("recent_exports")), 12)

    def test_toggle_favorite(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(Path(tmp) / "s.json")
            self.assertTrue(s.toggle_favorite({"key": "k1"}))
            self.assertTrue(s.is_favorite("k1"))
            self.assertFalse(s.toggle_favorite({"key": "k1"}))
            self.assertFalse(s.is_favorite("k1"))


if __name__ == "__main__":
    os.environ.setdefault("C3D_TEST", "1")
    unittest.main(verbosity=2)
