"""tests/test_sdf.py —— v3 新引擎回归测试：SDF 内核 / 网格化 / 概念库 / 打印体检 / 导出。

运行：  python tests/test_sdf.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from creative3d import (  # noqa: E402
    concepts,
    config,
    exporter,
    ideas,
    nlp,
    printcheck,
    quality,
    sdf as S,
    sdfmesh,
    session as session_mod,
)
from creative3d.settings import Settings  # noqa: E402


def pts(*rows) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
class TestSdfPrimitives(unittest.TestCase):
    def test_sphere_distance(self):
        fn = S.sphere(1.0)
        d = fn(pts([0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0.5]))
        np.testing.assert_allclose(d, [-1.0, 0.0, 1.0, -0.5], atol=1e-6)

    def test_box_distance(self):
        fn = S.box(1.0, 1.0, 1.0)
        d = fn(pts([0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 2, 2]))
        self.assertAlmostEqual(d[0], -1.0)
        self.assertAlmostEqual(d[1], 0.0)
        self.assertAlmostEqual(d[2], 1.0)
        self.assertAlmostEqual(d[3], math.sqrt(3.0))

    def test_cylinder_inside_outside(self):
        fn = S.cyl(1.0, 2.0)
        d = fn(pts([0, 0, 0], [1, 0, 0], [0, 0, 1.0], [0, 0, 2.0]))
        self.assertAlmostEqual(d[0], -1.0)
        self.assertAlmostEqual(d[1], 0.0)
        self.assertAlmostEqual(d[2], 0.0)
        self.assertAlmostEqual(d[3], 1.0)

    def test_cone_and_torus_sign(self):
        cone = S.cone(1.0, 0.0, 2.0)
        self.assertLess(cone(pts([0, 0, -0.5]))[0], 0.0)
        self.assertGreater(cone(pts([0, 0, 1.5]))[0], 0.0)
        torus = S.torus(1.0, 0.25)
        self.assertLess(torus(pts([1.0, 0, 0]))[0], 0.0)
        self.assertGreater(torus(pts([0, 0, 0]))[0], 0.0)

    def test_lathe_matches_cylinder(self):
        """把剖面做成等半径，lathe 应当等价于圆柱。"""
        fn = S.lathe([(0.0, 1.0), (2.0, 1.0)])
        p = pts([0.5, 0, 1.0], [1.5, 0, 1.0], [0.5, 0, -0.4], [0.5, 0, 2.4])
        d = fn(p)
        self.assertLess(d[0], 0.0)
        self.assertGreater(d[1], 0.0)
        self.assertGreater(d[2], 0.0)
        self.assertGreater(d[3], 0.0)

    def test_hollow_lathe_has_cavity(self):
        fn = S.hollow_lathe([(0.0, 1.0), (2.0, 1.0)], 0.2)
        d = fn(pts([0.0, 0, 1.0], [0.9, 0, 1.0], [1.1, 0, 1.0]))
        self.assertGreater(d[0], 0.0, "中心应当被掏空")
        self.assertLess(d[1], 0.0, "壁内应当是实体")
        self.assertGreater(d[2], 0.0, "壁外应当在外部")

    def test_union_and_sub(self):
        a = S.sphere(1.0)
        b = S.at(S.sphere(1.0), 1.5, 0, 0)
        u = S.union(a, b)
        self.assertLess(u(pts([0.75, 0, 0]))[0], 0.0)
        sub = S.sub(a, S.at(S.sphere(0.5), 1.0, 0, 0))
        self.assertGreater(sub(pts([1.0, 0, 0]))[0], -0.2)

    def test_transform_places_shape(self):
        fn = S.at(S.sphere(0.5), 2.0, 0, 0)
        self.assertLess(fn(pts([2.0, 0, 0]))[0], 0.0)
        self.assertGreater(fn(pts([0.0, 0, 0]))[0], 1.0)

    def test_rotated_box_orientation(self):
        bar = S.box(1.0, 0.1, 0.1)
        self.assertLess(bar(pts([0.9, 0, 0]))[0], 0.0)
        upright = S.rotz(bar, math.pi * 0.5)
        self.assertLess(upright(pts([0, 0.9, 0]))[0], 0.0)
        self.assertGreater(upright(pts([0.9, 0, 0]))[0], 0.0)

    def test_displace_changes_field(self):
        base = S.sphere(1.0)
        noisy = S.displace(base, 0.1, freq=4.0, seed=3, octaves=2)
        p = pts([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])
        self.assertFalse(np.allclose(base(p), noisy(p)))

    def test_extrude_and_poly2(self):
        sq = S.extrude(S.box2(0.5, 0.5), 0.2)
        self.assertLess(sq(pts([0, 0, 0]))[0], 0.0)
        self.assertGreater(sq(pts([0, 0, 0.5]))[0], 0.0)
        poly = S.poly2([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
        self.assertLess(poly(np.array([[0.0, 0.0]]))[0], 0.0)
        self.assertGreater(poly(np.array([[1.0, 0.0]]))[0], 0.0)


# ---------------------------------------------------------------------------
class TestPolygonize(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sphere = sdfmesh.polygonize(S.sphere(1.0), 64, refine=2, smooth=1)

    def test_watertight(self):
        ok, boundary, nonmanifold = self.sphere.check_watertight()
        self.assertTrue(ok, f"边界 {boundary} 非流形 {nonmanifold}")

    def test_volume_accuracy(self):
        vol = self.sphere.signed_volume()
        self.assertAlmostEqual(vol / (4.0 / 3.0 * math.pi), 1.0, delta=0.02)

    def test_normals_outward(self):
        m = self.sphere
        centre = m.verts.mean(axis=0)
        cen = m.verts[m.faces].mean(axis=1)
        out = np.einsum("ij,ij->i", m.face_normals(), cen - centre)
        self.assertGreater((out > 0).mean(), 0.999)

    def test_radius_error_small(self):
        r = np.linalg.norm(self.sphere.verts, axis=1)
        self.assertLess(float(np.abs(r - 1.0).max()), 0.02)

    def test_resolution_increases_faces(self):
        lo = sdfmesh.polygonize(S.sphere(1.0), 32, refine=1, smooth=0).n_faces
        hi = sdfmesh.polygonize(S.sphere(1.0), 72, refine=1, smooth=0).n_faces
        self.assertGreater(hi, lo * 2)

    def test_deterministic(self):
        a = sdfmesh.polygonize(S.at(S.sphere(0.7), 0.3, 0, 0), 48)
        b = sdfmesh.polygonize(S.at(S.sphere(0.7), 0.3, 0, 0), 48)
        np.testing.assert_array_equal(a.verts, b.verts)
        np.testing.assert_array_equal(a.faces, b.faces)

    def test_empty_field_raises(self):
        from creative3d.errors import AppError

        with self.assertRaises(AppError):
            sdfmesh.polygonize(lambda p: np.full(p.shape[0], 1.0), 32)

    def test_auto_bounds_scales(self):
        """任意局部尺度的形体都能被自动框住。"""
        for scale in (0.02, 1.0, 40.0):
            m = sdfmesh.polygonize(S.sphere(scale), 40)
            size = m.size()
            self.assertAlmostEqual(float(size.max()) / (2 * scale), 1.0, delta=0.1)


# ---------------------------------------------------------------------------
class TestConcepts(unittest.TestCase):
    def test_registry_consistent(self):
        ids = [c.id for c in concepts.CONCEPTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 50)
        for c in concepts.CONCEPTS:
            with self.subTest(concept=c.id):
                self.assertIn(c.category, concepts.CATEGORIES)
                self.assertTrue(c.keywords)
                self.assertEqual(len(c.pa), 2)
                self.assertEqual(len(c.pb), 2)

    def test_every_concept_builds(self):
        """每个概念都要能产出一个可求值的 SDF（不网格化，保持测试快）。"""
        for c in concepts.CONCEPTS:
            with self.subTest(concept=c.id):
                bp = concepts.make_params(7, salt=c.id)
                fn = concepts.build(c.id, bp)
                p = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
                d = np.asarray(fn(p), dtype=np.float64)
                self.assertEqual(d.shape, (2,))
                self.assertTrue(np.isfinite(d).all(), f"{c.id} 距离场出现 NaN/Inf")

    def test_sample_concepts_are_printable(self):
        """抽查若干概念：封闭、法线朝外、面数合理。"""
        sample = ("vase", "mug", "cat", "gear", "house", "mushroom", "heart", "robot", "tree")
        for cid in sample:
            with self.subTest(concept=cid):
                bp = concepts.make_params(2026, salt=cid)
                m = sdfmesh.mesh_from_sdf(concepts.build(cid, bp), quality.get("standard"))
                ok, boundary, nonmanifold = m.check_watertight()
                self.assertTrue(ok, f"{cid}: 边界 {boundary} 非流形 {nonmanifold}")
                self.assertGreater(m.signed_volume(), 0.0)
                self.assertGreater(m.n_faces, 1000)

    def test_seed_changes_shape(self):
        for cid in ("vase", "cat", "gear"):
            a = sdfmesh.mesh_from_sdf(
                concepts.build(cid, concepts.make_params(1, salt=cid)), quality.get("draft"))
            b = sdfmesh.mesh_from_sdf(
                concepts.build(cid, concepts.make_params(999, salt=cid)), quality.get("draft"))
            with self.subTest(concept=cid):
                same = a.verts.shape == b.verts.shape and np.allclose(a.verts, b.verts)
                self.assertFalse(same, f"{cid} 换种子没变化")

    def test_matching(self):
        cases = {
            "一只猫": "cat",
            "帮我做个齿轮": "gear",
            "城堡": "castle",
            "蘑菇摆件": "mushroom",
            "马克杯": "mug",
            "五角星": "star",
            "a robot": "robot",
        }
        for text, expect in cases.items():
            with self.subTest(text=text):
                self.assertEqual(concepts.match(text)[0], expect)

    def test_unknown_prompt_falls_back(self):
        self.assertIsNone(concepts.match("烫烫烫烫")[0])
        self.assertEqual(concepts.fallback_for("烫烫烫烫"), "abstract")

    def test_category_palette_valid(self):
        for c in concepts.CONCEPTS:
            self.assertIn(concepts.palette_for(c.id), config.PALETTES)


# ---------------------------------------------------------------------------
class TestQuality(unittest.TestCase):
    def test_levels_increase(self):
        grids = [q.grid for q in quality.QUALITIES]
        self.assertEqual(grids, sorted(grids))
        self.assertGreaterEqual(len(grids), 4)

    def test_lookup_and_bounds(self):
        self.assertEqual(quality.get("fine").id, "fine")
        self.assertEqual(quality.get("不存在").id, quality.DEFAULT_QUALITY)
        self.assertEqual(quality.at(-5).id, quality.QUALITIES[0].id)
        self.assertEqual(quality.at(99).id, quality.QUALITIES[-1].id)

    def test_face_estimate_reasonable(self):
        m = sdfmesh.polygonize(S.sphere(1.0), quality.get("standard").grid, refine=1, smooth=1)
        est = quality.get("standard").approx_faces
        self.assertLess(abs(m.n_faces - est) / est, 0.5)


# ---------------------------------------------------------------------------
class TestPrintCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = sdfmesh.polygonize(S.sphere(1.0), 48, refine=1, smooth=1).normalize(80)

    def test_report_fields(self):
        rep = printcheck.analyze(self.mesh)
        self.assertTrue(rep.watertight)
        self.assertAlmostEqual(rep.size_mm[2], 80.0, delta=1.0)
        self.assertGreater(rep.volume_cm3, 0.0)
        self.assertGreater(rep.weight_pla_g, rep.volume_cm3)  # 密度 > 1
        self.assertGreaterEqual(rep.overhang_ratio, 0.0)
        self.assertLessEqual(rep.overhang_ratio, 1.0)
        self.assertIn(rep.grade, ("优", "良", "需注意"))
        self.assertTrue(rep.notes)

    @staticmethod
    def _mushroom():
        """菌柄 + 大伞盖：伞盖下沿是典型的大面积悬垂。"""
        shape = S.union(S.at(S.cyl(0.15, 0.6), 0, 0, 0.3),
                        S.at(S.cyl(0.5, 0.12), 0, 0, 0.66))
        return sdfmesh.polygonize(shape, 48, refine=1, smooth=0).normalize(80)

    def test_overhang_high_for_floating_shape(self):
        """伞盖下方的大平面应当被识别为高悬垂。"""
        rep = printcheck.analyze(self._mushroom())
        self.assertGreater(rep.overhang_ratio, 0.15)

    def test_auto_orient_reduces_overhang(self):
        slab = self._mushroom()
        before = printcheck.analyze(slab).overhang_ratio
        oriented, rep, label = printcheck.auto_orient(slab)
        self.assertLessEqual(rep.overhang_ratio, before + 1e-6)
        self.assertTrue(label)
        self.assertAlmostEqual(float(oriented.bounds()[0][2]), 0.0, places=3)

    def test_report_text(self):
        text = printcheck.report_text(printcheck.analyze(self.mesh), title="测试")
        self.assertIn("打印体检", text)
        self.assertIn("体积", text)


# ---------------------------------------------------------------------------
class TestExportV3(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="c3d_v3_"))
        self.mesh = sdfmesh.polygonize(S.at(S.sphere(0.8), 0, 0, 0.8), 40, refine=1, smooth=1)
        self.mesh.normalize(60)
        self.idea = ideas.generate_idea("vase", 5, concept_id="vase", size_mm=60,
                                        quality_label="标准")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bundle(self, **opts):
        options = exporter.ExportOptions(**opts)
        return exporter.export_bundle(
            self.mesh, form_id="vase", seed=5, res=50, params={"a": 50, "b": 50},
            palette_id="frost", palette=config.PALETTES["frost"], idea=self.idea,
            image_info=None, out_dir=self.tmp, options=options, concept_id="vase",
            size_mm=60.0, quality_label="标准",
        )

    def test_three_mf_is_valid_zip(self):
        result = self._bundle(stl=False, png=False, card=False, three_mf=True)
        self.assertEqual(len(result.files), 1)
        path = result.files[0]
        self.assertTrue(path.name.endswith(".3mf"))
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            self.assertIn("[Content_Types].xml", names)
            self.assertIn("3D/3dmodel.model", names)
            model = zf.read("3D/3dmodel.model").decode("utf-8")
        self.assertIn('unit="millimeter"', model)
        self.assertIn("<triangle", model)
        self.assertIn("<vertex", model)

    def test_report_written(self):
        result = self._bundle(stl=False, png=False, card=False, report=True)
        text = result.files[0].read_text(encoding="utf-8")
        self.assertIn("水密性", text)

    def test_card_contains_concept_and_check(self):
        result = self._bundle(stl=False, png=False, card=True)
        text = result.files[0].read_text(encoding="utf-8")
        self.assertIn("花瓶", text)
        self.assertIn("打印体检", text)

    def test_stl_still_default(self):
        result = self._bundle(stl=True, png=False, card=False)
        self.assertTrue(result.files[0].name.endswith(".stl"))
        self.assertEqual(result.files[0].stat().st_size, 84 + self.mesh.n_faces * 50)


# ---------------------------------------------------------------------------
class TestNlpV3(unittest.TestCase):
    def test_concept_and_size(self):
        r = nlp.parse("一只坐着的猫，高 120mm，打印级")
        self.assertEqual(r.concept_id, "cat")
        self.assertEqual(r.size_mm, 120.0)
        self.assertEqual(r.quality, "print")
        self.assertIsNone(r.seed, "尺寸数字不该被当成种子")

    def test_cm_size(self):
        self.assertEqual(nlp.parse("咖啡杯 8厘米").size_mm, 80.0)
        self.assertEqual(nlp.parse("花瓶 90mm").size_mm, 90.0)

    def test_style_flags(self):
        r = nlp.parse("圆润的猫")
        self.assertGreater(r.style.get("organic", 0), 0)
        self.assertTrue(nlp.parse("中空的杯子").hollow)
        self.assertEqual(nlp.parse("高精度齿轮").quality, "fine")

    def test_random_flag_counts_as_understood(self):
        r = nlp.parse("随便来一个")
        self.assertTrue(r.random_seed)
        self.assertTrue(r.understood)

    def test_help_mentions_models(self):
        self.assertIn("模型", nlp.keywords_help())


# ---------------------------------------------------------------------------
class TestSessionV3(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="c3d_s3_")
        self.settings = Settings(Path(self.tmp) / "s.json")
        self.s = session_mod.Session(self.settings)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_is_concept_mode(self):
        self.assertTrue(self.s.is_concept_mode())
        self.assertIn(self.s.concept, concepts.CONCEPT_MAP)

    def test_concept_switch_resets_craft(self):
        self.s.set_concept("mug")
        self.assertEqual(self.s.concept, "mug")
        self.assertTrue(self.s.hollow)  # 杯子默认中空
        self.assertEqual(self.s.palette_id, concepts.palette_for("mug"))

    def test_generate_concept_mesh(self):
        self.s.set_concept("cat")
        m = self.s.generate_mesh(preview=True)
        self.assertFalse(m.is_empty())
        self.assertGreater(m.n_faces, 100)

    def test_size_normalised(self):
        self.s.set_size(150.0)
        m = self.s.generate_mesh(preview=True)
        self.assertAlmostEqual(float(m.size().max()), 150.0, delta=1.0)

    def test_quality_changes_face_count(self):
        self.s.set_concept("vase")
        self.s.set_quality("draft")
        lo = self.s.generate_mesh(preview=False).n_faces
        self.s.set_quality("standard")
        hi = self.s.generate_mesh(preview=False).n_faces
        self.assertGreater(hi, lo)

    def test_hollow_changes_volume(self):
        self.s.set_concept("mug")
        self.s.set_hollow(False)
        solid = self.s.generate_mesh(preview=True).signed_volume()
        self.s.set_hollow(True)
        hollow = self.s.generate_mesh(preview=True).signed_volume()
        self.assertLess(hollow, solid)

    def test_style_toggle_roundtrip(self):
        self.s.toggle_style("sharp")
        self.assertTrue(self.s.has_style("sharp"))
        self.s.toggle_style("sharp")
        self.assertFalse(self.s.has_style("sharp"))

    def test_prompt_applies_everything(self):
        self.s.apply_prompt_result(nlp.parse("齿轮 高 80mm 打印级 尖锐"))
        self.assertEqual(self.s.concept, "gear")
        self.assertEqual(self.s.size_mm, 80.0)
        self.assertEqual(self.s.quality_id, "print")
        self.assertTrue(self.s.has_style("sharp"))

    def test_undo_restores_concept(self):
        before = self.s.concept
        self.s.set_concept("robot")
        self.assertTrue(self.s.undo())
        self.assertEqual(self.s.concept, before)

    def test_settings_roundtrip(self):
        self.s.set_concept("castle")
        self.s.set_quality("ultra")
        self.s.set_size(180)
        self.s.save()
        again = session_mod.Session(Settings(Path(self.tmp) / "s.json"))
        self.assertEqual(again.concept, "castle")
        self.assertEqual(again.quality_id, "ultra")
        self.assertEqual(again.size_mm, 180.0)

    def test_classic_mode_still_works(self):
        self.s.set_mode("classic")
        self.s.set_form("relief")
        self.assertFalse(self.s.is_concept_mode())
        m = self.s.generate_mesh(preview=True)
        self.assertFalse(m.is_empty())


if __name__ == "__main__":
    os.environ.setdefault("C3D_TEST", "1")
    unittest.main(verbosity=2)
