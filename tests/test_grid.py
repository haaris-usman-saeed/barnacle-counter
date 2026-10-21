import unittest

import numpy as np

import tests.helpers  # noqa: F401  (path setup)
from barnacles.frame import Frame
from barnacles.grid import GridSpec, assign, build_cells, verify, boundary_selftest, map_centers


class TestAssignment(unittest.TestCase):
    def test_boundaries(self):
        spec = GridSpec(4, 5)
        row, col, inside = assign([[0, 0], [1, 1], [0.2, 0.25], [0.999999, 0], [1, 0.5]], spec)
        self.assertTrue(inside.all())
        self.assertEqual(list(col), [0, 4, 1, 4, 4])
        self.assertEqual(list(row), [0, 3, 1, 0, 2])

    def test_outside(self):
        _, _, inside = assign([[-1e-12, 0.5], [0.5, 1 + 1e-12], [2, 2]], GridSpec(3, 3))
        self.assertFalse(inside.any())

    def test_selftest_many_grids(self):
        for r, c in [(1, 1), (2, 3), (7, 13), (10, 10), (60, 60)]:
            self.assertTrue(boundary_selftest(GridSpec(r, c))["ok"])

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            GridSpec(0, 3)


class TestReconciliation(unittest.TestCase):
    def test_sum_equals_inside_for_random_points(self):
        f = Frame([[100, 100], [900, 120], [880, 900], [120, 880]], (1000, 1000), "manual")
        rng = np.random.default_rng(0)
        recs = [{"id": f"p{i}", "x": float(x), "y": float(y), "counted": True, "flags": []}
                for i, (x, y) in enumerate(rng.uniform(50, 950, (3000, 2)))]
        for r, c in [(1, 1), (3, 4), (10, 10), (17, 9)]:
            spec = GridSpec(r, c)
            mapped = map_centers(recs, f)
            cells = build_cells(f, spec, mapped)
            chk = verify(f, spec, mapped, cells)
            self.assertTrue(chk["all_passed"])
            self.assertEqual(chk["sum_of_cell_counts"] + chk["n_outside_frame"], len(recs))


if __name__ == "__main__":
    unittest.main()
