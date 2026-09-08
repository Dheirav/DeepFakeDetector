"""The splitter produced 40/30/30 for every build while every config said 70/15/15.

The greedy cost normalised each split's deviation by its own target, so while a
split was under target the cost equalled 1 - fill_fraction and argmin picked the
split already proportionally fullest. Small splits kept winning until they hit
2x target. Every model in this repository trained on 31,183 images instead of
54,505 as a result, and no audit caught it for the life of the project.
"""
import logging
import unittest

from modules.splitter import assign_clusters_to_splits

LOG = logging.getLogger("test-splitter")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def singletons(n, sources=5):
    classes = ["real", "ai_generated", "ai_edited"]
    return [[{"path": f"p{i}", "class_label": classes[i % 3],
              "dataset_source": f"src{i % sources}"}] for i in range(n)]


class TestSplitRatios(unittest.TestCase):
    def test_ratios_match_the_config(self):
        for ratios in ({"train": 0.7, "val": 0.15, "test": 0.15},
                       {"train": 0.8, "val": 0.1, "test": 0.1},
                       {"train": 0.6, "val": 0.2, "test": 0.2}):
            with self.subTest(ratios=ratios):
                n = 3000
                mapping = assign_clusters_to_splits(singletons(n), ratios, 42, LOG)
                for name, want in ratios.items():
                    got = sum(1 for v in mapping.values() if v == name) / n
                    self.assertLess(abs(got - want), 0.02,
                                    f"{name}: configured {want}, got {got:.3f}")

    def test_no_cluster_spans_two_splits(self):
        clusters = [[{"path": f"c{i}_{j}", "class_label": "real", "dataset_source": "s0"}
                     for j in range(3)] for i in range(200)]
        mapping = assign_clusters_to_splits(
            clusters, {"train": 0.7, "val": 0.15, "test": 0.15}, 7, LOG)
        for cluster in clusters:
            self.assertEqual(len({mapping[r["path"]] for r in cluster}), 1)

    def test_deterministic_for_a_fixed_seed(self):
        clusters = singletons(500)
        r = {"train": 0.7, "val": 0.15, "test": 0.15}
        self.assertEqual(assign_clusters_to_splits(clusters, r, 42, LOG),
                         assign_clusters_to_splits(clusters, r, 42, LOG))


class TestLeaveOneSourceOut(unittest.TestCase):
    def test_held_out_source_appears_only_in_test(self):
        clusters = singletons(2000)
        mapping = assign_clusters_to_splits(
            clusters, {"train": 0.7, "val": 0.15, "test": 0.15}, 42, LOG,
            holdout_sources={"src3"})
        by_split = {}
        for cluster in clusters:
            for row in cluster:
                by_split.setdefault(mapping[row["path"]], set()).add(row["dataset_source"])
        self.assertEqual(by_split["test"], {"src3"})
        self.assertNotIn("src3", by_split["train"])
        self.assertNotIn("src3", by_split.get("val", set()))


if __name__ == "__main__":
    unittest.main()
