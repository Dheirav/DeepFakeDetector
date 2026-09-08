"""Near-duplicate detection had ~0% recall and nobody measured it.

Both the deduplicator and the splitter bucketed on the first 12 hex characters
of a 64-bit pHash before testing Hamming distance, which requires 48 of 64 bits
to match exactly. On the project's own stored hashes it caught 0 of 1,287
genuine pairs -- what actually ran was exact-pHash matching, and the threshold
was decorative. 9,871 near-duplicate pairs ended up spanning train/test while
every audit reported zero leakage.
"""
import random
import unittest

from modules.clustering import build_clusters_from_hashes, hamming_distance


def planted_pairs(n, distance, seed=0):
    rng = random.Random(seed)
    items, truth = [], []
    for i in range(n):
        base = rng.getrandbits(64)
        flip = 0
        for bit in rng.sample(range(64), distance):
            flip |= 1 << bit
        items += [(f"a{i}", f"{base:016x}"), (f"b{i}", f"{base ^ flip:016x}")]
        truth.append((f"a{i}", f"b{i}"))
    return items, truth


class TestRecall(unittest.TestCase):
    def test_total_recall_up_to_the_guaranteed_distance(self):
        """8 bands of 8 bits guarantee a shared band for any pair at d <= 7."""
        for d in range(1, 8):
            with self.subTest(hamming=d):
                items, truth = planted_pairs(300, d)
                a = build_clusters_from_hashes(items, threshold=8)
                found = sum(1 for x, y in truth if a[x] == a[y])
                self.assertEqual(found, len(truth), f"d={d}: {found}/{len(truth)}")

    def test_distant_pairs_are_not_merged(self):
        items, truth = planted_pairs(300, distance=20)
        a = build_clusters_from_hashes(items, threshold=8)
        merged = sum(1 for x, y in truth if a[x] == a[y])
        self.assertEqual(merged, 0, f"{merged} pairs at Hamming 20 wrongly clustered")

    def test_identical_hashes_cluster(self):
        a = build_clusters_from_hashes(
            [("x", "aaaaaaaaaaaaaaaa"), ("y", "aaaaaaaaaaaaaaaa")], threshold=0)
        self.assertEqual(a["x"], a["y"])


class TestMalformedInput(unittest.TestCase):
    def test_bad_hashes_become_singletons(self):
        """The old hamming_distance returned 999 on error, silently making a
        malformed hash 'maximally distant' instead of surfacing the problem."""
        items = [("good", "0123456789abcdef"), ("empty", ""),
                 ("junk", "zzzz"), ("none", None)]
        a = build_clusters_from_hashes(items, threshold=8)
        self.assertEqual(len(set(a.values())), 4)
        self.assertIsNone(hamming_distance("zzzz", "0123456789abcdef"))


if __name__ == "__main__":
    unittest.main()
