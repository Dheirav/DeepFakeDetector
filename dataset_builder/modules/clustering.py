"""Near-duplicate clustering over perceptual hashes.

Shared by ``deduplicator`` and ``splitter``, which previously carried two copies
of the same algorithm and drifted apart.

**Why the old scheme found nothing.** Both modules bucketed on the first
``phash_bucket_chars`` (12) hex characters of a 16-hex-char (64-bit) pHash and
then union-found at Hamming <= 8 *within a bucket*. Requiring 48 of 64 bits to
match exactly before the distance test can even run means every differing bit
must land in the trailing 16. Measured recall on genuine near-duplicates:

    Hamming d   P(same bucket)      measured recall
        1         2.50e-01              25.0%
        2         5.95e-02               5.96%
        4         2.86e-03               0.27%
        8         2.91e-06               0.0005%

On the project's own stored hashes it caught **0 of 1,287** genuine pairs. What
actually ran was exact-pHash matching; the Hamming threshold was decorative. The
consequence was 27,712 near-duplicate pairs spanning splits, 9,871 of them across
train<->test, with every audit reporting zero leakage because md5 differed.

**What this does instead.** Multi-index hashing: split the 64 bits into ``bands``
contiguous segments and treat two hashes as candidates if *any* segment matches
exactly, then verify true Hamming distance. Two hashes at distance ``d`` differ
in at most ``d`` segments, so with 8 bands any pair at ``d <= 7`` is guaranteed to
share a segment and be found. At ``d = 8`` the only miss is the pathological case
of exactly one differing bit in each of the 8 bands.
"""

from collections import defaultdict

__all__ = ["hamming_distance", "build_clusters_from_hashes", "DEFAULT_BANDS"]

DEFAULT_BANDS = 8
_HASH_BITS = 64


def hamming_distance(phash1, phash2):
    """Bit distance between two hex pHash strings; ``None`` if either is unusable.

    Returns ``None`` rather than the old sentinel 999, so a malformed hash is
    distinguishable from a genuinely distant one instead of silently becoming
    "maximally far apart".
    """
    try:
        return bin(int(phash1, 16) ^ int(phash2, 16)).count("1")
    except (TypeError, ValueError):
        return None


def _band_keys(value, bands, bits):
    width = bits // bands
    mask = (1 << width) - 1
    return [(i, (value >> (i * width)) & mask) for i in range(bands)]


def build_clusters_from_hashes(items, threshold, bands=DEFAULT_BANDS, logger=None):
    """Cluster ``items`` -- ``(key, phash_hex)`` pairs -- by pHash proximity.

    Returns ``{key: cluster_id}``. Items with a missing or malformed hash become
    their own singleton cluster.
    """
    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    values, malformed = {}, 0
    for key, phash in items:
        parent[key] = key
        try:
            if phash is None or len(str(phash).strip()) == 0:
                raise ValueError
            values[key] = int(str(phash).strip(), 16)
        except (TypeError, ValueError):
            malformed += 1

    buckets = defaultdict(list)
    for key, value in values.items():
        for band_key in _band_keys(value, bands, _HASH_BITS):
            buckets[band_key].append(key)

    comparisons = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            vi = values[members[i]]
            for j in range(i + 1, len(members)):
                comparisons += 1
                if bin(vi ^ values[members[j]]).count("1") <= threshold:
                    union(members[i], members[j])

    if logger is not None:
        n_clusters = len({find(k) for k in parent})
        logger.info(
            f"Clustered {len(parent)} items into {n_clusters} clusters "
            f"({bands} bands, threshold {threshold}, {comparisons} distance checks, "
            f"{malformed} malformed hashes treated as singletons)"
        )
    return {key: find(key) for key in parent}
