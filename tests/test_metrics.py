import numpy as np
from macs.metrics.represent import normalise, ast_features, FEATURE_NAMES
from macs.metrics.novelty import ReferenceStats
from macs.metrics.diversity import cluster_families, family_entropy, diversity_report, validate_against_labels
from macs.metrics.usefulness import pass_at_k

BRUTE = "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n"
HASH = "def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n"
HASH_RENAMED = "def two_sum(a, t):\n    # find complement\n    d = {}\n    for k, v in enumerate(a):\n        if t - v in d:\n            return [d[t - v], k]\n        d[v] = k\n"

def test_normalise_erases_renaming():
    assert normalise(HASH) == normalise(HASH_RENAMED)

def test_ast_features_shape_and_semantics():
    f = ast_features(HASH)
    assert f.shape == (len(FEATURE_NAMES),)
    assert f[FEATURE_NAMES.index("uses_dict")] == 1
    assert ast_features(BRUTE)[FEATURE_NAMES.index("max_loop_depth")] == 2

def test_novelty_zero_mean_on_reference():
    rng = np.random.default_rng(0)
    X = np.hstack([rng.normal(size=(50, 3)), np.zeros((50, 5))])       # reference lives in dims 0-2
    rs = ReferenceStats.fit(X, k=5)
    z = rs.novelty(np.hstack([rng.normal(size=(50, 3)), np.zeros((50, 5))]))
    assert abs(z.mean()) < 0.75                                         # same distribution -> typical
    far = rs.novelty(np.hstack([np.zeros((5, 3)), rng.normal(size=(5, 5))]))  # orthogonal subspace
    assert far.min() > z.max()                                          # every far point beats every typical one

def test_families_and_entropy():
    X = np.array([[1, 0], [1, 0.01], [0, 1], [0, 1.01], [0.7, 0.7]])
    labels = cluster_families(X, distance_threshold=0.2)
    assert len(set(labels)) == 3
    assert family_entropy(np.array([0, 0, 0, 0])) == 0
    assert validate_against_labels(labels, ["a", "a", "b", "b", "c"]) == 1.0

def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9
    assert 0.3 < pass_at_k(10, 3, 5) < 1.0
