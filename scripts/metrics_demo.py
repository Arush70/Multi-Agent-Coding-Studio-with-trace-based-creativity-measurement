"""See the creativity metrics behave, each on a clean example.

Arush - run this to watch the four pieces work before any big experiment:

    python scripts/metrics_demo.py

Part 1 shows the representation ignores renaming (only a real change of approach counts).
Part 2 shows novelty: a typical point scores ~0, an outlier scores high.
Part 3 shows diversity: three different strategies form three families, matching hand labels.
Part 4 shows correctness x novelty combined into the headline 'correct-and-novel' rate.
"""
import numpy as np
from macs.metrics.represent import normalise, ast_features, FEATURE_NAMES
from macs.metrics.novelty import ReferenceStats
from macs.metrics.diversity import cluster_families, diversity_report, validate_against_labels
from macs.metrics.usefulness import pass_at_k, correct_and_novel_rate

HASH         = "def two_sum(nums, t):\n    seen = {}\n    for i, x in enumerate(nums):\n        if t - x in seen:\n            return [seen[t - x], i]\n        seen[x] = i\n"
HASH_RENAMED = "def two_sum(a, tgt):\n    # same idea, different names\n    d = {}\n    for k, v in enumerate(a):\n        if tgt - v in d:\n            return [d[tgt - v], k]\n        d[v] = k\n"
BRUTE        = "def two_sum(nums, t):\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):\n            if nums[i] + nums[j] == t:\n                return [i, j]\n"

def main():
    print("PART 1 — representation ignores cosmetic change")
    same = normalise(HASH) == normalise(HASH_RENAMED)
    print(f"  hash vs hash-with-renamed-vars normalise to the same code?  {same}")
    print(f"  brute uses nested loops (loop depth) = {int(ast_features(BRUTE)[FEATURE_NAMES.index('max_loop_depth')])}, "
          f"hash uses a dict = {int(ast_features(HASH)[FEATURE_NAMES.index('uses_dict')])}")

    print("\nPART 2 — novelty (z-score: ~0 typical, high = unusual)")
    # novelty uses cosine distance, so 'unusual' means pointing in a new DIRECTION.
    # Reference solutions live in one subspace; the outlier points in an orthogonal one.
    rng = np.random.default_rng(0)
    X_ref = np.hstack([rng.normal(size=(100, 3)), np.zeros((100, 5))])   # frozen reference cloud
    rs = ReferenceStats.fit(X_ref, k=5)
    typical = rs.novelty(np.hstack([rng.normal(size=(1, 3)), np.zeros((1, 5))]))  # same subspace
    outlier = rs.novelty(np.array([[0, 0, 0, 1, 1, 1, 1, 1.0]]))                   # new direction
    print(f"  a typical solution:  z = {typical[0]:+.2f}")
    print(f"  an unusual solution: z = {outlier[0]:+.2f}")

    print("\nPART 3 — diversity (distinct strategies -> distinct families)")
    X = np.array([[1,0],[1,0.02],[0,1],[0,1.02],[0.7,0.7],[0.71,0.69]])  # 3 clusters of 2
    labels = cluster_families(X, distance_threshold=0.2)
    ari = validate_against_labels(labels, ["a","a","b","b","c","c"])
    rep = diversity_report(X, distance_threshold=0.2, first_n=6)
    print(f"  families found = {rep.n_families_total} (expected 3), entropy = {rep.entropy:.2f}")
    print(f"  agreement with hand labels (ARI) = {ari:.2f}  (1.0 = perfect)")

    print("\nPART 4 — correctness x novelty")
    print(f"  pass@1 with 3 of 10 samples correct = {pass_at_k(10,3,1):.2f}")
    print(f"  pass@5 with 3 of 10 samples correct = {pass_at_k(10,3,5):.2f}")
    is_correct = np.array([True, True, False, True])
    is_novel   = np.array([True, False, True, True])
    print(f"  correct-and-novel rate = {correct_and_novel_rate(is_correct, is_novel):.0%} "
          f"(both correct AND novel; an original wrong answer scores nothing)")

if __name__ == "__main__":
    main()
