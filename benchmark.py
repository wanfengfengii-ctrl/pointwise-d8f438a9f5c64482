"""Scaling measurements and pointwise runtime manifest generation.

Run:

    python3 benchmark.py

The script writes ``.pointwise-runtime.json`` in the current directory.  It also
checks every measured case against the independent sorting reference.
"""

import json
import platform
import random
import statistics
import time
from array import array

from ranges import Index, MAX_INT64, MIN_INT64
from reference_sorting import ReferenceIndex


SIZES = [1_000, 10_000, 100_000, 1_000_000]
QUERY_COUNT = 10_000
REPEATS = 3
REFERENCE_CHECKS = 64
OUTPUT_PATH = ".pointwise-runtime.json"


def make_values(n, distribution, rng):
    if distribution == "high-cardinality":
        return [rng.randint(MIN_INT64, MAX_INT64) for _ in range(n)]

    # Low cardinality gives a short wavelet matrix; include signed values and
    # many duplicates.
    alphabet = [-10**18, -42, -1, 0, 1, 42, 10**15, MAX_INT64]
    return [rng.choice(alphabet) for _ in range(n)]


def make_queries(n, rng):
    queries = []
    for _ in range(QUERY_COUNT):
        left = rng.randrange(n + 1)
        right = rng.randrange(left, n + 1)
        if rng.random() < 0.5:
            if left == right:
                # Rank supports empty intervals.
                x = rng.choice([MIN_INT64, -1, 0, 1, MAX_INT64])
                queries.append(("rank", left, right, x))
            else:
                k = rng.randrange(right - left)
                queries.append(("kth", left, right, k))
        else:
            x = rng.choice([MIN_INT64, -10**18, -1, 0, 1, 42, MAX_INT64])
            queries.append(("rank", left, right, x))
    return queries


def run_queries(index, queries):
    total = 0
    for kind, left, right, argument in queries:
        if kind == "kth":
            total ^= index.kth(left, right, argument)
        else:
            total += index.rank(left, right, argument)
    return total


def verify_against_reference(values, index, rng):
    reference = ReferenceIndex(values)
    n = len(values)

    for _ in range(REFERENCE_CHECKS):
        left = rng.randrange(n + 1)
        if n > 500:
            # Keep the deliberately unsophisticated sorting oracle affordable.
            width = rng.randrange(1, min(501, n - left + 1))
        else:
            width = rng.randrange(1, n - left + 1)
        right = left + width
        k = rng.randrange(width)

        assert index.kth(left, right, k) == reference.kth(left, right, k)

        x = rng.choice(
            [
                MIN_INT64,
                values[rng.randrange(n)],
                values[rng.randrange(n)] - 1,
                values[rng.randrange(n)] + 1,
                0,
                MAX_INT64,
            ]
        )
        assert index.rank(left, right, x) == reference.rank(left, right, x)

    # Also verify the empty-interval contract.
    for pos in (0, n // 2, n):
        assert index.rank(pos, pos, 0) == 0
        try:
            index.kth(pos, pos, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("kth on an empty interval must raise ValueError")


def measure_case(n, distribution, seed):
    rng = random.Random(seed)
    values = make_values(n, distribution, rng)
    queries = make_queries(n, rng)

    build_start = time.perf_counter()
    index = Index(values)
    build_seconds = time.perf_counter() - build_start

    verify_against_reference(values, index, random.Random(seed + 1))

    # Warm allocator/branch behavior once, then collect pointwise timings.
    run_queries(index, queries)
    query_seconds = []
    checksum = 0
    for _ in range(REPEATS):
        start = time.perf_counter()
        checksum = run_queries(index, queries)
        query_seconds.append(time.perf_counter() - start)

    report = index.space_report()
    return {
        "n": n,
        "distribution": distribution,
        "sigma": index.sigma,
        "levels": index.space_report()["levels"],
        "query_count_per_repeat": QUERY_COUNT,
        "repeats": REPEATS,
        "build_seconds": build_seconds,
        "query_seconds_pointwise": query_seconds,
        "minimum_query_batch_seconds": min(query_seconds),
        "median_query_batch_seconds": statistics.median(query_seconds),
        "minimum_microseconds_per_query": min(query_seconds) / QUERY_COUNT * 1e6,
        "reference_checks": REFERENCE_CHECKS,
        "checksum": checksum,
        "space_report": report,
    }


def main():
    cases = []
    seed = 20261114
    for n in SIZES:
        for distribution in ("high-cardinality", "low-cardinality"):
            print(f"measuring n={n:>8,} {distribution} ...", flush=True)
            cases.append(measure_case(n, distribution, seed))
            seed += 1009

    manifest = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timing_clock": "time.perf_counter",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "array_Q_item_bytes": array("Q").itemsize,
        "method": {
            "index": "ranges.Index, packed wavelet matrix with fixed-block rank directory",
            "reference": "reference_sorting.ReferenceIndex, whole-slice sorted() and bisect_left",
            "note": (
                "Query timing does not scan the requested interval or copy/sort it; "
                "sorting is used only by the independent correctness oracle."
            ),
        },
        "cases": cases,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")

    print()
    print(f"wrote {OUTPUT_PATH}")
    for case in cases:
        print(
            f"n={case['n']:>8,} {case['distribution']:<17} "
            f"sigma={case['sigma']:>8,} levels={case['levels']:>2} "
            f"build={case['build_seconds']:>7.4f}s "
            f"query={case['minimum_microseconds_per_query']:>9.3f} us/op"
        )


if __name__ == "__main__":
    main()
