"""Runnable demonstration of the wavelet-matrix range index."""

from ranges import Index
from reference_sorting import ReferenceIndex


def main():
    values = [4, -2, 4, 7, 1, -2, 7, 7, 0, 4]
    index = Index(values)
    oracle = ReferenceIndex(values)

    print("values:", values)
    print("distinct alphabet:", index.alphabet)
    print("dense rank map:", {value: index.dense_rank(value) for value in index.alphabet})
    print()

    examples = [
        (0, 3, 1, 4),
        (1, 4, 2, None),
        (2, 8, 5, None),
        (0, 10, 0, None),
        (3, 7, 2, None),
    ]

    for left, right, k, _ in examples:
        got = index.kth(left, right, k)
        expected = oracle.kth(left, right, k)
        print(
            f"kth({left}, {right}, {k}) = {got:<5} "
            f"reference sorted({values[left:right]}) = {sorted(values[left:right])} "
            f"[{ 'ok' if got == expected else 'FAIL'}]"
        )

    print()
    for left, right, xs in [
        (1, 4, (4, 5, -2)),
        (0, 10, (4, 5, -100)),
        (4, 8, (7, 8)),
        (0, 0, (0,)),
    ]:
        for x in xs:
            got = index.rank(left, right, x)
            expected = oracle.rank(left, right, x)
            print(
                f"rank({left}, {right}, {x:>4}) = {got:<3} "
                f"reference={expected:<3} [{'ok' if got == expected else 'FAIL'}]"
            )

    print()
    report = index.space_report()
    print("space report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print()
    print("snapshot check: mutating the original list cannot change query results")
    original = [3, 1, 3]
    snapshot_index = Index(original)
    original.append(-999)
    original[0] = 999
    print("  after mutation, kth(0, 3, 0) =", snapshot_index.kth(0, 3, 0))
    print("  after mutation, rank(0, 3, 3) =", snapshot_index.rank(0, 3, 3))


if __name__ == "__main__":
    main()
