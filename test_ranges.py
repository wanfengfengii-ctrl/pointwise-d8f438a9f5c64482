"""Correctness verification against the independent sorting oracle."""

import random
import unittest
from bisect import bisect_left

from ranges import Index, MAX_INT64, MIN_INT64
from reference_sorting import ReferenceIndex


class RangesIndexTests(unittest.TestCase):
    def assert_index_matches_reference(self, values):
        index = Index(values)
        reference = ReferenceIndex(values)
        n = len(values)

        # Exhaustively cover every interval for small inputs.
        if n <= 12:
            interval_starts = range(n + 1)
        else:
            interval_starts = random.sample(range(n + 1), min(n, 32))

        for left in interval_starts:
            if n <= 12:
                rights = range(left, n + 1)
            else:
                rights = {
                    min(n, left + offset)
                    for offset in (1, 2, 5, 17, n // 4, n // 2, n)
                }

            for right in rights:
                if right < left:
                    continue
                length = right - left
                ordered = sorted(values[left:right])

                for k in range(length):
                    self.assertEqual(
                        index.kth(left, right, k),
                        reference.kth(left, right, k),
                        (values, left, right, k),
                    )
                    self.assertEqual(index.kth(left, right, k), ordered[k])

                candidates = [
                    MIN_INT64,
                    MIN_INT64 + 1,
                    MAX_INT64 - 1,
                    MAX_INT64,
                    -10**18,
                    10**18,
                    -3,
                    0,
                    3,
                ]
                candidates.extend(values)
                if ordered:
                    candidates.extend((ordered[0] - 1, ordered[-1] + 1))

                for x in candidates:
                    expected = bisect_left(ordered, x)
                    self.assertEqual(index.rank(left, right, x), expected)
                    self.assertEqual(
                        index.rank(left, right, x),
                        reference.rank(left, right, x),
                    )

    def test_specification_example(self):
        index = Index([4, -2, 4, 7])
        self.assertEqual(index.kth(0, 3, 1), 4)
        self.assertEqual(index.rank(1, 4, 4), 1)

    def test_empty_index(self):
        index = Index([])
        self.assertEqual(index.n, 0)
        self.assertEqual(index.sigma, 0)
        self.assertEqual(index.alphabet, ())
        self.assertEqual(index.rank(0, 0, 0), 0)
        self.assertEqual(index.rank(0, 0, MIN_INT64), 0)
        with self.assertRaises(ValueError):
            index.kth(0, 0, 0)
        with self.assertRaises(ValueError):
            index.kth(0, 0, -1)

    def test_one_value_and_repeated_values(self):
        for values in (
            [42],
            [42, 42],
            [-7, -7, -7],
            [MIN_INT64, MIN_INT64, MAX_INT64],
            [0, 0, 0, 0, 0],
        ):
            self.assert_index_matches_reference(values)

    def test_signed_and_64_bit_boundaries(self):
        values = [
            MAX_INT64,
            MIN_INT64,
            0,
            -1,
            1,
            MIN_INT64,
            MAX_INT64,
            42,
            -42,
        ]
        self.assert_index_matches_reference(values)

    def test_block_boundary_widths(self):
        for n in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025):
            values = [((i * 37) % 101) - 50 for i in range(n)]
            self.assert_index_matches_reference(values)

    def test_permutations_with_distinct_values(self):
        for n in (1, 2, 3, 7, 16, 17, 31, 32):
            values = list(range(-n // 2, n - n // 2))
            random.Random(1000 + n).shuffle(values)
            self.assert_index_matches_reference(values)

    def test_random_cases(self):
        rng = random.Random(0xBADC0FFEE)
        for case in range(30):
            n = rng.randrange(0, 250)
            distinct = rng.randrange(1, min(n + 1, 12) + 1)
            alphabet = rng.sample(range(-100, 101), distinct)
            values = [rng.choice(alphabet) for _ in range(n)]
            self.assert_index_matches_reference(values)

        for case in range(10):
            n = rng.randrange(1, 250)
            values = [rng.randint(MIN_INT64, MAX_INT64) for _ in range(n)]
            self.assert_index_matches_reference(values)

    def test_original_list_mutation_does_not_affect_snapshot(self):
        values = [5, 1, 5, -9, 2]
        index = Index(values)
        values[0] = -100
        values.append(999)
        values.clear()
        self.assertEqual(index.n, 5)
        self.assertEqual(index.kth(0, 5, 0), -9)
        self.assertEqual(index.rank(0, 5, 5), 3)
        self.assertEqual(index.alphabet, (-9, 1, 2, 5))

    def test_invalid_ranges_and_k(self):
        index = Index([1, 2, 3])
        invalid_ranges = [
            (-1, 3),
            (0, 4),
            (2, 1),
            (-1, -1),
            (4, 4),
        ]
        for left, right in invalid_ranges:
            with self.assertRaises(ValueError):
                index.kth(left, right, 0)
            with self.assertRaises(ValueError):
                index.rank(left, right, 0)

        for bad_k in (-1, 3, 4):
            with self.assertRaises(ValueError):
                index.kth(0, 3, bad_k)
        with self.assertRaises(ValueError):
            index.kth(1, 2, 1)

        # Empty intervals accept rank but never accept kth.
        self.assertEqual(index.rank(1, 1, 100), 0)
        with self.assertRaises(ValueError):
            index.kth(1, 1, 0)

    def test_invalid_input_types_and_values(self):
        with self.assertRaises(TypeError):
            Index([1, True])
        with self.assertRaises(TypeError):
            Index([1, 2.0])
        with self.assertRaises(TypeError):
            Index((1, 2, 3))
        with self.assertRaises(ValueError):
            Index([MIN_INT64 - 1])
        with self.assertRaises(ValueError):
            Index([MAX_INT64 + 1])

        index = Index([1, 2])
        with self.assertRaises(TypeError):
            index.kth(0, 2, 0.5)
        with self.assertRaises(TypeError):
            index.rank(0, 2, None)

        # Any exact integer threshold is valid, including values outside the
        # stored signed 64-bit domain; non-integers are not.
        self.assertEqual(index.rank(0, 2, MAX_INT64 + 1), 2)
        self.assertEqual(index.rank(0, 2, MIN_INT64 - 1), 0)

    def test_dense_rank_mapping(self):
        index = Index([4, -2, 4, 7])
        self.assertEqual(index.dense_rank(-2), 0)
        self.assertEqual(index.dense_rank(4), 1)
        self.assertEqual(index.dense_rank(7), 2)
        with self.assertRaises(ValueError):
            index.dense_rank(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
