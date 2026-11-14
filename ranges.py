"""Succinct wavelet matrix for local order-statistics on signed 64-bit integers.

The public class is :class:`Index`.  It stores packed C ``uint64`` arrays rather
than per-level Python object matrices.  Each bit vector has a fixed-block rank
directory, so bit-rank calls examine only a constant number of machine words.
"""

from array import array
from bisect import bisect_left

MIN_INT64 = -(1 << 63)
MAX_INT64 = (1 << 63) - 1

_WORD_BITS = 64
_BLOCK_WORDS = 16               # 16 * 64 == 1024 bits per rank block
_BLOCK_BITS = _WORD_BITS * _BLOCK_WORDS


class _BitVector:
    """A packed bit vector with O(1) rank-one queries.

    ``prefix`` holds an absolute 64-bit one-count before each block.
    ``word_prefix`` holds a relative count for every 64-bit word in its block.
    A rank call reads one coarse prefix and one fine prefix, then masks at most
    one word; it performs no interval or word-length scan.
    """

    __slots__ = ("words", "prefix", "word_prefix", "zeros", "n", "level")

    def __init__(self, words, n, zeros, level):
        self.words = words
        self.n = n
        self.zeros = zeros
        self.level = level

        word_count = len(words)
        prefix = array("Q", [0])
        word_prefix = array("H")
        absolute_total = 0

        # Coarse prefix: one absolute one-count before each 1024-bit block.
        for start in range(0, word_count, _BLOCK_WORDS):
            if start > 0:
                prefix.append(absolute_total)
            block_total = 0
            for j in range(start, min(start + _BLOCK_WORDS, word_count)):
                block_total += words[j].bit_count()
            absolute_total += block_total

        # Fine prefix: relative one-count for every 64-bit word in its block.
        # Every block has only 16 words, so each relative value fits in 16
        # bits. The final value is the complete count for the last partial
        # block.
        block_total = 0
        for j in range(word_count):
            if j % _BLOCK_WORDS == 0:
                block_total = 0
            word_prefix.append(block_total)
            block_total += words[j].bit_count()
        word_prefix.append(block_total)

        # Support rank1(n) when n is exactly on a block boundary.
        prefix.append(absolute_total)

        self.prefix = prefix
        self.word_prefix = word_prefix

    def rank1(self, pos):
        """Return the number of one-bits in positions ``[0, pos)``."""
        if pos <= 0:
            return 0

        if pos % _BLOCK_BITS == 0:
            # This also handles pos == 0 for an empty bit vector.
            return self.prefix[pos // _BLOCK_BITS]

        block = (pos - 1) >> 10  # Block containing the last counted bit.
        word = pos >> 6         # _WORD_BITS == 64
        remainder = pos & 63
        relative_index = block * _BLOCK_WORDS + (word & (_BLOCK_WORDS - 1))

        count = self.prefix[block]
        if remainder:
            count += self.word_prefix[relative_index]
            count += (self.words[word] & ((1 << remainder) - 1)).bit_count()
        else:
            # At a word boundary, relative_index identifies the preceding word
            # and the fine prefix at that index is its cumulative count.
            count += self.word_prefix[relative_index]

        return count


class Index:
    """Immutable wavelet-matrix index over a list of signed 64-bit integers.

    Building an index snapshots ``values``.  Later mutations of the caller's
    list do not affect the index.

    Complexity with ``n == len(values)`` and ``sigma == number of distinct
    values``:

    * Construction: O(n log(sigma + 1) + sigma log(sigma + 1))
    * :meth:`kth` and :meth:`rank`: O(log(sigma + 1))
    * Index bits: O(n log(sigma + 1)) plus O(sigma) machine words for the
      alphabet and value-to-dense-rank mapping.
    """

    def __init__(self, values):
        if not isinstance(values, list):
            raise TypeError("values must be a list")

        for value in values:
            # Reject bool as well as floats, strings, etc.
            if type(value) is not int:
                raise TypeError("all values must be signed 64-bit integers")
            if value < MIN_INT64 or value > MAX_INT64:
                raise ValueError("all values must be signed 64-bit integers")

        n = len(values)

        # Sorted distinct values and the dense alphabet rank mapping.
        keys = tuple(sorted(set(values)))
        sigma = len(keys)
        value_to_rank = {value: i for i, value in enumerate(keys)}

        # Packed dense codes; no Python int is retained per element after
        # construction.
        codes = array("Q", (value_to_rank[value] for value in values))

        levels_count = sigma.bit_length()  # ceil(log2(sigma + 1)), or 0
        levels = []

        for level in range(levels_count - 1, -1, -1):
            words = array("Q")
            zero_codes = array("Q")
            one_codes = array("Q")
            one_count = 0
            mask = 1 << level

            for i, code in enumerate(codes):
                if i & 63 == 0:
                    words.append(0)

                if code & mask:
                    words[-1] |= 1 << (i & 63)
                    one_codes.append(code)
                    one_count += 1
                else:
                    zero_codes.append(code)

            levels.append(_BitVector(words, n, n - one_count, level))

            # Stable partition by this bit, forming the next wavelet level.
            zero_codes.extend(one_codes)
            codes = zero_codes

        self._n = n
        self._sigma = sigma
        self._levels_count = levels_count
        self._levels = levels
        self._keys = keys
        self._value_to_rank = value_to_rank

    @property
    def n(self):
        return self._n

    @property
    def sigma(self):
        return self._sigma

    @property
    def alphabet(self):
        """Distinct indexed values in ascending order."""
        return self._keys

    def dense_rank(self, value):
        """Return the dense rank of an exact stored value."""
        if type(value) is not int:
            raise TypeError("value must be an integer")
        try:
            return self._value_to_rank[value]
        except KeyError:
            raise ValueError("value is not present in the index") from None

    def _check_bounds(self, left, right):
        if not isinstance(left, int) or not isinstance(right, int):
            raise TypeError("range endpoints must be integers")
        if not (0 <= left <= right <= self._n):
            raise ValueError(
                f"required 0 <= left <= right <= {self._n}, "
                f"got [{left}, {right})"
            )

    @staticmethod
    def _check_integer(value, name):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    def kth(self, left, right, k):
        """Return the 0-based ``k``-th smallest value in ``[left, right)``.

        Repeated values are counted separately.  Raises :class:`ValueError`
        for an invalid half-open range or when ``k`` is outside
        ``0 <= k < right - left``.
        """
        self._check_bounds(left, right)
        self._check_integer(k, "k")

        length = right - left
        if not (0 <= k < length):
            raise ValueError(
                f"required 0 <= k < {length}, got k={k}"
            )

        result_code = 0

        for level in range(self._levels_count - 1, -1, -1):
            bit_vector = self._levels[self._levels_count - 1 - level]
            ones_left = bit_vector.rank1(left)
            ones_right = bit_vector.rank1(right)

            zeros_left = left - ones_left
            zeros_right = right - ones_right
            zero_count = zeros_right - zeros_left

            if k < zero_count:
                left, right = zeros_left, zeros_right
            else:
                k -= zero_count
                result_code |= 1 << level
                left = bit_vector.zeros + ones_left
                right = bit_vector.zeros + ones_right

        return self._keys[result_code]

    def rank(self, left, right, x):
        """Return the number of values in ``[left, right)`` strictly below ``x``."""
        self._check_bounds(left, right)
        self._check_integer(x, "x")
        if left == right:
            return 0

        # Dense rank that x would have if it were inserted: number of distinct
        # stored values strictly below x.
        target = bisect_left(self._keys, x)

        if target == 0:
            return 0
        if target == self._sigma:
            return right - left

        answer = 0

        for level in range(self._levels_count - 1, -1, -1):
            bit_vector = self._levels[self._levels_count - 1 - level]
            ones_left = bit_vector.rank1(left)
            ones_right = bit_vector.rank1(right)

            zeros_left = left - ones_left
            zeros_right = right - ones_right

            if (target >> level) & 1:
                # All values in the zero branch are smaller at this bit.
                answer += zeros_right - zeros_left
                left = bit_vector.zeros + ones_left
                right = bit_vector.zeros + ones_right
            else:
                left, right = zeros_left, zeros_right

        return answer

    def space_report(self):
        """Return structural space accounting for the packed index."""
        packed_bits = 0
        directory_bits = 0

        for bit_vector in self._levels:
            packed_bits += len(bit_vector.words) * _WORD_BITS
            directory_bits += len(bit_vector.prefix) * _WORD_BITS
            directory_bits += len(bit_vector.word_prefix) * 16

        return {
            "n": self._n,
            "sigma": self._sigma,
            "levels": self._levels_count,
            "packed_bitvector_bits": packed_bits,
            "block_rank_directory_bits": directory_bits,
            "wavelet_matrix_bits": packed_bits + directory_bits,
            "wavelet_matrix_bytes": (packed_bits + directory_bits) // 8,
            "alphabet_tuple_machine_words": self._sigma,
            "value_rank_map_machine_words_order": self._sigma,
            "level_zero_counter_words": self._levels_count,
            "machine_word_bits": _WORD_BITS,
            "rank_block_bits": _BLOCK_BITS,
            "word_prefix_bits": 16,
        }
