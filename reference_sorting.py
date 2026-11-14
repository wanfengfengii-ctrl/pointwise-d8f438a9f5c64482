"""Independent reference implementation used only by verification scripts.

This implementation deliberately uses whole-interval sorted Python slices and
``bisect``.  It is independent of the wavelet-matrix code and serves as an
oracle; it is not part of the query-time index implementation.
"""

from bisect import bisect_left


class ReferenceIndex:
    def __init__(self, values):
        # Defensive copy gives the reference the same snapshot semantics.
        self._values = list(values)
        self._n = len(self._values)

    def kth(self, left, right, k):
        if not (0 <= left <= right <= self._n):
            raise ValueError("invalid interval")
        length = right - left
        if not (0 <= k < length):
            raise ValueError("invalid k")
        return sorted(self._values[left:right])[k]

    def rank(self, left, right, x):
        if not (0 <= left <= right <= self._n):
            raise ValueError("invalid interval")
        ordered = sorted(self._values[left:right])
        return bisect_left(ordered, x)
