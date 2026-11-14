# ranges.Index — packed wavelet matrix for local quantiles and ranks

This repository implements an immutable index over a list of signed 64-bit
integers using only the Python standard library.

```python
from ranges import Index

index = Index([4, -2, 4, 7])
index.kth(0, 3, 1)   # 4
index.rank(1, 4, 4)  # 1
```

## API

### `Index(values)`

* `values` must be a Python list of signed 64-bit integers.
* The constructor snapshots its input. Later mutating or clearing the caller's
  list does not affect the index.
* Repeated values are retained and counted separately.

### `Index.kth(left, right, k)`

Returns the zero-based `k`-th smallest value in the half-open interval
`[left, right)`.

* Requires `0 <= left <= right <= n`.
* Requires `0 <= k < right - left`.
* Invalid ranges or `k` raise `ValueError`; non-integer endpoints or `k` raise
  `TypeError`.

### `Index.rank(left, right, x)`

Returns the number of values in `[left, right)` strictly less than integer `x`.

* Requires `0 <= left <= right <= n`.
* Returns `0` for every empty interval, including `[0, 0)`.
* Any exact Python integer threshold is accepted, including values outside the
  stored signed 64-bit domain; such thresholds naturally return `0` or the full
  interval length.
* Invalid ranges raise `ValueError`; non-integer endpoints or thresholds raise
  `TypeError`.

## Algorithm

The distinct input values are sorted into an alphabet of size `σ`, and each
input value is replaced by its dense alphabet rank. Those ranks form a wavelet
matrix:

1. At each bit level, from most significant to least significant, the current
   codes are stably partitioned into zero-bit and one-bit subsequences.
2. The partition bits form a bit vector.
3. A query follows the wavelet matrix through at most `L` levels, where
   `L = ceil(log2(σ + 1))`.
4. Each level needs two bit-rank calls. Bit vectors are packed into C
   `uint64` words via `array('Q')`.

### Constant-time bit rank

Every bit vector has a two-level rank directory:

1. one absolute 64-bit prefix count per 1024-bit block;
2. a 16-bit fine-prefix array for the relative one-counts of the at-most-16
   packed words inside that block.

A bit-rank query reads one block prefix and one fine prefix, then masks at most
one word. That work is independent of both `n` and the requested interval
width.

### `kth`

At each level, the query computes how many values in the current interval are
in the zero branch. If `k` is smaller than that count, the answer has bit zero
at this level and the interval moves to the zero branch. Otherwise it subtracts
the zero-branch count, sets the result bit, and moves to the one branch. After
all levels, the resulting dense rank is mapped back to the original integer.

### `rank`

`rank(l, r, x)` first uses binary search over the sorted alphabet to find the
dense rank `x` would have, i.e. the number of distinct stored values below `x`.
It then walks the wavelet matrix: whenever the target's current bit is one, all
elements in the zero branch are smaller and their interval count is added to
the answer.

## Complexity

Let `n` be the number of indexed elements and `σ` the number of distinct
values.

| Operation | Complexity |
| --- | --- |
| Construction | `O(n log(σ+1) + σ log(σ+1))` |
| `kth` | `O(log(σ+1))` |
| `rank` | `O(log(σ+1))`, including alphabet binary search |
| Bit-vector index space | `O(n log(σ+1))` bits |
| Alphabet/rank mapping | `O(σ)` machine words |

The query never scans `[left, right)` and never copies or sorts that interval.

## Space details and Python object policy

There is no retained per-level, per-element Python object matrix:

* Wavelet bit vectors use `array('Q')`, so each 64 entries occupy one packed
  64-bit word.
* Block rank prefixes also use `array('Q')`.
* Temporary `array('Q')` code partitions are replaced level by level and become
  unreachable when construction finishes.
* The sorted distinct alphabet is a tuple of `σ` values.
* The value-to-dense-rank mapping is a dictionary of `O(σ)` machine-word-scale
  entries.
* Each level retains a small fixed amount of metadata, notably its zero-branch
  count.

`Index.space_report()` reports actual packed bit-vector and rank-directory bit
counts for a constructed index. The two-level directory uses one 64-bit block
prefix per 1024 bits and one 16-bit word prefix per 64 bits, so its total is
linear in the packed bit-vector length and remains within the stated
`O(nL)`-bit index bound.

## Verification and demo

Run the full unittest suite:

```bash
python3 -m unittest -v test_ranges.py
```

`reference_sorting.py` is an independent oracle. It deliberately uses whole
interval slices, `sorted(...)`, and `bisect_left`; no wavelet-matrix code is
shared with it. The tests compare both APIs against that oracle over:

* empty, singleton, repeated, signed, and 64-bit-boundary values;
* exhaustive intervals for small cases;
* randomized low- and high-cardinality data;
* packed-word and rank-block boundary lengths;
* invalid ranges, invalid `k`, bad types, and out-of-range stored values;
* original-list mutation after construction.

Run the demonstration:

```bash
python3 demo.py
```

## Scale measurements

Run:

```bash
python3 benchmark.py
```

This regenerates `.pointwise-runtime.json`. Each case includes the pointwise
timings from all repeated 10,000-query batches, build time, independent sorting
checks, and a structural space report.

Observed on the environment recorded in the checked-in manifest (CPython 3.11,
Linux aarch64):

| `n` | Distribution | `σ` | Levels | Build seconds | Microseconds/query |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,000 | high-cardinality | 1,000 | 10 | 0.0035 | 10.64 |
| 1,000 | low-cardinality | 8 | 4 | 0.0018 | 4.34 |
| 10,000 | high-cardinality | 10,000 | 14 | 0.0494 | 15.92 |
| 10,000 | low-cardinality | 8 | 4 | 0.0079 | 5.38 |
| 100,000 | high-cardinality | 100,000 | 17 | 0.6996 | 17.90 |
| 100,000 | low-cardinality | 8 | 4 | 0.1059 | 4.14 |
| 1,000,000 | high-cardinality | 1,000,000 | 20 | 8.2175 | 26.05 |
| 1,000,000 | low-cardinality | 8 | 4 | 1.3589 | 5.43 |

At `n = 1,000,000` and `σ = 1,000,000`, the wavelet matrix uses 20 packed
levels and its reported bit-vector/rank-directory storage is 26,252,160 bits
(3,281,520 bytes) before the alphabet mapping.
