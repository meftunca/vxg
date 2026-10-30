# Correctness testing

## Round-trip stress (`tests/roundtrip_stress.rs`)

Deterministic randomized `decompress(compress(x)) == x` over structured inputs: NDJSON (with schema
drift and missing/extra keys), CSV and logfmt, mixed numeric/string columns, columns whose number
of distinct values crosses 16 / 256 / 4096, split lines, and inputs without a trailing newline. Each
input runs at block sizes 64 KiB and 16 KiB, decoded with 1 and 4 threads.

```sh
cargo test --release --test roundtrip_stress                       # default: 40 iterations
VXG_STRESS_ITERS=2500 VXG_STRESS_SEED=11 cargo test --release --test roundtrip_stress randomized
VXG_STRESS_SEEDS=a,b,c  ...      # replay exactly these seeds, in order, on one thread
VXG_STRESS_DUMP=1       ...      # write each input to /tmp/vxg_stress_<seed>.bin
```

Tips that found real bugs:

- Failures that depend on earlier inputs: replay with `VXG_STRESS_SEEDS` and drop seeds until the
  history is minimal.
- Failures that differ between runs of the same seed point at undefined behavior. On macOS run with
  `MallocPreScribble=1 MallocScribble=1` (fills new / freed memory with junk) and look at the exit
  code: a run that ends right after `running 1 test` with exit 139 is a segfault.
- Compare against the other decoder: `columns::decode_block` vs `decode_block_strict`, and decode on
  a brand-new thread to get cold thread-local scratch.

## Bugs found this way (all fixed, each with a regression test)

| Symptom | Root cause |
| :--- | :--- |
| `decompress` fails with `column bit index out of bounds` | Columns with more than 256 distinct values were encoded with 9..16-bit dictionary ids, but the fast decoder only read 8-bit ids. |
| `column front dict repeat bit0` | "Previous rank" sentinel `u8::MAX` collided with rank 255 when a column has exactly 256 entries. |
| `block kind/size` (frame rejected) | A packet only slightly smaller than `1 + uleb + raw` was tagged COLUMN, but the frame requires `cn < rn`. |
| Silent data corruption (CRC mismatch) on the first decode of a thread | Use-after-free: stride decode kept raw pointers into a buffer that a later plane run could reallocate. |
| Segfaults and assorted errors, varying by run | Heap overflow: fast row loops bounded their unchecked writes by column 0's capacity, but pooled column vectors have different capacities. |
| `fast lz zero offset` depending on earlier inputs | LZ hash table was only zero-filled in its new tail on growth; stale positions from an earlier input could equal the current position. |

Note on compatibility: frames that an older version wrote with a COLUMN block of `cn >= rn` are
rejected by the frame validator (`block kind/size`) and are not rescued by these fixes. Frames with
wide dictionaries written by older versions are now readable.
