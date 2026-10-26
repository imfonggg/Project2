# COP6527/4520 Project 2 Report: Optimizing SDH Computation on GPUs

Phong Lien

## 1. Overview

`proj2.cu` computes the Spatial Distance Histogram (SDH) of N uniformly distributed 3D points on the GPU. It is run as:

```
/apps/GPU_course/runScript.sh proj2.cu {#of_samples} {bucket_width} {block_size}
```

The program checks its inputs and prints four things: the CPU histogram, the GPU histogram, the difference between the two, and the total kernel time. The GPU histogram is required to match the CPU histogram exactly. All the work runs in one kernel, `sdh_tiled_kernel`. When the histogram counters could overflow, the kernel is launched several times, and the timer covers all of those launches.

## 2. Optimization techniques

### 2.1 Structure-of-arrays layout and coalesced loads
The Project 1 code stored points as an array of `{x, y, z}` double structs. In Project 2 the points are split into three separate arrays: X, Y, and Z. When a block loads a tile, thread `t` reads element `j0 + t` of each array. Consecutive threads therefore read consecutive addresses, so every warp load is fully coalesced.

### 2.2 Tiling with shared memory and registers
A block of B threads owns one *tile* of B points, and each thread keeps its own point in registers. The other tiles are loaded one at a time into shared memory. Once a tile is loaded, every thread compares its point against all B points in the tile. All threads read the same shared-memory address at the same moment, so the read is a broadcast with no bank conflicts. Each global-memory load is reused B times. In Project 1, by contrast, every pair read both points from global memory.

### 2.3 Balanced workload: circular tile assignment
The usual `j > i` loop gives the first blocks far more work than the last ones. With M tiles, block `b` instead pairs its tile with tiles `b+1, ..., b+M/2` (mod M). When M is even, the pair at offset exactly M/2 is handled only by the lower-numbered block. Every unordered pair of tiles is still visited exactly once, but each block now does about the same amount of work (M/2 tiles), so no SM sits idle at the end of the kernel. Pairs inside a block's own tile use the same circular trick at thread level: thread `t` compares with `t+1 ... t+(B-1)/2` (mod B), so the threads in a block also share the work evenly.

### 2.4 Privatized shared-memory histograms (reducing atomic contention)
Each block keeps the histogram in shared memory, using 32-bit counters, which are cheaper than 64-bit atomics. When the histogram is small, the block keeps several private copies (up to one per warp, limited to 16 KB), and each warp updates its own copy. This matters most for large bucket widths: with only a few dozen buckets, a single copy would put many threads on the same counter. At the end of the block, the copies are summed and written to the global 64-bit histogram with **one** atomic per bucket.

To stop the 32-bit counters from overflowing, the host splits the tile offsets across several launches. Each launch keeps the number of pairs per block below 2^32.

### 2.5 Histograms that do not fit in shared memory
For small bucket widths, the histogram can be larger than the shared memory available to a block. In that case the program asks for the maximum opt-in shared memory (`cudaFuncAttributeMaxDynamicSharedMemorySize`) and uses it to cache a *window* of buckets. Pairs whose bucket falls outside the window are added directly to the global histogram with 64-bit atomics. Distances between uniform points in a cube cluster around 0.66 x BOX_SIZE, so the window is centred there, which keeps most updates in shared memory. The program never assumes that the whole histogram fits.

### 2.6 Fast single-precision distances with an exact double-precision fallback
Most GPUs run double-precision arithmetic many times slower than single precision. The kernel first computes each distance `d` in single precision. It then works out the bucket of `d - 0.1` and of `d + 0.1`. If both give the same bucket, that bucket is certainly the correct one. Otherwise, the pair lies close to a bucket boundary, so it is recomputed in double precision. The recomputation uses exactly the same operations and rounding as the CPU code, and the `__dsub_rn`/`__dmul_rn`/`__dadd_rn`/`__dsqrt_rn`/`__ddiv_rn` intrinsics stop the compiler from fusing them into FMAs. The largest float error I measured was 0.0044, so the 0.1 margin leaves more than 20x headroom. For w = 500, only about 0.04% of pairs take the double-precision path. I checked the scheduling and the filter against the CPU baseline for N from 1 to 20,000, bucket widths from 0.3 to 2000, and block sizes from 32 to 1024, including sizes that are not multiples of 32. Every case matched exactly.

## 3. Experimental results

All runs were on GAIVI (GPU: `<fill in>`), CUDA 11.4.

**Kernel time vs. number of points** (bucket width 500, block size 256):

| Points | CPU (given) | GPU Project 1 | GPU Project 2 | Speedup vs. Project 1 |
|---|---|---|---|---|
| 10,000 | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| 100,000 | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| 500,000 | ~1 hour | `<fill>` | `<fill>` | `<fill>` |

**Effect of block size** (100,000 points, bucket width 500):

| Block size | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| Kernel time | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

**Effect of bucket width** (100,000 points, block size 256):

| Bucket width | 2000 | 500 | 100 | 10 | 1 |
|---|---|---|---|---|---|
| Buckets | 20 | 80 | 399 | 3,984 | 39,837 |
| Kernel time | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## 4. Discussion

`<Write 3 to 5 sentences once the numbers are in. For example: which block size was fastest and why (occupancy against the cost of tile loads); how much privatization helped at large bucket widths; how the run time changed once the histogram stopped fitting in shared memory (w = 1).>`
