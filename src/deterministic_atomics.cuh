/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __DETERMINISTIC_ATOMICS_CUH__
#define __DETERMINISTIC_ATOMICS_CUH__

// -----------------------------------------------------------------------------
// Deterministic atomic accumulation for CUDA Bundle Adjustment
//
// This header declares the primitives needed for Phase 1/2 of Option 4
// (see reports/eval-reports/20260418/option4-cuda-determinism-design.md in the
// outer triorb-slam-for-amr repo).
//
// Context
//   `atomicAdd(double*, double)` is not associative for floating-point values,
//   so the order in which threads update the same address directly influences
//   the final bit-level value. For Bundle Adjustment Hessian blocks
//   (Hpp / Hll / HscDirect / Hpl) that aggregate per-edge contributions, this
//   makes repeated runs of the same BA problem produce subtly different
//   solutions. On the JOINT=1 ext-optimization path that non-determinism leaks
//   through the stage accept/reject gates into landmark count / pose variance
//   (observed range [307, 887] at n=5 in the 2026-04-18 late-night evaluation).
//
// Approach
//   Scale the double value to a signed 64-bit integer using a fixed
//   `FIXED_POINT_SCALE`, then accumulate via `atomicAdd(long long*, long long)`
//   which is commutative and associative. Convert back to double after the
//   accumulation kernel completes. Integer add is order-independent, so the
//   result is bit-exact across runs (given identical inputs and edge layout).
//
// Scope of this header
//   * Only declares the primitives. It is deliberately NOT included from any
//     .cu file yet so that existing kernels remain bit-for-bit unchanged until
//     Phase 2 wires the new path through an opt-in flag.
//   * Does not allocate / deallocate any storage. That responsibility belongs
//     to `CudaBlockSolver` once it adopts this path (Phase 2).
//
// Scale factor reasoning
//   * AMR SLAM BA runs we care about have Hpp diagonal magnitudes reaching
//     ~1e6 (omega * edge_count) and off-diagonal / bp components up to ~1e4.
//   * With `FIXED_POINT_SCALE = 2^30 ≈ 1.07e9`, a scalar value of 1e6 maps to
//     ~1.07e15 which fits comfortably in int64 (max 9.22e18), leaving ~4 orders
//     of magnitude safety margin even with accumulation from 1e5 edges.
//   * Precision: values smaller than `1 / FIXED_POINT_SCALE ≈ 9.3e-10` are
//     quantized. That is well below double precision relative error for the
//     Jacobian magnitudes involved, and below our typical Levenberg-Marquardt
//     step size, so no solver behaviour regression is expected.
//
// Follow-up (Phase 2-6 per design doc)
//   Phase 1: Add unit test (`test_fixed_point_scale.cu`) covering representative
//            Jacobian ranges [1e-4, 1e2] and edge counts [100, 100000].
//   Phase 2: Wire into `Hpp.at(iPExt)` accumulation only (line 1237 of
//            cuda_block_solver.cu), gated by a runtime flag that the outer
//            repo forwards via `TRIORB_CUDA_BA_DETERMINISTIC_ACCUM`.
//   Phase 3: Extend to remaining call sites (1100, 1103, 1108, 1111, 1238,
//            1242, 1247, 1379, 1409).
//   Phase 4: Flip default once JOINT=1 n=5 rel stddev < 10% is confirmed.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "scalar.h"

namespace cuba
{
namespace gpu
{
namespace deterministic
{

// Number of fractional bits. 2^30 gives ~9.3e-10 precision and leaves room for
// aggregating values up to ~1e9 into a 64-bit integer without overflow.
// Kept as a constexpr so callers can reason about overflow bounds at compile
// time.
constexpr int FIXED_POINT_FRACTIONAL_BITS = 30;
constexpr double FIXED_POINT_SCALE = static_cast<double>(1ULL << FIXED_POINT_FRACTIONAL_BITS);
constexpr double FIXED_POINT_INV_SCALE = 1.0 / FIXED_POINT_SCALE;

// The maximum absolute value that can be accumulated without overflow, given
// the current scale factor. Expressed in the underlying double domain.
// `INT64_MAX / FIXED_POINT_SCALE ≈ 8.59e9`.
constexpr double FIXED_POINT_MAX_ABS = 9.22337e18 / FIXED_POINT_SCALE;

// -----------------------------------------------------------------------------
// Device primitives
// -----------------------------------------------------------------------------

// Convert a double-precision increment to a signed 64-bit fixed-point quantum.
// Uses round-to-nearest-even (hardware default for `__double2ll_rn`), which is
// itself deterministic.
__device__ __forceinline__ long long
toFixedPoint(Scalar value)
{
	// Clamp to the representable range to avoid UB from the cvt instruction.
	if (value >= FIXED_POINT_MAX_ABS) return (long long)((1ULL << 63) - 1ULL);
	if (value <= -FIXED_POINT_MAX_ABS) return (long long)(1ULL << 63);
#ifdef USE_FLOAT32
	return __float2ll_rn(value * (float)FIXED_POINT_SCALE);
#else
	return __double2ll_rn(value * FIXED_POINT_SCALE);
#endif
}

// Host+device accessible: Phase 3g reads per-iteration chi2 / scale scalars back
// to the CPU before converting to double, so this helper must compile for both.
__host__ __device__ __forceinline__ Scalar
fromFixedPoint(long long q)
{
	return static_cast<Scalar>(q) * static_cast<Scalar>(FIXED_POINT_INV_SCALE);
}

// Deterministic additive accumulation: add `value` to the fixed-point scalar
// at `address`. Thread-safe across any number of blocks; the result is
// bit-identical to a serial accumulation (independent of execution order).
__device__ __forceinline__ void
atomicAccumDet(long long* address, Scalar value)
{
	long long quantum = toFixedPoint(value);
	atomicAdd(reinterpret_cast<unsigned long long*>(address),
	          static_cast<unsigned long long>(quantum));
}

// Deterministic subtractive accumulation (mirror of DEACCUM_ATOMIC).
__device__ __forceinline__ void
atomicDeaccumDet(long long* address, Scalar value)
{
	atomicAccumDet(address, -value);
}

// -----------------------------------------------------------------------------
// Global kernels (Phase 2+ will launch these; kept here so the reviewer can
// see the full shape in one place).
// -----------------------------------------------------------------------------

// Zero-initialise an int64 buffer of length `n`. Call this before any
// accumulation kernel writes to the buffer for a fresh BA iteration.
__global__ inline void
zeroFixedPointBufferKernel(int n, long long* data)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) data[i] = 0;
}

// Convert an int64 fixed-point buffer of length `n` into the target double
// buffer. Overwrites `dst`; caller is responsible for issuing this kernel
// exactly once per accumulation phase.
__global__ inline void
fixedPointToDoubleKernel(int n, const long long* src, Scalar* dst)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) dst[i] = fromFixedPoint(src[i]);
}

} // namespace deterministic
} // namespace gpu
} // namespace cuba

#endif // !__DETERMINISTIC_ATOMICS_CUH__
