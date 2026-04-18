/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0
*/

// -----------------------------------------------------------------------------
// test_fixed_point_scale.cu
//
// Option 4 Phase 2 precision validation for the int64 fixed-point atomic
// accumulation path (`deterministic_atomics.cuh`). The unit test validates
// two independent properties:
//
//   1. Round-trip precision: `fromFixedPoint(toFixedPoint(x))` stays within a
//      strict tolerance for scalars covering the range [1e-4, 1e2] that we
//      encounter in Jacobian entries of real AMR BA problems.
//
//   2. Accumulation determinism: summing N random scalars with
//      `atomicAccumDet` (int64 atomicAdd) from many threads yields a result
//      that is bit-identical across repeated runs and within a tight bound of
//      the reference CPU serial sum. We sweep N ∈ {100, 1000, 10000, 100000}
//      to exercise the worst-case edge count of typical keyframe BA windows.
//
// The test builds and runs standalone (no test framework dependency). Each
// failing case prints a human-readable message and the process returns 1 on
// any failure so it can be wired into CI later.
//
// Compile:
//   nvcc -std=c++17 -I../src tests/test_fixed_point_scale.cu
//        -o build/test_fixed_point_scale
//
// (The test only uses the header primitives from `deterministic_atomics.cuh`;
//  it does not need to link against `libcuda_bundle_adjustment` so nvcc alone
//  is sufficient.)
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "deterministic_atomics.cuh"

namespace
{

constexpr double kDefaultTolerance = 1e-8;  // ~10 * (1 / 2^30); see scale reasoning in the header.

#define CHECK_CUDA(call)                                                                        \
  do {                                                                                          \
    cudaError_t status = (call);                                                                \
    if (status != cudaSuccess) {                                                                \
      std::fprintf(stderr, "CUDA call failed at %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(status));                                                 \
      std::exit(2);                                                                             \
    }                                                                                           \
  } while (0)

// Kernel: round-trip a list of doubles through toFixedPoint/fromFixedPoint.
__global__ void roundtripKernel(int n, const double* src, double* dst)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long q = cuba::gpu::deterministic::toFixedPoint(src[i]);
  dst[i] = cuba::gpu::deterministic::fromFixedPoint(q);
}

// Kernel: accumulate a list of values into a single int64 fixed-point slot via
// the deterministic atomicAccumDet primitive.
__global__ void accumulateKernel(int n, const double* values, long long* slot)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  cuba::gpu::deterministic::atomicAccumDet(slot, values[i]);
}

bool testRoundTripPrecision()
{
  // Cover a range of magnitudes encountered in Jacobian entries.
  const std::vector<double> samples = {
    1e-4, -1e-4, 1e-3, 5e-3, -7.5e-3,
    0.1, -0.25, 0.5, 0.9999, -0.9999,
    1.0, -1.0, 2.5, -3.14159, 10.0,
    -25.0, 1e2, -1e2, 99.9999, -99.9999,
  };

  const int n = static_cast<int>(samples.size());
  double* d_src = nullptr;
  double* d_dst = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, n * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_dst, n * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_src, samples.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  const int block = 64;
  const int grid = (n + block - 1) / block;
  roundtripKernel<<<grid, block>>>(n, d_src, d_dst);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<double> got(n);
  CHECK_CUDA(cudaMemcpy(got.data(), d_dst, n * sizeof(double), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));

  bool ok = true;
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    const double err = std::abs(samples[i] - got[i]);
    if (err > kDefaultTolerance) {
      std::fprintf(stderr,
        "  [FAIL] roundtrip: x=%+.17g recovered=%+.17g err=%.3e tol=%.3e\n",
        samples[i], got[i], err, kDefaultTolerance);
      ok = false;
    }
    max_err = std::max(max_err, err);
  }
  std::printf("  roundtrip: n=%d max_err=%.3e tol=%.3e -> %s\n",
              n, max_err, kDefaultTolerance, ok ? "OK" : "FAIL");
  return ok;
}

bool testAccumulationDeterminism(int n, double magnitude_max, uint64_t seed)
{
  // Generate n random values with magnitudes in [-magnitude_max, magnitude_max].
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-magnitude_max, magnitude_max);
  std::vector<double> host_values(n);
  double cpu_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    host_values[i] = dist(rng);
    cpu_sum += host_values[i];
  }

  double* d_values = nullptr;
  long long* d_slot = nullptr;
  CHECK_CUDA(cudaMalloc(&d_values, n * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_slot, sizeof(long long)));
  CHECK_CUDA(cudaMemcpy(d_values, host_values.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  // Run the accumulation twice and verify bit-identical int64 result.
  long long slot_run1 = 0;
  long long slot_run2 = 0;
  const int block = 256;
  const int grid = (n + block - 1) / block;

  CHECK_CUDA(cudaMemset(d_slot, 0, sizeof(long long)));
  accumulateKernel<<<grid, block>>>(n, d_values, d_slot);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&slot_run1, d_slot, sizeof(long long), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaMemset(d_slot, 0, sizeof(long long)));
  accumulateKernel<<<grid, block>>>(n, d_values, d_slot);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&slot_run2, d_slot, sizeof(long long), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_slot));

  const double gpu_sum = static_cast<double>(slot_run1) *
                         cuba::gpu::deterministic::FIXED_POINT_INV_SCALE;
  // Expected absolute error vs CPU double sum: n rounds of 1-LSB quantization
  // plus the CPU sum's own rounding. We keep the bound generous (~ n * 2e-9).
  const double gpu_cpu_abs_err = std::abs(gpu_sum - cpu_sum);
  const double gpu_cpu_abs_bound = 2.0e-9 * static_cast<double>(n) + 1e-6;

  const bool bit_identical = (slot_run1 == slot_run2);
  const bool accurate = (gpu_cpu_abs_err <= gpu_cpu_abs_bound);

  std::printf(
    "  accum: n=%7d mag<=%.1e seed=%016lx bit_identical=%s gpu_cpu_abs_err=%.3e bound=%.3e -> %s\n",
    n, magnitude_max, (unsigned long) seed,
    bit_identical ? "yes" : "no",
    gpu_cpu_abs_err, gpu_cpu_abs_bound,
    (bit_identical && accurate) ? "OK" : "FAIL");

  if (!bit_identical) {
    std::fprintf(stderr,
      "  [FAIL] accumulation not bit-identical across runs: run1=%lld run2=%lld\n",
      slot_run1, slot_run2);
  }
  if (!accurate) {
    std::fprintf(stderr,
      "  [FAIL] accumulation diverges from CPU sum: gpu=%.17g cpu=%.17g err=%.3e > bound=%.3e\n",
      gpu_sum, cpu_sum, gpu_cpu_abs_err, gpu_cpu_abs_bound);
  }

  return bit_identical && accurate;
}

} // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::fprintf(stderr, "No CUDA device available; skipping test.\n");
    return 77;  // conventional "skip" status for CI frameworks
  }
  CHECK_CUDA(cudaSetDevice(0));

  std::printf("[test_fixed_point_scale] SCALE=2^%d (%g), INV_SCALE=%g\n",
              cuba::gpu::deterministic::FIXED_POINT_FRACTIONAL_BITS,
              cuba::gpu::deterministic::FIXED_POINT_SCALE,
              cuba::gpu::deterministic::FIXED_POINT_INV_SCALE);
  std::printf("[test_fixed_point_scale] MAX_ABS=%g\n",
              cuba::gpu::deterministic::FIXED_POINT_MAX_ABS);

  int failures = 0;

  // 1. Round-trip precision on representative magnitudes.
  std::printf("\n[1/2] round-trip precision\n");
  if (!testRoundTripPrecision()) {
    ++failures;
  }

  // 2. Determinism + accuracy on parallel accumulation over representative
  //    edge counts and magnitudes.
  std::printf("\n[2/2] parallel accumulation determinism & accuracy\n");
  const int edge_counts[] = {100, 1000, 10000, 100000};
  const double magnitudes[] = {1e-2, 1.0, 1e2};
  for (int ec : edge_counts) {
    for (double mag : magnitudes) {
      if (!testAccumulationDeterminism(ec, mag, 0x0123456789abcdefULL ^ (uint64_t) ec)) {
        ++failures;
      }
    }
  }

  std::printf("\n[test_fixed_point_scale] %s (failures=%d)\n",
              failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
