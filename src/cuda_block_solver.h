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

#ifndef __CUDA_BLOCK_SOLVER_H__
#define __CUDA_BLOCK_SOLVER_H__

#include "device_matrix.h"
#include "robust_kernel.h"

namespace cuba
{

namespace gpu
{

void waitForKernelCompletion();

void buildHplStructure(GpuVec3i& blockpos, GpuHplBlockMat& Hpl, GpuVec1i& indexPL, GpuVec1i& nnzPerCol);

void findHschureMulBlockIndices(const GpuHplBlockMat& Hpl, const GpuHscBlockMat& Hsc,
	GpuVec3i& mulBlockIds);

Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec2d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions,
	const RobustKernel& kernel,
	GpuVec2d& errors, GpuVec3d& Xcs, Scalar* chi);

Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec3d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts,
	const RobustKernel& kernel,
	GpuVec3d& errors, GpuVec3d& Xcs, Scalar* chi);

// Option 4 Phase 2+: `Hpp_int_ext_raw` / `bp_int_ext_raw` / `Hll_int_raw` /
// `bl_int_raw` / `HscDirect_int_raw` are pointers into `long long` buffers
// that mirror the layouts of `Hpp.values()`, `bp.values()`, `Hll.values()`,
// `bl.values()`, and `HscDirect.values()` respectively. When non-null, the
// kernel routes the corresponding block / vector accumulations through a
// fixed-point int64 atomicAdd path so the final values are bit-identical
// across runs. The caller is responsible for (a) zeroing each buffer before
// this call and (b) invoking the paired `convertFixedPoint*Range` /
// `convertFixedPointHscDirect` helpers after the call to propagate slots
// back into the double buffers. When null, the legacy `atomicAdd(double*)`
// path is used, preserving the pre-Phase-2 behavior. Phase 3a adds
// `bp_int_ext_raw`; Phase 3b adds body-range conversion through the same
// `Hpp_int_ext_raw` / `bp_int_ext_raw` buffers; Phase 3c adds
// `Hll_int_raw` / `bl_int_raw` for the full landmark range; Phase 3d adds
// `HscDirect_int_raw` for the direct body×ext Schur cross-block. Remaining
// atomic sites (`Hpl.at(hplExtSlot)`, Schur update) will be migrated in
// Phase 3e+.
void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec5d& cameras, const GpuVec2d& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1i& edge2HplExt,
	const GpuVec1i& edge2ExtIP, const GpuVec1i& edge2HscPE, const GpuVec1b& flags,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions,
	const RobustKernel& kernel,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl,
	GpuHscBlockMat& HscDirect,
	long long* Hpp_int_ext_raw = nullptr,
	long long* bp_int_ext_raw = nullptr,
	long long* Hll_int_raw = nullptr,
	long long* bl_int_raw = nullptr,
	long long* HscDirect_int_raw = nullptr);

void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec5d& cameras, const GpuVec3d& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1i& edge2HplExt,
	const GpuVec1i& edge2ExtIP, const GpuVec1i& edge2HscPE, const GpuVec1b& flags,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts,
	const RobustKernel& kernel,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl,
	GpuHscBlockMat& HscDirect,
	long long* Hpp_int_ext_raw = nullptr,
	long long* bp_int_ext_raw = nullptr,
	long long* Hll_int_raw = nullptr,
	long long* bl_int_raw = nullptr,
	long long* HscDirect_int_raw = nullptr);

// Option 4 Phase 2: copy the ext-range slots of a fixed-point int64 buffer
// (`src_int`, sized identically to `Hpp.values()`) into the corresponding
// double slots of `Hpp`. Only elements in `[numBody, numBody + numExt)` are
// touched; body-pose slots are left untouched.
void convertFixedPointHppExtRange(const long long* src_int,
	GpuPxPBlockVec& Hpp, int numBody, int numExt);

// Option 4 Phase 3a: same pattern for the PDIM-sized gradient vector `bp`.
// `src_int` is sized identically to `bp.values()`. Only elements in
// `[numBody, numBody + numExt)` are touched; body-pose slots are left
// untouched.
void convertFixedPointBpExtRange(const long long* src_int,
	GpuPx1BlockVec& bp, int numBody, int numExt);

// Option 4 Phase 3b: copy the body-range slots of a fixed-point int64 buffer
// into the corresponding double slots of `Hpp` / `bp`. Only elements in
// `[0, numBody)` are touched; ext-range slots are handled separately by the
// *ExtRange helpers. When the deterministic path is active the caller
// invokes both Body + Ext helpers after each `constructQuadraticForm()` to
// cover the full `[0, numBody + numExt)` range.
void convertFixedPointHppBodyRange(const long long* src_int,
	GpuPxPBlockVec& Hpp, int numBody);

void convertFixedPointBpBodyRange(const long long* src_int,
	GpuPx1BlockVec& bp, int numBody);

// Option 4 Phase 3c: copy the landmark-range slots of a fixed-point int64
// buffer (`src_int`, sized identically to `Hll.values()` / `bl.values()`) into
// the corresponding double slots. Touches `[0, numL)` slots — each block is
// LDIM*LDIM scalars for Hll, LDIM scalars for bl. Landmark slots are disjoint
// from Hpp / bp so these helpers can be called independently of the body /
// ext helpers above.
void convertFixedPointHllRange(const long long* src_int,
	GpuLxLBlockVec& Hll, int numL);

void convertFixedPointBlRange(const long long* src_int,
	GpuLx1BlockVec& bl, int numL);

// Option 4 Phase 3d: copy the full non-zero slot range of a fixed-point int64
// buffer into the double `HscDirect` block matrix. HscDirect only receives
// atomic accumulation writes (no ASSIGN path) so the entire `[0, nnz)` slot
// range (`nnz * PDIM * PDIM` scalars) is converted unconditionally.
// `src_int` must be sized identically to `HscDirect.values()` and zeroed
// prior to `constructQuadraticForm`.
void convertFixedPointHscDirect(const long long* src_int,
	GpuHscBlockMat& HscDirect, int nnz);

// Build per-edge ext Hpl slot vector. For each edge e:
//   edge2HplExt[e] = (dedup_slot[e] < 0) ? -1 : edge2Hpl[nedges_total + dedup_slot[e]]
void buildEdgeExtHpl(const int* h_dedup_slot_per_edge, int nedges_total,
	int nExtDedupSlots, const GpuVec1i& edge2Hpl, GpuVec1i& edge2HplExt);

// Fill edge2HplExt entirely with -1 sentinels (no ext contribution).
void fillEdge2HplExtSentinel(GpuVec1i& edge2HplExt, int nedges_total);

// Joint mode: scatter ext solution (qs[iPExt], ts[iPExt]) into per-edge q_exts/t_exts.
// Edges with edge2ExtIP[iE] < 0 (fixed or decoupled) keep their existing per-edge values.
void syncExtSolutionToPerEdge(const GpuVec4d& qs, const GpuVec3d& ts,
	const GpuVec1i& edge2ExtIP, GpuVec4d& q_exts, GpuVec3d& t_exts, int nedges);

// Apply a prior (λ_rot on rotation dof, λ_trans on translation dof) to the
// diagonal of Hpp[iP] for iP ∈ [numBody, numBody+numExt).
void addExtPrior(GpuPxPBlockVec& Hpp, int numBody, int numExt, Scalar lambda_rot, Scalar lambda_trans);

// Per-iteration SE(3) retraction, with clamping of delta for ext slots
// (iP >= num_body) to avoid large calibration jumps.
void updatePoses(const GpuPx1BlockVec& xp, GpuVec4d& qs, GpuVec3d& ts, int num_body,
	Scalar max_ext_trans, Scalar max_ext_rot);

void computeChiSquares(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec2d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions, GpuVec1d& chiSqs);

void computeChiSquares(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec3d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, GpuVec1d& chiSqs);

Scalar maxDiagonal(const GpuPxPBlockVec& Hpp, Scalar* maxD);

Scalar maxDiagonal(const GpuLxLBlockVec& Hll, Scalar* maxD);

void addLambda(GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup);

void addLambda(GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup);

void restoreDiagonal(GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup);

void restoreDiagonal(GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup);

void computeBschure(const GpuPx1BlockVec& bp, const GpuHplBlockMat& Hpl, const GpuLxLBlockVec& Hll,
	const GpuLx1BlockVec& bl, GpuPx1BlockVec& bsc, GpuLxLBlockVec& invHll, GpuPxLBlockVec& Hpl_invHll);

void computeHschure(const GpuPxPBlockVec& Hpp, const GpuHscBlockMat& HscDirect, const GpuPxLBlockVec& Hpl_invHll,
	const GpuHplBlockMat& Hpl, const GpuVec3i& mulBlockIds, GpuHscBlockMat& Hsc);

void convertHschureBSRToCSR(const GpuHscBlockMat& HscBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HscCSR);

void twistCSR(int size, int nnz, const int* srcRowPtr, const int* srcColInd, const int* P,
	int* dstRowPtr, int* dstColInd, int* dstMap, int* nnzPerRow);

void permute(int size, const Scalar* src, Scalar* dst, const int* P);

void schurComplementPost(const GpuLxLBlockVec& invHll, const GpuLx1BlockVec& bl,
	const GpuHplBlockMat& Hpl, const GpuPx1BlockVec& xp, GpuLx1BlockVec& xl);

// Legacy entry point: body-only update (Default OFF path).
void updatePoses(const GpuPx1BlockVec& xp, GpuVec4d& qs, GpuVec3d& ts);

void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& Xws);

void computeScale(const GpuVec1d& x, const GpuVec1d& b, Scalar* scale, Scalar lambda);

void solveDiagonalSystem(const GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuPx1BlockVec& xp);

void solveDiagonalSystem(const GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuLx1BlockVec& xl);

} // namespace gpu

} // namespace cuba

#endif // !__CUDA_BLOCK_SOLVER_H__
