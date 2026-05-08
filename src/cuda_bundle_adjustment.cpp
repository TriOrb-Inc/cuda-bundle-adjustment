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

#include "cuda_bundle_adjustment.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "constants.h"
#include "sparse_block_matrix.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "cuda_block_solver.h"
#include "cuda_linear_solver.h"
#include "robust_kernel.h"

namespace cuba
{

namespace
{

bool is_cuda_ba_trace_enabled()
{
	const char* env_value = std::getenv("TRIORB_CUDA_BA_TRACE");
	return env_value != nullptr && std::string(env_value) == "1";
}

void trace_cuda_ba(const std::string& message)
{
	if (!is_cuda_ba_trace_enabled())
		return;

	std::cerr << "[cuda_ba] " << message << std::endl;
}

// Joint-solve extrinsics (Option A). Default OFF: extrinsics are treated as
// fixed per-edge constants and routed through q_exts_/t_exts_ exactly as in
// the P1.1 scaffold. When TRIORB_OPTIMIZE_EXTRINSICS_JOINT=1, any unfixed
// extrinsics vertex is appended to verticesP_ after body poses and jointly
// optimized via its own Jacobian contribution.
bool is_joint_ext_optimization_enabled()
{
	const char* env_value = std::getenv("TRIORB_OPTIMIZE_EXTRINSICS_JOINT");
	return env_value != nullptr && std::string(env_value) == "1";
}

} // namespace

static constexpr int EDGE_TYPE_NUM = static_cast<int>(EdgeType::COUNT);

// Joint-mode ext optimization parameters. Prior lambdas are deliberately large
// so the ext degrees of freedom barely move unless strong evidence accumulates.
// Clamp bounds match the CPU-side decoupled solver for parity.
static constexpr double kExtPriorLambdaRot = 1.0e8;
static constexpr double kExtPriorLambdaTrans = 1.0e8;
static constexpr double kExtMaxTranslationPerIter = 0.0005;      // 0.5 mm
static constexpr double kExtMaxRotationPerIter = 0.0017453292519943296;  // 0.1 deg in rad

using VertexMapP = std::map<int, VertexP*>;
using VertexMapL = std::map<int, VertexL*>;
using VertexMapE = std::map<int, ExtrinsicsVertex*>;
using EdgeSet2D = std::unordered_set<Edge2D*>;
using EdgeSet3D = std::unordered_set<Edge3D*>;
using time_point = decltype(std::chrono::steady_clock::now());

static inline time_point get_time_point()
{
	gpu::waitForKernelCompletion();
	return std::chrono::steady_clock::now();
}

static inline double get_duration(const time_point& from, const time_point& to)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(to - from).count();
}

template <typename T>
static constexpr Scalar ScalarCast(T v) { return static_cast<Scalar>(v); }

template <typename T>
static constexpr int IntCast(T v) { return static_cast<int>(v); }

static Vec5d vectorize(const CameraParams& camera)
{
	Vec5d v;
	v[0] = ScalarCast(camera.fx);
	v[1] = ScalarCast(camera.fy);
	v[2] = ScalarCast(camera.cx);
	v[3] = ScalarCast(camera.cy);
	v[4] = ScalarCast(camera.bf);
	return v;
}

/** @brief Implementation of Block solver.
*/
class CudaBlockSolver
{
public:

	enum ProfileItem
	{
		PROF_ITEM_INITIALIZE,
		PROF_ITEM_BUILD_STRUCTURE,
		PROF_ITEM_COMPUTE_ERROR,
		PROF_ITEM_BUILD_SYSTEM,
		PROF_ITEM_SCHUR_COMPLEMENT,
		PROF_ITEM_DECOMP_SYMBOLIC,
		PROF_ITEM_DECOMP_NUMERICAL,
		PROF_ITEM_UPDATE,
		PROF_ITEM_NUM
	};

	struct PLIndex
	{
		int P, L;
		PLIndex(int P = 0, int L = 0) : P(P), L(L) {}
	};

	void clear()
	{
		verticesP_.clear();
		verticesL_.clear();
		baseEdges_.clear();
		qs_.clear();
		ts_.clear();
		Xws_.clear();
		measurements2D_.clear();
		measurements3D_.clear();
		omegas_.clear();
		edge2PL_.clear();
		edgeFlags_.clear();
		HplBlockPos_.clear();
		q_exts_.clear();
		t_exts_.clear();
		distortions_.clear();

		numP_ = numL_ = nedges2D_ = nedges3D_ = nHplBlocks_ = 0;
		optimizeP_ = optimizeL_ = false;

		// Joint ext optimization bookkeeping (Option A).
		numBody_ = 0;
		numExt_ = 0;
		extJoint_ = false;
		edge2ExtIP_.clear();
		edge_to_ext_dedup_slot_.clear();
		nExtHplBlocks_ = 0;

		// Option 4 Phase 2: deterministic accumulation bookkeeping. The flag
		// itself is owned by the solver via setDeterministicAccum() and is not
		// reset here (callers toggle it before each optimize() call). The
		// int64 mirror buffer is re-sized in buildStructure() to match d_Hpp_.
	}

	void initialize(const VertexMapP& vertexMapP, const VertexMapL& vertexMapL,
		const VertexMapE& vertexMapE,
		const EdgeSet2D& edgeSet2D, const EdgeSet3D& edgeSet3D, const RobustKernel kernels[])
	{
		trace_cuda_ba(
			"solver initialize begin: pose_vertices=" + std::to_string(vertexMapP.size()) +
			", landmark_vertices=" + std::to_string(vertexMapL.size()) +
			", edges2d=" + std::to_string(edgeSet2D.size()) +
			", edges3d=" + std::to_string(edgeSet3D.size()));
		const auto t0 = std::chrono::steady_clock::now();
		trace_cuda_ba("solver initialize after start timestamp");

		clear();
		trace_cuda_ba("solver initialize after clear");

		verticesP_.reserve(vertexMapP.size());
		verticesL_.reserve(vertexMapL.size());
		baseEdges_.reserve(edgeSet2D.size() + edgeSet3D.size());
		HplBlockPos_.reserve(edgeSet2D.size() + edgeSet3D.size());
		qs_.reserve(vertexMapP.size());
		ts_.reserve(vertexMapP.size());
		Xws_.reserve(vertexMapL.size());
		measurements2D_.reserve(edgeSet2D.size());
		measurements3D_.reserve(edgeSet3D.size());
		omegas_.reserve(edgeSet2D.size() + edgeSet3D.size());
		edge2PL_.reserve(edgeSet2D.size() + edgeSet3D.size());
		edgeFlags_.reserve(edgeSet2D.size() + edgeSet3D.size());
		q_exts_.reserve(edgeSet2D.size() + edgeSet3D.size());
		t_exts_.reserve(edgeSet2D.size() + edgeSet3D.size());
		distortions_.reserve(edgeSet2D.size());

		std::vector<VertexP*> fixedVerticesP_;
		std::vector<VertexL*> fixedVerticesL_;
		int numP = 0;
		int numL = 0;

		// assign pose vertex id
		// gather rotations and translations into each vector
		for (const auto& [id, vertexP] : vertexMapP)
		{
			if (vertexP->edges.empty())
				continue;

			if (!vertexP->fixed)
			{
				vertexP->iP = numP++;
				verticesP_.push_back(vertexP);
				qs_.emplace_back(vertexP->q.coeffs().data());
				ts_.emplace_back(vertexP->t.data());
				cameras_.emplace_back(vectorize(vertexP->camera));
			}
			else
			{
				fixedVerticesP_.push_back(vertexP);
			}
		}

		// assign landmark vertex id
		// gather 3D positions into vector
		for (const auto& [id, vertexL] : vertexMapL)
		{
			if (vertexL->edges.empty())
				continue;

			if (!vertexL->fixed)
			{
				vertexL->iL = numL++;
				verticesL_.push_back(vertexL);
				Xws_.emplace_back(vertexL->Xw.data());
			}
			else
			{
				fixedVerticesL_.push_back(vertexL);
			}
		}

		numBody_ = numP;
		extJoint_ = is_joint_ext_optimization_enabled();
		verticesEJoint_.clear();

		// Reset iP on every ext vertex so subsequent kernels can quickly detect
		// whether a vertex is currently participating in the joint solve.
		for (const auto& [id, vertexE] : vertexMapE)
		{
			if (vertexE != nullptr) vertexE->iP = -1;
		}

		// In joint mode, append unfixed extrinsics vertices immediately after the
		// active body poses. Their iP lives in [numBody_, numBody_+numExt). They
		// share qs_/ts_/cameras_ layout with body poses so existing device sizing
		// logic applies unchanged, but are tracked separately in verticesEJoint_
		// for write-back (finalize writes them back to ExtrinsicsVertex::q/t).
		int numExt = 0;
		if (extJoint_)
		{
			for (const auto& [id, vertexE] : vertexMapE)
			{
				if (vertexE == nullptr) continue;
				if (vertexE->fixed) continue;
				if (vertexE->edges.empty()) continue;
				// Require at least one connected edge with a non-fixed landmark so
				// the ext iP participates in the Schur complement structure
				// (constructFromVertices only enumerates landmark-shared cross blocks).
				bool hasUnfixedLandmarkEdge = false;
				for (const auto e : vertexE->edges)
				{
					if (e->landmarkVertex() != nullptr && !e->landmarkVertex()->fixed)
					{
						hasUnfixedLandmarkEdge = true;
						break;
					}
				}
				if (!hasUnfixedLandmarkEdge) continue;

				vertexE->iP = numP++;
				verticesEJoint_.push_back(vertexE);
				qs_.emplace_back(vertexE->q.coeffs().data());
				ts_.emplace_back(vertexE->t.data());
				// Dummy camera slot for ext pose; kernels never dereference cameras[iP_ext].
				Vec5d dummyCam;
				dummyCam[0] = 0; dummyCam[1] = 0; dummyCam[2] = 0; dummyCam[3] = 0; dummyCam[4] = 0;
				cameras_.emplace_back(dummyCam);
				numExt++;
			}
		}
		numExt_ = numExt;

		numP_ = numP;
		numL_ = numL;
		optimizeP_ = numP_ > 0;
		optimizeL_ = numL_ > 0;

		// inactive(fixed) vertices are added after active vertices
		for (auto vertexP : fixedVerticesP_)
		{
			vertexP->iP = numP++;
			verticesP_.push_back(vertexP);
			qs_.emplace_back(vertexP->q.coeffs().data());
			ts_.emplace_back(vertexP->t.data());
			cameras_.emplace_back(vectorize(vertexP->camera));
		}

		for (auto vertexL : fixedVerticesL_)
		{
			vertexL->iL = numL++;
			verticesL_.push_back(vertexL);
			Xws_.emplace_back(vertexL->Xw.data());
		}

		// Dedup map for ext Hpl block structure. Multiple edges may share the same
		// (iP_ext, iL) pair (same camera observing the same landmark from different
		// keyframes); each dedup slot points to a single Hpl column entry so that
		// multiple edges atomically accumulate into the same Hpl block.
		std::map<std::pair<int,int>, int> ext_dedup_map;

		// gather each edge members into each vector
		int edgeId = 0, nedges2D = 0, nedges3D = 0;
		for (const auto e : edgeSet2D)
		{
			const auto vertexP = e->vertexP;
			const auto vertexL = e->vertexL;

			if (!vertexP->fixed && !vertexL->fixed)
				HplBlockPos_.push_back({ vertexP->iP, vertexL->iL, edgeId });

			if (!vertexP->fixed || !vertexL->fixed)
			{
				baseEdges_.push_back(e);
				measurements2D_.emplace_back(e->measurement.data());
				omegas_.push_back(ScalarCast(e->information));
				edge2PL_.push_back({ vertexP->iP, vertexL->iL });
				uint8_t flag = makeEdgeFlag(vertexP->fixed, vertexL->fixed);

				// Per-edge extrinsics: prefer vertexE if present (P1.1 scaffold),
				// else fall back to per-edge q_ext/t_ext.
				if (e->vertexE != nullptr)
				{
					q_exts_.emplace_back(e->vertexE->q.coeffs().data());
					t_exts_.emplace_back(e->vertexE->t.data());
				}
				else if (e->hasExtrinsics)
				{
					q_exts_.emplace_back(e->q_ext.coeffs().data());
					t_exts_.emplace_back(e->t_ext.data());
				}
				else
				{
					// Identity quaternion: [0, 0, 0, 1], zero translation.
					const double identity_q[4] = { 0.0, 0.0, 0.0, 1.0 };
					const double zero_t[3] = { 0.0, 0.0, 0.0 };
					q_exts_.emplace_back(identity_q);
					t_exts_.emplace_back(zero_t);
				}

				// Joint ext bookkeeping: record iP_ext and dedup Hpl slot when the
				// connected extrinsics vertex participates in the joint solve.
				// Default EDGE_FLAG_FIXED_E is set unless we establish an ext slot.
				flag |= EDGE_FLAG_FIXED_E;
					int iPExt = -1;
					int extDedupSlot = -1;
					if (extJoint_ && e->vertexE != nullptr && !e->vertexE->fixed && e->vertexE->iP >= 0)
					{
						iPExt = e->vertexE->iP;
						flag &= static_cast<uint8_t>(~EDGE_FLAG_FIXED_E);
						if (!vertexL->fixed)
						{
							const auto key = std::make_pair(iPExt, vertexL->iL);
							auto [it, inserted] = ext_dedup_map.insert({ key, static_cast<int>(ext_dedup_map.size()) });
							extDedupSlot = it->second;
							if (inserted)
								HplBlockPos_.push_back({ iPExt, vertexL->iL, -1 /* patched below */ });
						}
					}
					edgeFlags_.push_back(flag);
					edge2ExtIP_.push_back(iPExt);
					edge2HscPE_.push_back(-1);
					edge_to_ext_dedup_slot_.push_back(extDedupSlot);

				// Per-edge distortion coefficients (Kannala-Brandt equidistant model).
				{
					Vec4d dist;
					if (e->hasDistortion) {
						dist[0] = ScalarCast(e->distortion[0]);
						dist[1] = ScalarCast(e->distortion[1]);
						dist[2] = ScalarCast(e->distortion[2]);
						dist[3] = ScalarCast(e->distortion[3]);
					} else {
						dist[0] = 0; dist[1] = 0; dist[2] = 0; dist[3] = 0;
					}
					distortions_.emplace_back(dist);
				}

				edgeId++;
				nedges2D++;
			}
		}

		// gather each edge members into each vector
		for (const auto e : edgeSet3D)
		{
			const auto vertexP = e->vertexP;
			const auto vertexL = e->vertexL;

			if (!vertexP->fixed && !vertexL->fixed)
				HplBlockPos_.push_back({ vertexP->iP, vertexL->iL, edgeId });

			if (!vertexP->fixed || !vertexL->fixed)
			{
				baseEdges_.push_back(e);
				measurements3D_.emplace_back(e->measurement.data());
				omegas_.push_back(ScalarCast(e->information));
				edge2PL_.push_back({ vertexP->iP, vertexL->iL });
				uint8_t flag = makeEdgeFlag(vertexP->fixed, vertexL->fixed);

				// Per-edge extrinsics: prefer vertexE if present, else fall back to q_ext/t_ext.
				if (e->vertexE != nullptr)
				{
					q_exts_.emplace_back(e->vertexE->q.coeffs().data());
					t_exts_.emplace_back(e->vertexE->t.data());
				}
				else if (e->hasExtrinsics)
				{
					q_exts_.emplace_back(e->q_ext.coeffs().data());
					t_exts_.emplace_back(e->t_ext.data());
				}
				else
				{
					const double identity_q[4] = { 0.0, 0.0, 0.0, 1.0 };
					const double zero_t[3] = { 0.0, 0.0, 0.0 };
					q_exts_.emplace_back(identity_q);
					t_exts_.emplace_back(zero_t);
				}

				// Joint ext bookkeeping (same as 2D path).
				flag |= EDGE_FLAG_FIXED_E;
					int iPExt = -1;
					int extDedupSlot = -1;
					if (extJoint_ && e->vertexE != nullptr && !e->vertexE->fixed && e->vertexE->iP >= 0)
					{
						iPExt = e->vertexE->iP;
						flag &= static_cast<uint8_t>(~EDGE_FLAG_FIXED_E);
						if (!vertexL->fixed)
						{
							const auto key = std::make_pair(iPExt, vertexL->iL);
							auto [it, inserted] = ext_dedup_map.insert({ key, static_cast<int>(ext_dedup_map.size()) });
							extDedupSlot = it->second;
							if (inserted)
								HplBlockPos_.push_back({ iPExt, vertexL->iL, -1 });
						}
					}
					edgeFlags_.push_back(flag);
					edge2ExtIP_.push_back(iPExt);
					edge2HscPE_.push_back(-1);
					edge_to_ext_dedup_slot_.push_back(extDedupSlot);

				edgeId++;
				nedges3D++;
			}
		}

		// Patch ext HplBlockPos ids and count dedup slots. Body entries reuse the
		// edge-indexed id space [0, nedges); ext entries occupy [nedges, nedges + n_ext_dedup).
		nExtHplBlocks_ = static_cast<int>(ext_dedup_map.size());
		if (nExtHplBlocks_ > 0)
		{
			const int nedges_total = nedges2D + nedges3D;
			std::vector<int> slot_seen(nExtHplBlocks_, 0);
			int patch_cursor = 0;
			for (auto& bp : HplBlockPos_)
			{
				if (bp.id >= 0) continue;  // body entry: already has valid edgeId
				// Find the dedup slot this entry corresponds to by scanning the map.
				// Order of push matches insertion order of ext_dedup_map.
				bp.id = nedges_total + patch_cursor;
				patch_cursor++;
			}
		}

		nedges2D_ = nedges2D;
		nedges3D_ = nedges3D;
		nHplBlocks_ = static_cast<int>(HplBlockPos_.size());
		trace_cuda_ba(
			"solver initialize host graph prepared: active_poses=" + std::to_string(numP_) +
			", active_landmarks=" + std::to_string(numL_) +
			", total_poses=" + std::to_string(verticesP_.size()) +
			", total_landmarks=" + std::to_string(verticesL_.size()) +
			", base_edges=" + std::to_string(baseEdges_.size()) +
			", hpl_blocks=" + std::to_string(nHplBlocks_));

		// set robust kernels
		for (int i = 0; i < EDGE_TYPE_NUM; i++)
			kernels_[i] = kernels[i];

		// create sparse linear solver
		if (!linearSolver_)
			linearSolver_ = SparseLinearSolver::create();

		profItems_.assign(PROF_ITEM_NUM, 0);

		const auto t1 = std::chrono::steady_clock::now();
		trace_cuda_ba("solver initialize end");
		profItems_[PROF_ITEM_INITIALIZE] += get_duration(t0, t1);
	}

	void buildStructure()
	{
		trace_cuda_ba(
			"buildStructure begin: poses=" + std::to_string(numP_) +
			", landmarks=" + std::to_string(numL_) +
			", edges2d=" + std::to_string(nedges2D_) +
			", edges3d=" + std::to_string(nedges3D_));

		const auto t0 = get_time_point();

		// allocate device buffers
		d_x_.resize(numP_ * PDIM + numL_ * LDIM);
		d_b_.resize(numP_ * PDIM + numL_ * LDIM);

		if (optimizeP_)
		{
			d_xp_.map(numP_, d_x_.data());
			d_bp_.map(numP_, d_b_.data());
			d_Hpp_.resize(numP_);
			d_HppBackup_.resize(numP_);
		}

		// Option 4 Phase 2+: when deterministic accumulation is enabled and the
		// joint extrinsics path is active, allocate fixed-point int64 mirror
		// buffers for d_Hpp_ (Phase 2) and d_bp_ (Phase 3a). The kernel routes
		// ext-range atomicAdd writes into these buffers; the matching
		// convertFixedPoint*ExtRange() helpers then propagate the ext slots
		// back into the double buffers after each buildSystem(). Body-range
		// slots remain zero and are never read back.
		if (deterministicAccum_ && optimizeP_ && extJoint_ && numExt_ > 0)
		{
			d_Hpp_int_ext_.resize(static_cast<size_t>(d_Hpp_.elemSize()));
			d_bp_int_ext_.resize(static_cast<size_t>(d_bp_.elemSize()));
		}

		if (optimizeL_)
		{
			d_xl_.map(numL_, d_x_.data() + numP_ * PDIM);
			d_bl_.map(numL_, d_b_.data() + numP_ * PDIM);
			d_Hll_.resize(numL_);
			d_HllBackup_.resize(numL_);

			// Phase 3c: allocate fixed-point int64 mirror buffers for d_Hll_ and
			// d_bl_ whenever deterministic accumulation is requested. The gate is
			// independent of extJoint_ because landmark atomic sites accumulate
			// for every edge (not just ext-coupled ones).
			if (deterministicAccum_ && numL_ > 0)
			{
				d_Hll_int_.resize(static_cast<size_t>(d_Hll_.elemSize()));
				d_bl_int_.resize(static_cast<size_t>(d_bl_.elemSize()));
			}
		}

		if (optimizeP_ && optimizeL_)
		{
			// build Hpl block matrix structure
			d_Hpl_.resize(numP_, numL_);
			d_Hpl_.resizeNonZeros(nHplBlocks_);

			d_HplBlockPos_.assign(nHplBlocks_, HplBlockPos_.data());
			d_nnzPerCol_.resize(numL_ + 1);
			// indexPL now holds both body-edge slots [0, nedges) and ext dedup
			// slots [nedges, nedges + nExtHplBlocks_).
			const int nedges_total = static_cast<int>(baseEdges_.size());
			d_edge2Hpl_.resize(nedges_total + nExtHplBlocks_);

			gpu::buildHplStructure(d_HplBlockPos_, d_Hpl_, d_edge2Hpl_, d_nnzPerCol_);

			// build Hschur block matrix structure. Joint mode passes ext vertices
			// so shared landmarks produce body-ext and ext-ext cross blocks.
			Hsc_.resize(numP_, numP_);
			Hsc_.constructFromVertices(verticesL_);
			Hsc_.convertBSRToCSR();

				d_Hsc_.resize(numP_, numP_);
				d_Hsc_.resizeNonZeros(Hsc_.nblocks());
				d_Hsc_.upload(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());
				d_HscDirect_.resize(numP_, numP_);
				d_HscDirect_.resizeNonZeros(Hsc_.nblocks());
				d_HscDirect_.upload(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());

				// Phase 3d: allocate fixed-point int64 mirror for d_HscDirect_
				// when deterministic accumulation is active on the joint ext
				// path. The HscDirect structure is only populated inside this
				// `optimizeP_ && optimizeL_` block and only receives atomic
				// writes (no ASSIGN), so sizing from `d_HscDirect_.nnz()` is
				// safe here (after resizeNonZeros above).
				if (deterministicAccum_ && extJoint_ && numExt_ > 0 && d_HscDirect_.nnz() > 0)
				{
					const size_t nHsc = static_cast<size_t>(d_HscDirect_.nnz()) *
						static_cast<size_t>(PDIM) * static_cast<size_t>(PDIM);
					d_HscDirect_int_.resize(nHsc);
				}

				// Phase 3e: allocate fixed-point int64 mirror for d_Hpl_ when
				// deterministic accumulation is active and the joint ext path
				// has atomic-accumulated Hpl slots. The mirror shares the full
				// `Hpl.values()` layout (nnz * PDIM * LDIM), but only the
				// `nExtHplBlocks_` ext dedup slots are written — body slots
				// use ASSIGN in the double buffer and are never touched via
				// this int64 pointer. Sized from `d_Hpl_.nnz()` which was
				// established by `resizeNonZeros(nHplBlocks_)` above.
				if (deterministicAccum_ && extJoint_ && nExtHplBlocks_ > 0 && d_Hpl_.nnz() > 0)
				{
					const size_t nHpl = static_cast<size_t>(d_Hpl_.nnz()) *
						static_cast<size_t>(PDIM) * static_cast<size_t>(LDIM);
					d_Hpl_ext_int_.resize(nHpl);
				}

				// Phase 3f: allocate fixed-point int64 mirrors for the Schur
				// complement DEACCUM_ATOMIC sites (bsc in computeBschureKernel
				// and Hsc in computeHschureKernel). These run on every LM
				// iteration inside `solve()` so the gate is independent of
				// extJoint_ — both body-only and joint-ext paths benefit from
				// deterministic accumulation here. Sized from numP_ / d_Hsc_.
				if (deterministicAccum_ && numP_ > 0)
				{
					d_bsc_int_.resize(static_cast<size_t>(numP_) *
						static_cast<size_t>(PDIM));
				}
				if (deterministicAccum_ && d_Hsc_.nnz() > 0)
				{
					const size_t nHsc = static_cast<size_t>(d_Hsc_.nnz()) *
						static_cast<size_t>(PDIM) * static_cast<size_t>(PDIM);
					d_Hsc_int_.resize(nHsc);
				}

			d_HscCSR_.resize(Hsc_.nnzSymm());
			d_BSR2CSR_.assign(Hsc_.nnzSymm(), (int*)Hsc_.BSR2CSR());

			const int hplPairCapacity = hplPairEnumerationUpperBound();
			trace_cuda_ba(
				"Hschur multiply slots: hsc_unique_pairs=" + std::to_string(Hsc_.nmulBlocks()) +
				", hpl_pair_capacity=" + std::to_string(hplPairCapacity) +
				", hpl_blocks=" + std::to_string(nHplBlocks_));
			d_HscMulBlockIds_.resize(hplPairCapacity);
			gpu::findHschureMulBlockIndices(d_Hpl_, d_Hsc_, d_HscMulBlockIds_);

			d_bsc_.resize(numP_);
			d_invHll_.resize(numL_);
			d_Hpl_invHll_.resize(nHplBlocks_);

			d_edge2Hpl2D_.map(nedges2D_, d_edge2Hpl_.data());
			d_edge2Hpl3D_.map(nedges3D_, d_edge2Hpl_.data() + nedges2D_);

			// Joint-mode: compose d_edge2HplExt_ per edge by indexing
			// d_edge2Hpl_[nedges_total + slot_id]. We resize a dedicated array even
			// when nExt=0 so downstream kernel call signatures stay uniform.
			d_edge2HplExt_.resize(nedges_total);
			if (extJoint_ && nExtHplBlocks_ > 0)
			{
				gpu::buildEdgeExtHpl(edge_to_ext_dedup_slot_.data(),
					nedges_total, nExtHplBlocks_, d_edge2Hpl_, d_edge2HplExt_);
			}
			else
			{
				gpu::fillEdge2HplExtSentinel(d_edge2HplExt_, nedges_total);
			}
			d_edge2HplExt2D_.map(nedges2D_, d_edge2HplExt_.data());
			d_edge2HplExt3D_.map(nedges3D_, d_edge2HplExt_.data() + nedges2D_);

			// Upload per-edge ext iP (needed by kernel to atomic-add into Hpp[iP_ext] / bp[iP_ext]).
				if (!edge2ExtIP_.empty())
				{
					d_edge2ExtIP_.assign(static_cast<size_t>(nedges_total), edge2ExtIP_.data());
				}
				else
			{
				d_edge2ExtIP_.resize(nedges_total);
				gpu::fillEdge2HplExtSentinel(d_edge2ExtIP_, nedges_total);
				}
				d_edge2ExtIP2D_.map(nedges2D_, d_edge2ExtIP_.data());
				d_edge2ExtIP3D_.map(nedges3D_, d_edge2ExtIP_.data() + nedges2D_);

				// Build per-edge direct Hsc(body, ext) block slot. Joint mode appends ext
				// vertices after active body pose vertices, so body/ext cross terms always
				// live in the upper-triangular Hsc structure at row=min(iP, iPExt), col=max(...).
				edge2HscPE_.assign(static_cast<size_t>(nedges_total), -1);
				if (extJoint_)
				{
					const int* hscOuter = Hsc_.outerIndices();
					const int* hscInner = Hsc_.innerIndices();
					for (int edge_idx = 0; edge_idx < nedges_total; ++edge_idx)
					{
						const int iPExt = edge2ExtIP_[edge_idx];
						if (iPExt < 0)
							continue;

						const int iPBody = edge2PL_[edge_idx].P;
						if (iPBody < 0 || iPBody == iPExt)
							continue;

						const int row = std::min(iPBody, iPExt);
						const int col = std::max(iPBody, iPExt);
						for (int hidx = hscOuter[row]; hidx < hscOuter[row + 1]; ++hidx)
						{
							if (hscInner[hidx] == col)
							{
								edge2HscPE_[edge_idx] = hidx;
								break;
							}
						}
					}
				}
				d_edge2HscPE_.assign(static_cast<size_t>(nedges_total), edge2HscPE_.data());
				d_edge2HscPE2D_.map(nedges2D_, d_edge2HscPE_.data());
				d_edge2HscPE3D_.map(nedges3D_, d_edge2HscPE_.data() + nedges2D_);
			}

		// upload solutions to device memory
		// Include joint-mode ext vertices: qs_/ts_ hold all slots (body active + ext + body fixed).
		d_solution_.resize(qs_.size() * 7 + Xws_.size() * 3);
		d_solutionBackup_.resize(d_solution_.size());

		d_qs_.map(qs_.size(), d_solution_.data());
		d_ts_.map(ts_.size(), d_qs_.data() + d_qs_.size());
		d_Xws_.map(Xws_.size(), d_ts_.data() + d_ts_.size());

		d_qs_.upload(qs_.data());
		d_ts_.upload(ts_.data());
		d_Xws_.upload(Xws_.data());

		// upload edge information to device memory
		d_measurements2D_.assign(nedges2D_, measurements2D_.data());
		d_measurements3D_.assign(nedges3D_, measurements3D_.data());
		d_errors2D_.resize(nedges2D_);
		d_errors3D_.resize(nedges3D_);
		d_omegas2D_.assign(nedges2D_, omegas_.data());
		d_omegas3D_.assign(nedges3D_, omegas_.data() + nedges2D_);
		d_Xcs2D_.resize(nedges2D_);
		d_Xcs3D_.resize(nedges3D_);
		d_edge2PL2D_.assign(nedges2D_, edge2PL_.data());
		d_edge2PL3D_.assign(nedges3D_, edge2PL_.data() + nedges2D_);
		d_edgeFlags2D_.assign(nedges2D_, edgeFlags_.data());
		d_edgeFlags3D_.assign(nedges3D_, edgeFlags_.data() + nedges2D_);

		// upload per-edge extrinsics to device memory
		d_q_exts_2D_.assign(nedges2D_, q_exts_.data());
		d_q_exts_3D_.assign(nedges3D_, q_exts_.data() + nedges2D_);
		d_t_exts_2D_.assign(nedges2D_, t_exts_.data());
		d_t_exts_3D_.assign(nedges3D_, t_exts_.data() + nedges2D_);

		// upload per-edge distortion coefficients to device memory (2D only)
		d_distortions_2D_.assign(nedges2D_, distortions_.data());

		d_chi_.resize(1);
		// Option 4 Phase 3g: single-element int64 accumulator shared between
		// computeErrors and computeScale. Same allocation gate as the other
		// Phase 3 mirrors: only needed when the deterministic path is enabled.
		if (deterministicAccum_)
			d_chi_int_.resize(1);

		d_chiSqs_.resize(baseEdges_.size());
		d_chiSqs2D_.map(nedges2D_, d_chiSqs_.data());
		d_chiSqs3D_.map(nedges3D_, d_chiSqs_.data() + nedges2D_);

		// upload camera parameters to device memory
		d_cameras_.assign(cameras_.size(), cameras_.data());

		const auto t1 = get_time_point();

		// analyze pattern of Hschur matrix (symbolic decomposition)
		if (optimizeP_ && optimizeL_)
		{
			trace_cuda_ba("buildStructure symbolic initialize begin");
			linearSolver_->initialize(Hsc_);
			trace_cuda_ba("buildStructure symbolic initialize end");
		}

		const auto t2 = get_time_point();

		profItems_[PROF_ITEM_BUILD_STRUCTURE] += get_duration(t0, t1);
		profItems_[PROF_ITEM_DECOMP_SYMBOLIC] += get_duration(t1, t2);
	}

	double computeErrors()
	{
		const auto t0 = get_time_point();

		// Joint mode: refresh per-edge q_exts/t_exts from the (potentially updated)
		// qs_/ts_ slots so the projection kernel sees the current ext state.
		syncExtSolutionToPerEdge();

		// Option 4 Phase 3g: route the final chi2 reduction through the
		// deterministic int64 accumulator when the flag is active. The 2D and
		// 3D paths reuse the same single-element buffer; they run sequentially
		// so no interference. Passing nullptr preserves the legacy double path.
		long long* chi_int_ptr = deterministicAccum_ && d_chi_int_.size() > 0
			? d_chi_int_.data() : nullptr;

		const Scalar chi2D = gpu::computeActiveErrors(d_qs_, d_ts_, d_cameras_, d_Xws_, d_measurements2D_,
			d_omegas2D_, d_edge2PL2D_, d_q_exts_2D_, d_t_exts_2D_, d_distortions_2D_, kernels_[0], d_errors2D_, d_Xcs2D_, d_chi_, chi_int_ptr);

		const Scalar chi3D = gpu::computeActiveErrors(d_qs_, d_ts_, d_cameras_, d_Xws_, d_measurements3D_,
			d_omegas3D_, d_edge2PL3D_, d_q_exts_3D_, d_t_exts_3D_, kernels_[1], d_errors3D_, d_Xcs3D_, d_chi_, chi_int_ptr);

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_COMPUTE_ERROR] += get_duration(t0, t1);

		return chi2D + chi3D;
	}

	// Joint mode: scatter the current ext solution (qs_/ts_ at iP in [numBody_, numBody_+numExt_))
	// into the per-edge q_exts_/t_exts_ arrays that the error/Jacobian kernels read.
	// No-op when joint mode is OFF (d_edge2ExtIP_ is all -1) and when numExt_==0.
	void syncExtSolutionToPerEdge()
	{
		if (!extJoint_ || numExt_ == 0)
			return;
		gpu::syncExtSolutionToPerEdge(d_qs_, d_ts_, d_edge2ExtIP2D_, d_q_exts_2D_, d_t_exts_2D_, nedges2D_);
		gpu::syncExtSolutionToPerEdge(d_qs_, d_ts_, d_edge2ExtIP3D_, d_q_exts_3D_, d_t_exts_3D_, nedges3D_);
	}

	void buildSystem()
	{
		const auto t0 = get_time_point();

		////////////////////////////////////////////////////////////////////////////////////
		// Build linear system about solution increments Δx
		// H*Δx = -b
		// 
		// coefficient matrix are divided into blocks, and each block is calculated
		// | Hpp  Hpl ||Δxp| = |-bp|
		// | HplT Hll ||Δxl|   |-bl|
		////////////////////////////////////////////////////////////////////////////////////

		d_Hpp_.fillZero();
		d_Hll_.fillZero();
			d_bp_.fillZero();
			d_bl_.fillZero();
			// Joint mode: ext Hpl slots use ACCUM_ATOMIC (multiple edges may share a dedup slot).
			// We zero the entire Hpl nnz region so ext slots start at 0; body slots are safely
			// overwritten by ASSIGN on every edge so the zero has no effect on them.
			if (extJoint_ && nExtHplBlocks_ > 0)
				d_Hpl_.fillZero();
			d_HscDirect_.fillZero();

			// Option 4 Phase 2+: deterministic Hpp[iPExt] (Phase 2) and
			// bp[iPExt] (Phase 3a) accumulation. When the flag is on and the
			// joint path has unfixed ext vertices, we zero both int64 mirror
			// buffers, pass their raw pointers to the kernels, and after both
			// 2D/3D accumulate calls we convert the ext-range slots back into
			// d_Hpp_ / d_bp_. When off, the legacy double atomicAdd path is
			// used and this whole block is a no-op (nullptr is passed to the
			// kernel).
			const bool useDetAccum = deterministicAccum_ && extJoint_ && numExt_ > 0;
			long long* d_Hpp_int_ext_ptr = nullptr;
			long long* d_bp_int_ext_ptr = nullptr;
			if (useDetAccum)
			{
				d_Hpp_int_ext_.fillZero();
				d_bp_int_ext_.fillZero();
				d_Hpp_int_ext_ptr = d_Hpp_int_ext_.data();
				d_bp_int_ext_ptr = d_bp_int_ext_.data();
			}

			// Phase 3c: deterministic Hll[iL] + bl[iL] accumulation. Gated by
			// deterministicAccum_ && optimizeL_ && numL_>0 — this is independent
			// of extJoint_ because every edge accumulates into landmark slots.
			// Zero the int64 mirrors, forward the raw pointers to the kernels,
			// and after both 2D/3D calls convert the [0, numL) range back into
			// d_Hll_ / d_bl_. When the flag is off the legacy atomicAdd(double)
			// path is used and this block is a no-op.
			const bool useDetAccumL = deterministicAccum_ && optimizeL_ && numL_ > 0;
			long long* d_Hll_int_ptr = nullptr;
			long long* d_bl_int_ptr = nullptr;
			if (useDetAccumL)
			{
				d_Hll_int_.fillZero();
				d_bl_int_.fillZero();
				d_Hll_int_ptr = d_Hll_int_.data();
				d_bl_int_ptr = d_bl_int_.data();
			}

			// Phase 3d: deterministic HscDirect[hscPESlot] accumulation. Shares
			// the ext-joint gate with Phase 2/3a/3b since HscDirect is only
			// populated when joint extrinsics optimization is active.
			const bool useDetAccumHsc = useDetAccum && d_HscDirect_.nnz() > 0;
			long long* d_HscDirect_int_ptr = nullptr;
			if (useDetAccumHsc)
			{
				d_HscDirect_int_.fillZero();
				d_HscDirect_int_ptr = d_HscDirect_int_.data();
			}

			// Phase 3e: deterministic Hpl[hplExtSlot] accumulation for the ext
			// dedup slots of the cross-block Hpl. Gated by
			// deterministicAccum_ && extJoint_ && nExtHplBlocks_>0 so body-only
			// or decoupled runs continue to use the legacy atomic path. Body
			// Hpl slots remain on the ASSIGN path in the double buffer and are
			// not touched via the int64 mirror.
			const bool useDetAccumHpl = deterministicAccum_ && extJoint_
				&& nExtHplBlocks_ > 0 && d_Hpl_.nnz() > 0
				&& d_Hpl_ext_int_.size() > 0;
			long long* d_Hpl_ext_int_ptr = nullptr;
			if (useDetAccumHpl)
			{
				d_Hpl_ext_int_.fillZero();
				d_Hpl_ext_int_ptr = d_Hpl_ext_int_.data();
			}

			gpu::constructQuadraticForm(d_Xcs2D_, d_qs_, d_cameras_, d_errors2D_, d_omegas2D_, d_edge2PL2D_,
				d_edge2Hpl2D_, d_edge2HplExt2D_, d_edge2ExtIP2D_, d_edge2HscPE2D_, d_edgeFlags2D_, d_q_exts_2D_, d_t_exts_2D_, d_distortions_2D_, kernels_[0], d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_, d_HscDirect_, d_Hpp_int_ext_ptr, d_bp_int_ext_ptr, d_Hll_int_ptr, d_bl_int_ptr, d_HscDirect_int_ptr, d_Hpl_ext_int_ptr);

			gpu::constructQuadraticForm(d_Xcs3D_, d_qs_, d_cameras_, d_errors3D_, d_omegas3D_, d_edge2PL3D_,
				d_edge2Hpl3D_, d_edge2HplExt3D_, d_edge2ExtIP3D_, d_edge2HscPE3D_, d_edgeFlags3D_, d_q_exts_3D_, d_t_exts_3D_, kernels_[1], d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_, d_HscDirect_, d_Hpp_int_ext_ptr, d_bp_int_ext_ptr, d_Hll_int_ptr, d_bl_int_ptr, d_HscDirect_int_ptr, d_Hpl_ext_int_ptr);

			if (useDetAccum)
			{
				// Phase 3b: convert the body-range slots first (kernel writes
				// body iP accumulations to [0, numBody) when the buffers are
				// non-null). Phase 2/3a then propagates the ext-range slots
				// [numBody, numBody+numExt). Together they cover the full
				// [0, numBody+numExt) range and restore d_Hpp_ / d_bp_ to a
				// consistent double view.
				gpu::convertFixedPointHppBodyRange(d_Hpp_int_ext_ptr, d_Hpp_, numBody_);
				gpu::convertFixedPointBpBodyRange(d_bp_int_ext_ptr, d_bp_, numBody_);
				gpu::convertFixedPointHppExtRange(d_Hpp_int_ext_ptr, d_Hpp_, numBody_, numExt_);
				gpu::convertFixedPointBpExtRange(d_bp_int_ext_ptr, d_bp_, numBody_, numExt_);
			}
			if (useDetAccumL)
			{
				// Phase 3c: propagate landmark-range slots [0, numL) back into
				// the double Hll / bl buffers. Kept independent of useDetAccum
				// above so landmark determinism can engage even when joint ext
				// optimization is disabled.
				gpu::convertFixedPointHllRange(d_Hll_int_ptr, d_Hll_, numL_);
				gpu::convertFixedPointBlRange(d_bl_int_ptr, d_bl_, numL_);
			}
			if (useDetAccumHsc)
			{
				// Phase 3d: propagate the full HscDirect non-zero range back
				// into double. HscDirect has no ASSIGN writes so the entire
				// `[0, nnz * PDIM * PDIM)` range is converted.
				gpu::convertFixedPointHscDirect(d_HscDirect_int_ptr, d_HscDirect_, d_HscDirect_.nnz());
			}
			if (useDetAccumHpl)
			{
				// Phase 3e: propagate only the ext dedup slots back into
				// `d_Hpl_`. `d_edge2Hpl_` layout: first `nedges_total` entries
				// are per-edge body slot positions (written with ASSIGN and
				// already valid); the tail `nExtHplBlocks_` entries are the
				// ext dedup slots' global nnz positions, which is exactly the
				// index mapping the conversion kernel needs.
				const int nedges_total = nedges2D_ + nedges3D_;
				const int* extSlotGlobalIds = d_edge2Hpl_.data() + nedges_total;
				gpu::convertFixedPointHplExtSlots(d_Hpl_ext_int_ptr, d_Hpl_,
					extSlotGlobalIds, nExtHplBlocks_);
			}

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_BUILD_SYSTEM] += get_duration(t0, t1);
	}

	void setDeterministicAccum(bool enabled)
	{
		deterministicAccum_ = enabled;
	}

	// Called from optimize() after maxDiagonal() so the prior doesn't skew
	// the LM lambda seed. Runs every LM iteration (after each buildSystem).
	void applyExtPriorIfEnabled()
	{
		if (extJoint_ && numExt_ > 0)
		{
			gpu::addExtPrior(d_Hpp_, numBody_, numExt_,
				ScalarCast(kExtPriorLambdaRot), ScalarCast(kExtPriorLambdaTrans));
		}
	}

	double maxDiagonal()
	{
		DeviceBuffer<Scalar> d_buffer(16);
		const Scalar maxP = gpu::maxDiagonal(d_Hpp_, d_buffer);
		const Scalar maxL = gpu::maxDiagonal(d_Hll_, d_buffer);
		return std::max(maxP, maxL);
	}

	void setLambda(double lambda)
	{
		gpu::addLambda(d_Hpp_, ScalarCast(lambda), d_HppBackup_);
		gpu::addLambda(d_Hll_, ScalarCast(lambda), d_HllBackup_);
	}

	void restoreDiagonal()
	{
		gpu::restoreDiagonal(d_Hpp_, d_HppBackup_);
		gpu::restoreDiagonal(d_Hll_, d_HllBackup_);
	}

	bool solve()
	{
		if (optimizeP_ && optimizeL_)
		{
			trace_cuda_ba("solve begin: schur complement");
			const auto t0 = get_time_point();

			////////////////////////////////////////////////////////////////////////////////////
			// Schur complement
			// bSc = -bp + Hpl*Hll^-1*bl
			// HSc = Hpp - Hpl*Hll^-1*HplT
			////////////////////////////////////////////////////////////////////////////////////
			// Phase 3f: deterministic DEACCUM path for the Schur complement
			// update. Gates run unconditionally when the mirror buffers are
			// allocated (allocation itself is gated on deterministicAccum_
			// in buildStructure). Caller contract: zero `src_int` before the
			// kernel launch, then propagate additively into the double buffer
			// after the kernel completes.
			const bool useDetAccumBsc = deterministicAccum_ && d_bsc_int_.size() > 0;
			const bool useDetAccumHsc = deterministicAccum_ && d_Hsc_int_.size() > 0;
			long long* d_bsc_int_ptr = nullptr;
			long long* d_Hsc_int_ptr = nullptr;
			if (useDetAccumBsc)
			{
				d_bsc_int_.fillZero();
				d_bsc_int_ptr = d_bsc_int_.data();
			}
			if (useDetAccumHsc)
			{
				d_Hsc_int_.fillZero();
				d_Hsc_int_ptr = d_Hsc_int_.data();
			}

			gpu::computeBschure(d_bp_, d_Hpl_, d_Hll_, d_bl_, d_bsc_, d_invHll_, d_Hpl_invHll_,
				d_bsc_int_ptr);
				gpu::computeHschure(d_Hpp_, d_HscDirect_, d_Hpl_invHll_, d_Hpl_, d_HscMulBlockIds_, d_Hsc_,
					d_Hsc_int_ptr);

			if (useDetAccumBsc)
			{
				gpu::convertFixedPointBsc(d_bsc_int_ptr, d_bsc_, numP_);
			}
			if (useDetAccumHsc)
			{
				gpu::convertFixedPointHsc(d_Hsc_int_ptr, d_Hsc_, d_Hsc_.nnz());
			}
			trace_cuda_ba("solve schur complement end");

			const auto t1 = get_time_point();

			////////////////////////////////////////////////////////////////////////////////////
			// Solve linear equation about Δxp
			// HSc*Δxp = bp
			////////////////////////////////////////////////////////////////////////////////////
			trace_cuda_ba("solve convertHschureBSRToCSR begin");
			gpu::convertHschureBSRToCSR(d_Hsc_, d_BSR2CSR_, d_HscCSR_);
			trace_cuda_ba("solve convertHschureBSRToCSR end");
			trace_cuda_ba("solve linearSolver begin");
			const bool success = linearSolver_->solve(d_HscCSR_, d_bsc_.values(), d_xp_.values());
			trace_cuda_ba(std::string("solve linearSolver end: success=") + (success ? "true" : "false"));
			if (!success)
				return false;

			const auto t2 = get_time_point();

			////////////////////////////////////////////////////////////////////////////////////
			// Solve linear equation about Δxl
			// Hll*Δxl = -bl - HplT*Δxp
			////////////////////////////////////////////////////////////////////////////////////
			trace_cuda_ba("solve schurComplementPost begin");
			gpu::schurComplementPost(d_invHll_, d_bl_, d_Hpl_, d_xp_, d_xl_);
			trace_cuda_ba("solve schurComplementPost end");

			const auto t3 = get_time_point();
			profItems_[PROF_ITEM_SCHUR_COMPLEMENT] += (get_duration(t0, t1) + get_duration(t2, t3));
			profItems_[PROF_ITEM_DECOMP_NUMERICAL] += get_duration(t1, t2);
		}
		// pose only optimization
		else if (optimizeP_)
		{
			gpu::solveDiagonalSystem(d_Hpp_, d_bp_, d_xp_);
		}
		// landmark only optimization
		else
		{
			gpu::solveDiagonalSystem(d_Hll_, d_bl_, d_xl_);
		}

		return true;
	}

	void update()
	{
		const auto t0 = get_time_point();

		if (extJoint_ && numExt_ > 0)
		{
			// Joint mode: body slots update normally; ext slots get per-iteration
			// clamping inside the kernel.
			gpu::updatePoses(d_xp_, d_qs_, d_ts_, numBody_,
				ScalarCast(kExtMaxTranslationPerIter), ScalarCast(kExtMaxRotationPerIter));
		}
		else
		{
			gpu::updatePoses(d_xp_, d_qs_, d_ts_);
		}
		gpu::updateLandmarks(d_xl_, d_Xws_);

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_UPDATE] += get_duration(t0, t1);
	}

	double computeScale(double lambda)
	{
		// Option 4 Phase 3g: same int64 accumulator used for chi2 is reused for
		// the LM denominator `x.(lambda*x + b)`. `gpu::computeScale` now returns
		// the scalar directly (unified with `computeActiveErrors`'s signature)
		// so there is no host-side `download` here anymore.
		long long* scale_int_ptr = deterministicAccum_ && d_chi_int_.size() > 0
			? d_chi_int_.data() : nullptr;
		return gpu::computeScale(d_x_, d_b_, d_chi_, ScalarCast(lambda), scale_int_ptr);
	}

	void push()
	{
		d_solution_.copyTo(d_solutionBackup_);
	}

	void pop()
	{
		d_solutionBackup_.copyTo(d_solution_);
	}

	void finalize()
	{
		d_qs_.download(qs_.data());
		d_ts_.download(ts_.data());
		d_Xws_.download(Xws_.data());

		// Layout: qs_/ts_ = [active_body_poses | ext_joint_slots | fixed_body_poses].
		// verticesP_ only stores body poses (active first, then fixed), so we write
		// back body slots via verticesP_ with an offset that skips the ext slots.
		const size_t nActiveBody = static_cast<size_t>(numBody_);
		const size_t nExt = static_cast<size_t>(numExt_);

		// Active body poses: qs_[0..nActiveBody) ↔ verticesP_[0..nActiveBody).
		for (size_t i = 0; i < nActiveBody && i < verticesP_.size(); i++)
		{
			qs_[i].copyTo(verticesP_[i]->q.coeffs().data());
			ts_[i].copyTo(verticesP_[i]->t.data());
		}

		// Ext joint slots: qs_[nActiveBody..nActiveBody+nExt) ↔ verticesEJoint_[0..nExt).
		for (size_t i = 0; i < nExt && i < verticesEJoint_.size(); i++)
		{
			const size_t qIdx = nActiveBody + i;
			qs_[qIdx].copyTo(verticesEJoint_[i]->q.coeffs().data());
			ts_[qIdx].copyTo(verticesEJoint_[i]->t.data());
		}

		// Fixed body poses (pinned at start): qs_[nActiveBody+nExt..qs_.size())
		// ↔ verticesP_[nActiveBody..verticesP_.size()).
		for (size_t vpi = nActiveBody; vpi < verticesP_.size(); vpi++)
		{
			const size_t qIdx = nExt + vpi;  // shift by ext slot count
			if (qIdx >= qs_.size()) break;
			qs_[qIdx].copyTo(verticesP_[vpi]->q.coeffs().data());
			ts_[qIdx].copyTo(verticesP_[vpi]->t.data());
		}

		for (size_t i = 0; i < verticesL_.size(); i++)
			Xws_[i].copyTo(verticesL_[i]->Xw.data());
	}

	void getChiSqs(std::unordered_map<const BaseEdge*, double>& chiSqs)
	{
		chiSqs.clear();

		chiSqs_.resize(baseEdges_.size());

		// Joint mode: final solution resides in qs_/ts_; refresh q_exts_/t_exts_ so
		// per-edge chi-squared uses the optimized extrinsics.
		syncExtSolutionToPerEdge();

		// compute chi-squares
		gpu::computeChiSquares(d_qs_, d_ts_, d_cameras_, d_Xws_, d_measurements2D_,
			d_omegas2D_, d_edge2PL2D_, d_q_exts_2D_, d_t_exts_2D_, d_distortions_2D_, d_chiSqs2D_);
		gpu::computeChiSquares(d_qs_, d_ts_, d_cameras_, d_Xws_, d_measurements3D_,
			d_omegas3D_, d_edge2PL3D_, d_q_exts_3D_, d_t_exts_3D_, d_chiSqs3D_);
		d_chiSqs_.download(chiSqs_.data());

		for (size_t i = 0; i < chiSqs_.size(); i++)
			chiSqs[baseEdges_[i]] = chiSqs_[i];
	}

	void getTimeProfile(TimeProfile& prof) const
	{
		static const char* profileItemString[PROF_ITEM_NUM] =
		{
			"0: Initialize Optimizer",
			"1: Build Structure",
			"2: Compute Error",
			"3: Build System",
			"4: Schur Complement",
			"5: Symbolic Decomposition",
			"6: Numerical Decomposition",
			"7: Update Solution"
		};

		prof.clear();
		for (int i = 0; i < PROF_ITEM_NUM; i++)
			prof[profileItemString[i]] = profItems_[i];
	}

private:

	static inline uint8_t makeEdgeFlag(bool fixedP, bool fixedL)
	{
		uint8_t flag = 0;
		if (fixedP) flag |= EDGE_FLAG_FIXED_P;
		if (fixedL) flag |= EDGE_FLAG_FIXED_L;
		return flag;
	}

	int hplPairEnumerationUpperBound() const
	{
		// Schur multiply index generation enumerates every Hpl row-slot pair per
		// landmark column. This can exceed Hsc_.nmulBlocks(), which counts only
		// unique Schur row pairs after host-side deduplication.
		std::vector<int> entriesPerLandmark(static_cast<size_t>(std::max(numL_, 0)), 0);
		for (const auto& blockPos : HplBlockPos_)
		{
			if (blockPos.col < 0 || blockPos.col >= numL_)
			{
				throw std::runtime_error(
					"invalid Hpl block column: col=" + std::to_string(blockPos.col) +
					", numL=" + std::to_string(numL_));
			}
			entriesPerLandmark[static_cast<size_t>(blockPos.col)]++;
		}

		size_t capacity = 0;
		for (const int count : entriesPerLandmark)
		{
			const size_t n = static_cast<size_t>(count);
			capacity += n * (n + 1U) / 2U;
			if (capacity > static_cast<size_t>(std::numeric_limits<int>::max()))
			{
				throw std::runtime_error("Hpl pair enumeration exceeds int capacity");
			}
		}
		return static_cast<int>(capacity);
	}

	////////////////////////////////////////////////////////////////////////////////////
	// host buffers
	////////////////////////////////////////////////////////////////////////////////////

	// graph components
	std::vector<VertexP*> verticesP_;
	std::vector<VertexL*> verticesL_;
	std::vector<ExtrinsicsVertex*> verticesEJoint_;  // joint-mode unfixed ext vertices (iP in [numBody_, numBody_+numExt_))
	std::vector<BaseEdge*> baseEdges_;
	int numP_, numL_, nedges2D_, nedges3D_;
	bool optimizeP_, optimizeL_;

	// Joint-mode extrinsics optimization bookkeeping (Option A).
		int numBody_;            //!< number of unfixed body pose vertices (iP in [0, numBody_))
		int numExt_;             //!< number of unfixed ext vertices (iP in [numBody_, numBody_+numExt_))
		bool extJoint_;          //!< TRIORB_OPTIMIZE_EXTRINSICS_JOINT=1 and unfixed ext vertices present
		int nExtHplBlocks_;      //!< number of unique (iP_ext, iL) pairs contributing to Hpl
		std::vector<int> edge2ExtIP_;                  //!< per-edge iP of ext vertex (-1 if absent/fixed)
		std::vector<int> edge2HscPE_;                  //!< per-edge direct Hsc(body, ext) slot (-1 when absent/fixed)
		std::vector<int> edge_to_ext_dedup_slot_;      //!< per-edge dedup slot id (-1 if no ext contribution)

	// Option 4 Phase 2: deterministic atomic accumulation (joint-mode only).
		bool deterministicAccum_ = false;              //!< set via setDeterministicAccum()

	// solution vectors
	std::vector<Vec4d> qs_;
	std::vector<Vec3d> ts_;
	std::vector<Vec3d> Xws_;

	// edge information
	std::vector<Vec2d> measurements2D_;
	std::vector<Vec3d> measurements3D_;
	std::vector<Scalar> omegas_;
	std::vector<PLIndex> edge2PL_;
	std::vector<uint8_t> edgeFlags_;
	std::vector<Scalar> chiSqs_;

	// per-edge extrinsics (camera_from_body transform)
	std::vector<Vec4d> q_exts_;
	std::vector<Vec3d> t_exts_;

	// per-edge distortion coefficients [k1, k2, k3, k4] (2D edges only)
	std::vector<Vec4d> distortions_;

	// block matrices
	HplSparseBlockMatrix Hpl_;
	HschurSparseBlockMatrix Hsc_;
	SparseLinearSolver::Ptr linearSolver_;
	std::vector<HplBlockPos> HplBlockPos_;
	int nHplBlocks_;

	// camera parameters
	std::vector<Vec5d> cameras_;

	// robust kernels
	RobustKernel kernels_[EDGE_TYPE_NUM];

	////////////////////////////////////////////////////////////////////////////////////
	// device buffers
	////////////////////////////////////////////////////////////////////////////////////

	// solution vectors
	GpuVec1d d_solution_, d_solutionBackup_;
	GpuVec4d d_qs_;
	GpuVec3d d_ts_, d_Xws_;

	// edge information
	GpuVec3d d_Xcs2D_, d_Xcs3D_;
	GpuVec1d d_omegas2D_, d_omegas3D_;
	GpuVec2d d_measurements2D_, d_errors2D_;
	GpuVec3d d_measurements3D_, d_errors3D_;
	GpuVec2i d_edge2PL2D_, d_edge2PL3D_;
	GpuVec1b d_edgeFlags2D_, d_edgeFlags3D_;
	GpuVec1i d_edge2Hpl_, d_edge2Hpl2D_, d_edge2Hpl3D_;
	// Joint-mode ext Hpl slot per edge (-1 sentinel when no ext contribution).
		GpuVec1i d_edge2HplExt_, d_edge2HplExt2D_, d_edge2HplExt3D_;
		// Joint-mode ext iP per edge (-1 sentinel when ext is fixed or unused).
		GpuVec1i d_edge2ExtIP_, d_edge2ExtIP2D_, d_edge2ExtIP3D_;
		// Joint-mode direct Hsc(body, ext) slot per edge (-1 sentinel when absent/fixed).
		GpuVec1i d_edge2HscPE_, d_edge2HscPE2D_, d_edge2HscPE3D_;
		GpuVec1d d_chiSqs_, d_chiSqs2D_, d_chiSqs3D_;

	// per-edge extrinsics on device
	GpuVec4d d_q_exts_2D_, d_q_exts_3D_;
	GpuVec3d d_t_exts_2D_, d_t_exts_3D_;

	// per-edge distortion on device (2D only)
	GpuVec4d d_distortions_2D_;

	// solution increments Δx = [Δxp Δxl]
	GpuVec1d d_x_;
	GpuPx1BlockVec d_xp_;
	GpuLx1BlockVec d_xl_;

	// coefficient matrix of linear system
	// | Hpp  Hpl ||Δxp| = |-bp|
	// | HplT Hll ||Δxl|   |-bl|
	GpuPxPBlockVec d_Hpp_;
	// Option 4 Phase 2: fixed-point int64 mirror of d_Hpp_.values() used for
	// deterministic atomicAdd on the joint ext path. Allocated only when
	// deterministicAccum_ && extJoint_ && numExt_>0. Size == d_Hpp_.elemSize().
	DeviceBuffer<long long> d_Hpp_int_ext_;
	GpuLxLBlockVec d_Hll_;
	// Option 4 Phase 3c: fixed-point int64 mirror of d_Hll_.values() used for
	// deterministic atomicAdd on the landmark Hessian blocks. Allocated only
	// when deterministicAccum_ && optimizeL_ && numL_>0. Size ==
	// d_Hll_.elemSize() (numL_ * LDIM * LDIM).
	DeviceBuffer<long long> d_Hll_int_;
	GpuHplBlockMat d_Hpl_;
	// Option 4 Phase 3e: fixed-point int64 mirror of d_Hpl_.values() used for
	// deterministic atomicAdd on the ext-range slots of the cross-block Hpl
	// (`hplExtSlot` writes in `constructQuadraticFormKernel`). Allocated only
	// when deterministicAccum_ && extJoint_ && nExtHplBlocks_>0 &&
	// d_Hpl_.nnz()>0. Size == d_Hpl_.nnz() * PDIM * LDIM. Body Hpl slots
	// share the same buffer layout but remain untouched by the int64 path
	// (they use ASSIGN in the double buffer).
	DeviceBuffer<long long> d_Hpl_ext_int_;
	GpuVec3i d_HplBlockPos_;
	GpuVec1d d_b_;
	GpuPx1BlockVec d_bp_;
	// Option 4 Phase 3a: fixed-point int64 mirror of d_bp_.values() used for
	// deterministic atomicAdd on the joint ext gradient vector. Allocated only
	// when deterministicAccum_ && extJoint_ && numExt_>0. Size ==
	// d_bp_.elemSize().
	DeviceBuffer<long long> d_bp_int_ext_;
	GpuLx1BlockVec d_bl_;
	// Option 4 Phase 3c: fixed-point int64 mirror of d_bl_.values() used for
	// deterministic atomicAdd on the landmark gradient vector. Allocated only
	// when deterministicAccum_ && optimizeL_ && numL_>0. Size ==
	// d_bl_.elemSize() (numL_ * LDIM).
	DeviceBuffer<long long> d_bl_int_;
	GpuPx1BlockVec d_HppBackup_;
	GpuLx1BlockVec d_HllBackup_;

	// schur complement of the H matrix
	// HSc = Hpp - Hpl*inv(Hll)*HplT
	// bSc = -bp + Hpl*inv(Hll)*bl
		GpuHscBlockMat d_Hsc_;
		GpuHscBlockMat d_HscDirect_;
		// Option 4 Phase 3d: fixed-point int64 mirror of d_HscDirect_.values()
		// used for deterministic atomicAdd on the direct body×ext Schur cross
		// block. Allocated only when deterministicAccum_ && extJoint_ &&
		// numExt_>0 && d_HscDirect_.nnz()>0. Size ==
		// d_HscDirect_.nnz() * PDIM * PDIM.
		DeviceBuffer<long long> d_HscDirect_int_;
		// Option 4 Phase 3f: fixed-point int64 mirror of d_Hsc_.values() used
		// for deterministic DEACCUM_ATOMIC inside `computeHschureKernel`
		// (Schur-complement update `Hsc -= Hpl*invHll*HplT`). Allocated only
		// when deterministicAccum_ && d_Hsc_.nnz()>0. Unlike Phase 3d this is
		// active on both body-only and joint-ext paths (the Schur update runs
		// unconditionally during `solve()`). Size == d_Hsc_.nnz() * PDIM * PDIM.
		DeviceBuffer<long long> d_Hsc_int_;
		GpuPx1BlockVec d_bsc_;
		// Option 4 Phase 3f: fixed-point int64 mirror of d_bsc_.values() used
		// for deterministic DEACCUM_ATOMIC inside `computeBschureKernel`
		// (Schur RHS update `bsc -= Hpl*invHll*bl`). Allocated only when
		// deterministicAccum_ && numP_>0, independent of ext-joint path.
		// Size == numP_ * PDIM.
		DeviceBuffer<long long> d_bsc_int_;
	GpuLxLBlockVec d_invHll_;
	GpuPxLBlockVec d_Hpl_invHll_;
	GpuVec3i d_HscMulBlockIds_;

	// conversion matrix storage format BSR to CSR
	GpuVec1d d_HscCSR_;
	GpuVec1i d_BSR2CSR_;

	// camera parameters
	GpuVec5d d_cameras_;

	// temporary buffer
	DeviceBuffer<Scalar> d_chi_;
	// Option 4 Phase 3g: single-element int64 mirror of `d_chi_` used for the
	// deterministic final reduction in `computeActiveErrorsKernel` and
	// `computeScaleKernel`. Shared between the two call sites because they
	// never overlap in time (computeErrors runs before computeScale each LM
	// iteration). Allocated only when `deterministicAccum_` is active.
	DeviceBuffer<long long> d_chi_int_;
	GpuVec1i d_nnzPerCol_;

	////////////////////////////////////////////////////////////////////////////////////
	// statistics
	////////////////////////////////////////////////////////////////////////////////////

	std::vector<double> profItems_;
};

/** @brief Implementation of CudaBundleAdjustment.
*/
class CudaBundleAdjustmentImpl : public CudaBundleAdjustment
{
public:

	void addPoseVertex(VertexP* v) override
	{
		vertexMapP_.insert({ v->id, v });
	}

	void addLandmarkVertex(VertexL* v) override
	{
		vertexMapL_.insert({ v->id, v });
	}

	void addExtrinsicsVertex(ExtrinsicsVertex* v) override
	{
		vertexMapE_.insert({ v->id, v });
	}

	void addMonocularEdge(Edge2D* e) override
	{
		edges2D_.insert(e);

		e->vertexP->edges.insert(e);
		e->vertexL->edges.insert(e);
		if (e->vertexE != nullptr) {
			e->vertexE->edges.insert(e);
		}
	}

	void addStereoEdge(Edge3D* e) override
	{
		edges3D_.insert(e);

		e->vertexP->edges.insert(e);
		e->vertexL->edges.insert(e);
		if (e->vertexE != nullptr) {
			e->vertexE->edges.insert(e);
		}
	}

	VertexP* poseVertex(int id) const override
	{
		return vertexMapP_.at(id);
	}

	VertexL* landmarkVertex(int id) const override
	{
		return vertexMapL_.at(id);
	}

	ExtrinsicsVertex* extrinsicsVertex(int id) const override
	{
		auto it = vertexMapE_.find(id);
		return it == vertexMapE_.end() ? nullptr : it->second;
	}

	void removePoseVertex(PoseVertex* v) override
	{
		auto it = vertexMapP_.find(v->id);
		if (it == std::end(vertexMapP_))
			return;

		for (auto e : it->second->edges)
			removeEdge(e);

		vertexMapP_.erase(it);
	}

	void removeLandmarkVertex(LandmarkVertex* v) override
	{
		auto it = vertexMapL_.find(v->id);
		if (it == std::end(vertexMapL_))
			return;

		for (auto e : it->second->edges)
			removeEdge(e);

		vertexMapL_.erase(it);
	}

	void removeEdge(BaseEdge* e) override
	{
		auto vertexP = e->poseVertex();
		if (vertexP->edges.count(e))
			vertexP->edges.erase(e);

		auto vertexL = e->landmarkVertex();
		if (vertexL->edges.count(e))
			vertexL->edges.erase(e);

		if (auto vertexE = e->extrinsicsVertex()) {
			if (vertexE->edges.count(e))
				vertexE->edges.erase(e);
		}

		if (e->dim() == 2)
		{
			auto edge2D = reinterpret_cast<Edge2D*>(e);
			if (edges2D_.count(edge2D))
				edges2D_.erase(edge2D);
		}

		if (e->dim() == 3)
		{
			auto edge3D = reinterpret_cast<Edge3D*>(e);
			if (edges3D_.count(edge3D))
				edges3D_.erase(edge3D);
		}
	}

	size_t nposes() const override
	{
		return vertexMapP_.size();
	}

	size_t nlandmarks() const override
	{
		return vertexMapL_.size();
	}

	size_t nedges() const override
	{
		return edges2D_.size() + edges3D_.size();
	}

	void setRobustKernels(RobustKernelType kernelType, double delta, EdgeType edgeType)
	{
		kernels_[IntCast(edgeType)] = RobustKernel(IntCast(kernelType), delta);
	}

	void setDeterministicAccum(bool enabled) override
	{
		solver_.setDeterministicAccum(enabled);
	}

	void initialize() override
	{
		solver_.initialize(vertexMapP_, vertexMapL_, vertexMapE_, edges2D_, edges3D_, kernels_);

		stats_.clear();
	}

	void optimize(int niterations) override
	{
		const int maxq = 10;
		const double tau = 1e-5;

		double nu = 2;
		double lambda = 0;
		double F = 0;

		// Levenberg-Marquardt iteration
		for (int iteration = 0; iteration < niterations; iteration++)
		{
			trace_cuda_ba("optimize iteration begin: iteration=" + std::to_string(iteration));
			if (iteration == 0)
				solver_.buildStructure();

			const double iniF = solver_.computeErrors();
			F = iniF;
			trace_cuda_ba("optimize computeErrors end: iteration=" + std::to_string(iteration) +
				", initial_error=" + std::to_string(iniF));

			solver_.buildSystem();
			trace_cuda_ba("optimize buildSystem end: iteration=" + std::to_string(iteration));
			
			if (iteration == 0)
				lambda = tau * solver_.maxDiagonal();
			// Apply ext prior after maxDiagonal() so the LM lambda seed reflects
			// the data Hessian rather than the prior-augmented diagonal.
			solver_.applyExtPriorIfEnabled();
			trace_cuda_ba("optimize lambda prepared: iteration=" + std::to_string(iteration) +
				", lambda=" + std::to_string(lambda));

			int q = 0;
			double rho = -1;
			for (; q < maxq && rho < 0; q++)
			{
				trace_cuda_ba("optimize LM attempt begin: iteration=" + std::to_string(iteration) +
					", attempt=" + std::to_string(q) +
					", lambda=" + std::to_string(lambda));
				solver_.push();

				solver_.setLambda(lambda);

				const bool success = solver_.solve();
				trace_cuda_ba(std::string("optimize solver end: iteration=") + std::to_string(iteration) +
					", attempt=" + std::to_string(q) +
					", success=" + (success ? "true" : "false"));

				solver_.update();
				trace_cuda_ba("optimize update end: iteration=" + std::to_string(iteration) +
					", attempt=" + std::to_string(q));

				const double Fhat = solver_.computeErrors();
				const double scale = solver_.computeScale(lambda) + 1e-3;
				rho = success ? (F - Fhat) / scale : -1;
				trace_cuda_ba("optimize rho evaluated: iteration=" + std::to_string(iteration) +
					", attempt=" + std::to_string(q) +
					", Fhat=" + std::to_string(Fhat) +
					", scale=" + std::to_string(scale) +
					", rho=" + std::to_string(rho));

				if (rho > 0)
				{
					lambda *= clamp(attenuation(rho), 1./3, 2./3);
					nu = 2;
					F = Fhat;
					break;
				}
				else
				{
					lambda *= nu;
					nu *= 2;
					solver_.restoreDiagonal();
					solver_.pop();
				}
			}

			stats_.push_back({ iteration, F });

			if (q == maxq || rho <= 0 || !std::isfinite(lambda))
				break;
		}

		solver_.finalize();
		solver_.getChiSqs(chiSqs_);
		solver_.getTimeProfile(timeProfile_);
	}

	void clear() override
	{
		vertexMapP_.clear();
		vertexMapL_.clear();
		vertexMapE_.clear();
		edges2D_.clear();
		edges3D_.clear();
		stats_.clear();
	}

	const BatchStatistics& batchStatistics() const override
	{
		return stats_;
	}

	const TimeProfile& timeProfile() const override
	{
		return timeProfile_;
	}

	double chiSquared(const BaseEdge* e) const override
	{
		return chiSqs_.count(e) ? chiSqs_.at(e) : 0;
	}

	~CudaBundleAdjustmentImpl()
	{
		clear();
	}

private:

	static inline double attenuation(double x) { return 1 - std::pow(2 * x - 1, 3); }
	static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

	CudaBlockSolver solver_;
	VertexMapP vertexMapP_;
	VertexMapL vertexMapL_;
	VertexMapE vertexMapE_;
	EdgeSet2D edges2D_;
	EdgeSet3D edges3D_;
	RobustKernel kernels_[EDGE_TYPE_NUM];

	BatchStatistics stats_;
	TimeProfile timeProfile_;
	std::unordered_map<const BaseEdge*, double> chiSqs_;
};

CudaBundleAdjustment::Ptr CudaBundleAdjustment::create()
{
	return std::make_unique<CudaBundleAdjustmentImpl>();
}

CudaBundleAdjustment::~CudaBundleAdjustment()
{
}

} // namespace cuba
