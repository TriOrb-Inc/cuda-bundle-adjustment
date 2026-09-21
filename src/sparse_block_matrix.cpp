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

#include "sparse_block_matrix.h"

#include <algorithm>

namespace cuba
{

void HplSparseBlockMatrix::constructFromBlockPos(std::vector<HplBlockPos>& blockpos)
{
	Eigen::VectorXi& bcolPtr = outerIndices_;
	Eigen::VectorXi& browInd = innerIndices_;
	Eigen::VectorXi nnzPerCol;

	nnzPerCol.resize(bcols_);
	nnzPerCol.setZero();

	std::sort(std::begin(blockpos), std::end(blockpos), [](const HplBlockPos& lhs, const HplBlockPos& rhs)
	{
		return lhs.row < rhs.row;
	});

	for (const auto& pos : blockpos)
		nnzPerCol[pos.col]++;

	// set colPtr
	bcolPtr.resize(bcols_ + 1);
	bcolPtr[0] = 0;
	for (int c = 0; c < bcols_; c++)
		bcolPtr[c + 1] = bcolPtr[c] + nnzPerCol[c];
	nblocks_ = bcolPtr[bcols_];

	// set rowInd
	nnzPerCol = bcolPtr;
	browInd.resize(nblocks_);
	for (const auto& pos : blockpos)
		browInd[nnzPerCol[pos.col]++] = pos.row;
}

void HschurSparseBlockMatrix::constructFromVertices(const std::vector<VertexL*>& verticesL)
{
	constructFromVerticesAndRelativeEdges(verticesL, {});
}

void HschurSparseBlockMatrix::constructFromVerticesAndRelativeEdges(
	const std::vector<VertexL*>& verticesL,
	const std::vector<RelativePoseEdge*>& relativePoseEdges)
{
	struct BlockPos { int row, col; };

	Eigen::VectorXi& browPtr_ = outerIndices_;
	Eigen::VectorXi& bcolInd_ = innerIndices_;

	std::vector<uint8_t> map(brows_ * bcols_, 0);
	std::vector<int> indices;

	std::vector<BlockPos> blockpos;
	blockpos.reserve(brows_ * bcols_);

	int countmul = 0;
	for (auto vL : verticesL)
	{
		if (vL->fixed)
			continue;

		indices.clear();
		for (const auto e : vL->edges)
		{
			const auto vP = e->poseVertex();
			if (!vP->fixed)
				indices.push_back(vP->iP);
			// Joint ext mode: each edge's ext vertex (if unfixed) also contributes
			// a row/col in Hschur so body-ext and ext-ext cross blocks are allocated.
			// The legacy path (extrinsicsVertex returns nullptr or vE->fixed==true)
			// naturally skips this branch and keeps byte-identical behavior.
			const auto vE = e->extrinsicsVertex();
			if (vE != nullptr && !vE->fixed && vE->iP >= 0)
				indices.push_back(vE->iP);
		}

		std::sort(std::begin(indices), std::end(indices));
		// Hpl uses one row per unique (iP, landmark) pair. Keep Hschur on the
		// same deduplicated row set so nmultiplies_ matches the actual row-pair
		// enumeration used by findHschureMulBlockIndices.
		indices.erase(std::unique(std::begin(indices), std::end(indices)), std::end(indices));
		const int nindices = static_cast<int>(indices.size());
		for (int i = 0; i < nindices; i++)
		{
			const int rowId = indices[i];
			uint8_t* ptrMap = map.data() + rowId * bcols_;
			for (int j = i; j < nindices; j++)
			{
				const int colId = indices[j];
				if (!ptrMap[colId])
				{
					blockpos.push_back({ rowId, colId });
					ptrMap[colId] = 1;
				}

				countmul++;
			}
		}
	}

	nmultiplies_ = countmul;

	for (auto edge : relativePoseEdges)
	{
		if (edge == nullptr || edge->fromVertex == nullptr || edge->toVertex == nullptr)
			continue;
		const auto from = edge->fromVertex;
		const auto to = edge->toVertex;
		if (from->fixed && to->fixed)
			continue;
		if (!from->fixed && from->iP >= 0)
		{
			const int rowId = from->iP;
			uint8_t* ptrMap = map.data() + rowId * bcols_;
			if (!ptrMap[rowId])
			{
				blockpos.push_back({ rowId, rowId });
				ptrMap[rowId] = 1;
			}
		}
		if (!to->fixed && to->iP >= 0)
		{
			const int rowId = to->iP;
			uint8_t* ptrMap = map.data() + rowId * bcols_;
			if (!ptrMap[rowId])
			{
				blockpos.push_back({ rowId, rowId });
				ptrMap[rowId] = 1;
			}
		}
		if (!from->fixed && !to->fixed && from->iP >= 0 && to->iP >= 0 && from->iP != to->iP)
		{
			const int rowId = std::min(from->iP, to->iP);
			const int colId = std::max(from->iP, to->iP);
			uint8_t* ptrMap = map.data() + rowId * bcols_;
			if (!ptrMap[colId])
			{
				blockpos.push_back({ rowId, colId });
				ptrMap[colId] = 1;
			}
		}
	}

	// joint-ext mode で、固定 landmark 経由の (pose, ext) cross block も確保する。
	//
	// 上の landmark loop は `vL->fixed` を弾くので、固定 landmark としか繋がらない
	// (pose, ext) の組は Hschur に cross block を持たない。一方
	// cuda_bundle_adjustment.cpp の edge2HscPE_ 解決は「pose と ext を同時に持つ edge には
	// 必ず (min, max) の cross block がある」と前提し、見つからなければ黙って -1 を通す。
	// その edge の Jp^T Omega Je は HscDirect へ積まれず消える一方、Hpp / bp には
	// ext 寄与が入るので、Schur 系が右辺と整合しなくなる。
	//
	// ext vertex 単位の空 row は cuda_bundle_adjustment.cpp:393-405 の
	// hasUnfixedLandmarkEdge guard が塞いでいるが、(pose, ext) pair 単位は塞げていない。
	// ここで固定 landmark 経由の組を seed して穴を埋める。
	//
	// nmultiplies_ は増やさない。固定 landmark は Hpl に row を持たないので
	// Schur multiply は発生しない。既に cross block がある通常ケースでは 1 block も増えない。
	for (auto vL : verticesL)
	{
		if (!vL->fixed)
			continue;

		for (const auto e : vL->edges)
		{
			const auto vP = e->poseVertex();
			const auto vE = e->extrinsicsVertex();
			if (vP == nullptr || vE == nullptr)
				continue;
			if (vP->fixed || vE->fixed)
				continue;
			if (vP->iP < 0 || vE->iP < 0 || vP->iP == vE->iP)
				continue;

			const int rowId = std::min(vP->iP, vE->iP);
			const int colId = std::max(vP->iP, vE->iP);
			if (rowId < 0 || colId >= bcols_)
				continue;

			uint8_t* ptrMap = map.data() + rowId * bcols_;
			if (!ptrMap[colId])
			{
				blockpos.push_back({ rowId, colId });
				ptrMap[colId] = 1;
			}
		}
	}

	// 有効な pose row には必ず対角 block を確保する。
	//
	// ここまでで対角 block が入るのは「非固定 landmark を共有する pose」と
	// 「relative pose edge の端点」だけである。固定 landmark としか繋がらない
	// pose は row が空のまま残り、次の 2 つが同時に壊れる。
	//
	//   1. initializeHschurKernel は全 row で無条件に
	//      `Hsc.at(HscRowPtr[rowId])` へ Hpp を書く。空 row では
	//      `HscRowPtr[rowId] == HscRowPtr[rowId + 1]` なので、その pose の Hpp が
	//      隣の row の先頭 block を上書きする (静かな数値破壊)。
	//   2. nnzSymm() は対角 block 数を brows_ で決め打ちする。空 row があると
	//      戻り値が真値より小さくなり、convertBSRToCSR() の colInd_ / BSR2CSR_ が
	//      溢れる (5 camera + 固定 landmark のみの keyframe で実測: brows=3 /
	//      nblocks=3 / ndiag=2 のとき idx=108 に対し size=108)。
	//
	// 対角 block は Hpp をそのまま持つので数値的にも正しく、既に対角がある
	// 通常ケースでは 1 block も追加されない。
	for (int rowId = 0; rowId < brows_; rowId++)
	{
		uint8_t* ptrMap = map.data() + rowId * bcols_;
		if (!ptrMap[rowId])
		{
			blockpos.push_back({ rowId, rowId });
			ptrMap[rowId] = 1;
		}
	}

	// set nonzero blocks
	nblocks_ = static_cast<int>(blockpos.size());

	std::sort(std::begin(blockpos), std::end(blockpos), [](const BlockPos& lhs, const BlockPos& rhs)
	{
		return lhs.col < rhs.col;
	});

	// set rowPtr
	nnzPerRow_.resize(brows_);
	nnzPerRow_.setZero();
	for (int i = 0; i < nblocks_; i++)
		nnzPerRow_[blockpos[i].row]++;

	browPtr_.resize(brows_ + 1);
	browPtr_[0] = 0;
	for (int r = 0; r < brows_; r++)
		browPtr_[r + 1] = browPtr_[r] + nnzPerRow_[r];

	// set colInd
	nnzPerRow_ = browPtr_;
	bcolInd_.resize(nblocks_);
	for (int i = 0; i < nblocks_; i++)
	{
		const int rowId = blockpos[i].row;
		const int colId = blockpos[i].col;
		const int k = nnzPerRow_[rowId]++;
		bcolInd_[k] = colId;
	}
}

void HschurSparseBlockMatrix::convertBSRToCSR()
{
	const int PDIM = BLOCK_ROWS;
	const int nnz = nnzSymm();
	const int drows = rows();

	Eigen::VectorXi& browPtr_ = outerIndices_;
	Eigen::VectorXi& bcolInd_ = innerIndices_;

	rowPtr_.resize(drows + 1);
	colInd_.resize(nnz);
	BSR2CSR_.resize(nnz);

	nnzPerRow_.resize(drows);
	nnzPerRow_.setZero();

	for (int blockRowId = 0; blockRowId < brows_; blockRowId++)
	{
		for (int i = browPtr_[blockRowId]; i < browPtr_[blockRowId + 1]; i++)
		{
			const int blockColId = bcolInd_[i];
			const int dstColId0 = blockColId * PDIM;
			const int dstRowId0 = blockRowId * PDIM;
			if (blockRowId == blockColId)
			{
				for (int dr = 0; dr < PDIM; dr++)
					nnzPerRow_[dstRowId0 + dr] += PDIM;
			}
			else
			{
				for (int dr = 0; dr < PDIM; dr++)
					nnzPerRow_[dstRowId0 + dr] += PDIM;
				for (int dc = 0; dc < PDIM; dc++)
					nnzPerRow_[dstColId0 + dc] += PDIM;
			}
		}
	}

	rowPtr_[0] = 0;
	for (int r = 0; r < drows; r++)
		rowPtr_[r + 1] = rowPtr_[r] + nnzPerRow_[r];

	for (int r = 0; r < drows; r++)
		nnzPerRow_[r] = rowPtr_[r];

	for (int blockRowId = 0; blockRowId < brows_; blockRowId++)
	{
		for (int i = browPtr_[blockRowId]; i < browPtr_[blockRowId + 1]; i++)
		{
			const int blockColId = bcolInd_[i];
			const int dstColId0 = blockColId * PDIM;
			const int dstRowId0 = blockRowId * PDIM;
			int srck = i * PDIM * PDIM;

			if (blockRowId == blockColId)
			{
				for (int dc = 0; dc < PDIM; ++dc)
				{
					for (int dr = 0; dr < PDIM; ++dr)
					{
						const int colId = dstColId0 + dc;
						const int rowId = dstRowId0 + dr;
						const int dstk = nnzPerRow_[rowId]++;
						colInd_[dstk] = colId;
						BSR2CSR_[dstk] = srck;
						srck++;
					}
				}
			}
			else
			{
				for (int dc = 0; dc < PDIM; ++dc)
				{
					for (int dr = 0; dr < PDIM; ++dr)
					{
						const int colId0 = dstColId0 + dc;
						const int rowId0 = dstRowId0 + dr;

						const int colId1 = rowId0;
						const int rowId1 = colId0;

						const int dstk0 = nnzPerRow_[rowId0]++;
						const int dstk1 = nnzPerRow_[rowId1]++;

						colInd_[dstk0] = colId0;
						colInd_[dstk1] = colId1;
						BSR2CSR_[dstk0] = srck;
						BSR2CSR_[dstk1] = srck;
						srck++;
					}
				}
			}
		}
	}
}

} // namespace cuba
