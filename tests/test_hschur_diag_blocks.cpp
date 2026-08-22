/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0
*/

// -----------------------------------------------------------------------------
// test_hschur_diag_blocks.cpp
//
// `HschurSparseBlockMatrix` の row が 1 つでも対角 block を欠くと、2 箇所が同時に壊れる。
//
//   1. `initializeHschurKernel` は全 row で無条件に
//      `Hsc.at(HscRowPtr[rowId])` へ `Hpp[rowId]` を書き込む。空 row では
//      `HscRowPtr[rowId] == HscRowPtr[rowId + 1]` なので、その pose の `Hpp` が
//      隣の row の先頭 block を上書きする。assertion では止まらない数値破壊になる。
//   2. `nnzSymm()` は対角 block 数を `brows_` で決め打ちする
//      (`(2 * nblocks_ - brows_) * BLOCK_AREA`)。対角の無い row があると戻り値が
//      真値より小さくなり、`convertBSRToCSR()` の `colInd_` / `BSR2CSR_` が溢れる。
//
// 対角を欠く row は、非固定 landmark を 1 つも持たない pose (固定 landmark としか
// 繋がらない pose) で発生する。`constructFromVerticesAndRelativeEdges` は
// 非固定 landmark の共有関係と relative pose edge からしか block を作らないためである。
//
// 実測: 5 camera bag (odom+cam0-4) の local BA で
// `brows=3 nblocks=3 ndiag=2 empty_rows=1` となり、
// `nnz_assumed=108` に対し真値 144、`colInd_[108]` (size 108) で Eigen assert が発火した。
//
// Build:
//   g++ -std=c++17 -I src -I include -I /usr/include/eigen3 \
//       tests/test_hschur_diag_blocks.cpp src/sparse_block_matrix.cpp \
//       -o build/test_hschur_diag_blocks
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_bundle_adjustment_types.h"
#include "sparse_block_matrix.h"

using cuba::BaseEdge;
using cuba::HschurSparseBlockMatrix;
using cuba::LandmarkVertex;
using cuba::PoseVertex;
using cuba::RelativePoseEdge;

namespace
{

int g_failures = 0;

void expectEq(const char* what, int actual, int expected)
{
	if (actual != expected)
	{
		std::printf("  FAIL %-28s actual=%d expected=%d\n", what, actual, expected);
		g_failures++;
	}
	else
	{
		std::printf("  ok   %-28s = %d\n", what, actual);
	}
}

// 構造だけを見る最小の edge。値は使わない。
struct StubEdge : BaseEdge
{
	PoseVertex* vP = nullptr;
	LandmarkVertex* vL = nullptr;

	PoseVertex* poseVertex() const override { return vP; }
	LandmarkVertex* landmarkVertex() const override { return vL; }
	int dim() const override { return 2; }
};

void connect(StubEdge& e, PoseVertex& p, LandmarkVertex& l)
{
	e.vP = &p;
	e.vL = &l;
	p.edges.insert(&e);
	l.edges.insert(&e);
}

// 対角 block を持つ row の数を数える。
int countDiagonalRows(const HschurSparseBlockMatrix& Hsc)
{
	const int* rowPtr = Hsc.outerIndices();
	const int* colInd = Hsc.innerIndices();
	int ndiag = 0;
	for (int r = 0; r < Hsc.brows(); r++)
	{
		for (int i = rowPtr[r]; i < rowPtr[r + 1]; i++)
		{
			if (colInd[i] == r) { ndiag++; break; }
		}
	}
	return ndiag;
}

} // namespace

int main()
{
	// pose0 - L0 - pose1 は非固定 landmark を共有する。
	// pose2 は固定 landmark L1 としか繋がらないので、修正前は row 2 が空になる。
	PoseVertex p0, p1, p2;
	p0.iP = 0; p1.iP = 1; p2.iP = 2;
	p0.fixed = p1.fixed = p2.fixed = false;

	LandmarkVertex l0, l1;
	l0.fixed = false;
	l1.fixed = true;

	StubEdge e00, e10, e21;
	connect(e00, p0, l0);
	connect(e10, p1, l0);
	connect(e21, p2, l1);

	const std::vector<LandmarkVertex*> verticesL{&l0, &l1};
	const std::vector<RelativePoseEdge*> relativePoseEdges{};

	HschurSparseBlockMatrix Hsc;
	Hsc.resize(3, 3);
	Hsc.constructFromVerticesAndRelativeEdges(verticesL, relativePoseEdges);

	std::printf("[case] 固定 landmark としか繋がらない pose が 1 つある局面\n");

	// (0,0) (0,1) (1,1) に加えて (2,2) が確保されること。
	expectEq("nblocks", Hsc.nblocks(), 4);
	expectEq("rows with diagonal", countDiagonalRows(Hsc), Hsc.brows());

	// nnzSymm() の (2 * nblocks_ - brows_) が真値と一致すること。
	const int ndiag = countDiagonalRows(Hsc);
	const int nnzTrue = (2 * Hsc.nblocks() - ndiag) * HschurSparseBlockMatrix::BLOCK_AREA;
	expectEq("nnzSymm", Hsc.nnzSymm(), nnzTrue);

	// 修正前はここで colInd_ が溢れて Eigen assert が発火した。
	Hsc.convertBSRToCSR();
	expectEq("rowPtr[rows]", Hsc.rowPtr()[Hsc.rows()], Hsc.nnzSymm());

	// 全 row が非空であること (initializeHschurKernel の書き込み先が重複しない条件)。
	const int* rowPtr = Hsc.outerIndices();
	int emptyRows = 0;
	for (int r = 0; r < Hsc.brows(); r++)
		if (rowPtr[r] == rowPtr[r + 1]) emptyRows++;
	expectEq("empty block rows", emptyRows, 0);

	// 既に全 row が対角を持つ通常の局面では block を 1 つも増やさないこと。
	{
		PoseVertex q0, q1;
		q0.iP = 0; q1.iP = 1;
		LandmarkVertex m0;
		m0.fixed = false;
		StubEdge f0, f1;
		connect(f0, q0, m0);
		connect(f1, q1, m0);

		const std::vector<LandmarkVertex*> verticesL2{&m0};
		HschurSparseBlockMatrix Hsc2;
		Hsc2.resize(2, 2);
		Hsc2.constructFromVerticesAndRelativeEdges(verticesL2, relativePoseEdges);

		std::printf("[case] 全 pose が非固定 landmark を共有する通常の局面\n");
		expectEq("nblocks", Hsc2.nblocks(), 3);            // (0,0) (0,1) (1,1)
		expectEq("rows with diagonal", countDiagonalRows(Hsc2), Hsc2.brows());
		Hsc2.convertBSRToCSR();
		expectEq("rowPtr[rows]", Hsc2.rowPtr()[Hsc2.rows()], Hsc2.nnzSymm());
	}

	if (g_failures != 0)
	{
		std::printf("FAILED: %d\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("PASSED\n");
	return EXIT_SUCCESS;
}
