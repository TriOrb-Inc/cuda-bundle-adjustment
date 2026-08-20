/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0
*/

// -----------------------------------------------------------------------------
// test_sym3x3_inv.cu
//
// LM adds lambda to the landmark block diagonal, so `Sym3x3Inv` is asked to
// invert matrices whose entries span many orders of magnitude. The shipping
// build uses `Scalar = float`, where the unscaled determinant overflows:
//
//   lambda >= cbrt(FLT_MAX) = 6.98e12   det -> Inf, invDet -> 0, B -> 0
//                                       (landmark update silently frozen)
//   lambda >= sqrt(FLT_MAX) = 1.85e19   cofactors overflow, 0 * Inf -> NaN
//
// A real 4-camera local BA hit lambda = 3.93e19 and every landmark position
// came back NaN. This test pins the float result to the double result across
// that whole range, so a regression to the unscaled form is caught.
//
// Build:
//   nvcc -std=c++17 -I../src tests/test_sym3x3_inv.cu -o build/test_sym3x3_inv
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <vector>

#include "sym3x3_inv.cuh"

namespace
{

// Landmark Hessian block shape seen in practice, plus the LM damping term.
template <typename S>
S invertDiagonalEntry(double lambda)
{
	S B00 = 0, B01 = 0, B02 = 0, B11 = 0, B12 = 0, B22 = 0;
	const S diagonal = static_cast<S>(1e4 + lambda);
	const S offDiagonal = static_cast<S>(1e3);
	cuba::gpu::sym3x3InvComponents<S>(
		diagonal, offDiagonal, offDiagonal, diagonal, offDiagonal, diagonal,
		&B00, &B01, &B02, &B11, &B12, &B22);
	return B00;
}

bool testFloatMatchesDoubleAcrossLambdaRange()
{
	// Spans below, at, and far past both float overflow thresholds.
	const std::vector<double> lambdas = {
		1e0, 1e6, 1e12,
		6.98e12,   // cbrt(FLT_MAX): unscaled det overflows from here
		1e15,
		1.85e19,   // sqrt(FLT_MAX): unscaled cofactors overflow from here
		3.93e19,   // the value measured when every landmark turned NaN
		5.15e24,
		1e30,
	};

	bool ok = true;
	for (const double lambda : lambdas) {
		const float actual = invertDiagonalEntry<float>(lambda);
		const double expected = invertDiagonalEntry<double>(lambda);

		if (!std::isfinite(actual)) {
			std::printf("  FAIL lambda=%-10.3g float B00 is not finite (%g)\n", lambda, (double) actual);
			ok = false;
			continue;
		}
		if (actual == 0.0f && expected != 0.0) {
			std::printf("  FAIL lambda=%-10.3g float B00 collapsed to 0 (double gives %g)\n",
			            lambda, expected);
			ok = false;
			continue;
		}
		// float carries ~7 significant digits; allow a generous relative margin.
		const double relative_error = std::fabs((double) actual - expected) / std::fabs(expected);
		if (relative_error > 1e-4) {
			std::printf("  FAIL lambda=%-10.3g float=%-14.7g double=%-14.7g rel_err=%.3g\n",
			            lambda, (double) actual, expected, relative_error);
			ok = false;
			continue;
		}
		std::printf("  ok   lambda=%-10.3g float=%-14.7g double=%-14.7g rel_err=%.3g\n",
		            lambda, (double) actual, expected, relative_error);
	}
	return ok;
}

// The inverse must still be correct for ordinary, well-scaled blocks.
bool testInverseIsCorrectForOrdinaryBlock()
{
	double B00 = 0, B01 = 0, B02 = 0, B11 = 0, B12 = 0, B22 = 0;
	// A = [[4,1,0],[1,3,1],[0,1,2]]
	cuba::gpu::sym3x3InvComponents<double>(4, 1, 0, 3, 1, 2, &B00, &B01, &B02, &B11, &B12, &B22);

	// A * B should be the identity; check the first row.
	const double row0_col0 = 4 * B00 + 1 * B01 + 0 * B02;
	const double row0_col1 = 4 * B01 + 1 * B11 + 0 * B12;
	const double row0_col2 = 4 * B02 + 1 * B12 + 0 * B22;

	const bool ok = std::fabs(row0_col0 - 1.0) < 1e-12 &&
	                std::fabs(row0_col1) < 1e-12 &&
	                std::fabs(row0_col2) < 1e-12;
	std::printf("  %s A*inv(A) row0 = [%g, %g, %g]\n", ok ? "ok  " : "FAIL",
	            row0_col0, row0_col1, row0_col2);
	return ok;
}

}  // namespace

int main()
{
	int failures = 0;

	std::printf("[1/2] float matches double across the LM lambda range\n");
	if (!testFloatMatchesDoubleAcrossLambdaRange()) {
		++failures;
	}

	std::printf("\n[2/2] inverse is correct for an ordinary block\n");
	if (!testInverseIsCorrectForOrdinaryBlock()) {
		++failures;
	}

	std::printf("\n[test_sym3x3_inv] %s (failures=%d)\n", failures == 0 ? "PASS" : "FAIL", failures);
	return failures == 0 ? 0 : 1;
}
