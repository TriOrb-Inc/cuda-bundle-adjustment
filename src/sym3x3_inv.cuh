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

#ifndef __SYM3X3_INV_CUH__
#define __SYM3X3_INV_CUH__

#include <cmath>

namespace cuba
{
namespace gpu
{

// Inverse of a symmetric 3x3 matrix, scaled so the determinant cannot overflow.
//
// The straightforward form computes `det = A00*A11*A22 - ...` from the raw
// entries. The shipping build uses `Scalar = float` (slam-core forces
// USE_FLOAT32 ON for CUDA compatibility; the double build does not compile for
// sm52), and LM adds lambda to this block's diagonal, so the entries can grow
// far beyond the reprojection-scale values the formula was written for:
//
//   lambda >= cbrt(FLT_MAX) = 6.98e12   det overflows to Inf, invDet -> 0,
//                                       B -> 0. The landmark update silently
//                                       freezes; nothing reports an error.
//   lambda >= sqrt(FLT_MAX) = 1.85e19   the cofactors overflow too, 0 * Inf
//                                       -> NaN. Every landmark in the block
//                                       gets a NaN position.
//
// Measured on a real 4-camera local BA: lambda reached 3.93e19 and all 542
// landmark positions came back NaN, which then propagated into a saved map.
// The same block inverts without trouble in double, so this is a float32
// range problem, not an ill-conditioned input -- `Hll + lambda*I` is diagonally
// dominant and trivially invertible.
//
// Factoring out the largest entry keeps every intermediate near unity:
// det(A/s) = det(A)/s^3 and inv(A/s) = s*inv(A), so inv(A) = inv(A/s)/s.
template <typename S>
__host__ __device__ inline void sym3x3InvComponents(
	S A00_raw, S A01_raw, S A02_raw, S A11_raw, S A12_raw, S A22_raw,
	S* B00, S* B01, S* B02, S* B11, S* B12, S* B22)
{
	S scale = fabs(A00_raw);
	scale = fmax(scale, fabs(A01_raw));
	scale = fmax(scale, fabs(A02_raw));
	scale = fmax(scale, fabs(A11_raw));
	scale = fmax(scale, fabs(A12_raw));
	scale = fmax(scale, fabs(A22_raw));
	// An all-zero block has no inverse. Keep the historical behaviour (1/det
	// on a zero determinant) rather than inventing a value here.
	if (!(scale > S(0)))
		scale = S(1);
	const S invScale = S(1) / scale;

	const S A00 = A00_raw * invScale;
	const S A01 = A01_raw * invScale;
	const S A02 = A02_raw * invScale;
	const S A11 = A11_raw * invScale;
	const S A12 = A12_raw * invScale;
	const S A22 = A22_raw * invScale;

	const S det
		= A00 * A11 * A22
		+ A01 * A12 * A02
		+ A02 * A01 * A12
		- A00 * A12 * A12
		- A02 * A11 * A02
		- A01 * A01 * A22;

	// inv(A) = inv(A/scale) / scale, so fold the scale back in once.
	const S invDet = invScale / det;

	*B00 = invDet * (A11 * A22 - A12 * A12);
	*B01 = invDet * (A02 * A12 - A01 * A22);
	*B11 = invDet * (A00 * A22 - A02 * A02);
	*B02 = invDet * (A01 * A12 - A02 * A11);
	*B12 = invDet * (A02 * A01 - A00 * A12);
	*B22 = invDet * (A00 * A11 - A01 * A01);
}

} // namespace gpu
} // namespace cuba

#endif // !__SYM3X3_INV_CUH__
