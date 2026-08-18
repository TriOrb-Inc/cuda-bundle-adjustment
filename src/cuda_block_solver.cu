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

#include "cuda_block_solver.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "deterministic_atomics.cuh"

namespace cuba
{
namespace gpu
{

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

template <int N>
using Vecxd = Vec<Scalar, N>;

template <int N>
using GpuVecxd = GpuVec<Vecxd<N>>;

using PxPBlockPtr = BlockPtr<Scalar, PDIM, PDIM>;
using LxLBlockPtr = BlockPtr<Scalar, LDIM, LDIM>;
using PxLBlockPtr = BlockPtr<Scalar, PDIM, LDIM>;
using Px1BlockPtr = BlockPtr<Scalar, PDIM, 1>;
using Lx1BlockPtr = BlockPtr<Scalar, LDIM, 1>;

////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////
constexpr int BLOCK_ACTIVE_ERRORS = 512;
constexpr int BLOCK_MAX_DIAGONAL = 512;
constexpr int BLOCK_COMPUTE_SCALE = 512;

inline void prepareCudaThreadContext()
{
	CUDA_CHECK(cudaSetDevice(0));
}

////////////////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////////////////
struct LessRowId
{
	__device__ bool operator()(const Vec3i& lhs, const Vec3i& rhs) const
	{
		if (lhs[0] == rhs[0])
		{
			if (lhs[1] == rhs[1])
				return lhs[2] < rhs[2];
			return lhs[1] < rhs[1];
		}
		return lhs[0] < rhs[0];
	}
};

struct LessColId
{
	__device__ bool operator()(const Vec3i& lhs, const Vec3i& rhs) const
	{
		if (lhs[1] == rhs[1])
		{
			if (lhs[0] == rhs[0])
				return lhs[2] < rhs[2];
			return lhs[0] < rhs[0];
		}
		return lhs[1] < rhs[1];
	}
};

template <typename T, int ROWS, int COLS>
struct MatView
{
	__device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
	__device__ inline MatView(T* data) : data(data) {}
	T* data;
};

template <typename T, int ROWS, int COLS>
struct ConstMatView
{
	__device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }
	__device__ inline ConstMatView(const T* data) : data(data) {}
	const T* data;
};

template <typename T, int ROWS, int COLS>
struct Matx
{
	using View = MatView<T, ROWS, COLS>;
	using ConstView = ConstMatView<T, ROWS, COLS>;
	__device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
	__device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }
	__device__ inline operator View() { return View(data); }
	__device__ inline operator ConstView() const { return ConstView(data); }
	T data[ROWS * COLS];
};

using MatView2x3d = MatView<Scalar, 2, 3>;
using MatView2x6d = MatView<Scalar, 2, 6>;
using MatView3x1d = MatView<Scalar, 3, 1>;
using MatView3x3d = MatView<Scalar, 3, 3>;
using MatView3x6d = MatView<Scalar, 3, 6>;
using ConstMatView3x1d = ConstMatView<Scalar, 3, 1>;
using ConstMatView3x3d = ConstMatView<Scalar, 3, 3>;
using ConstMatView6x6d = ConstMatView<Scalar, 6, 6>;
using ConstMatView6x1d = ConstMatView<Scalar, 6, 1>;

struct CameraParamView
{
	__device__ inline CameraParamView(const Scalar* data) : data(data) {}
	__device__ inline CameraParamView(const Vec5d& camera) : data(camera.data) {}
	__device__ inline Scalar fx() const { return data[0]; }
	__device__ inline Scalar fy() const { return data[1]; }
	__device__ inline Scalar cx() const { return data[2]; }
	__device__ inline Scalar cy() const { return data[3]; }
	__device__ inline Scalar bf() const { return data[4]; }

	const Scalar* data;
};

////////////////////////////////////////////////////////////////////////////////////
// Host functions
////////////////////////////////////////////////////////////////////////////////////
static int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}

////////////////////////////////////////////////////////////////////////////////////
// Device functions (template matrix and verctor operation)
////////////////////////////////////////////////////////////////////////////////////

// assignment operations
using AssignOP = void(*)(Scalar*, Scalar);
__device__ inline void ASSIGN(Scalar* address, Scalar value) { *address = value; }
__device__ inline void ACCUM(Scalar* address, Scalar value) { *address += value; }
__device__ inline void DEACCUM(Scalar* address, Scalar value) { *address -= value; }
__device__ inline void ACCUM_ATOMIC(Scalar* address, Scalar value) { atomicAdd(address, value); }
__device__ inline void DEACCUM_ATOMIC(Scalar* address, Scalar value) { atomicAdd(address, -value); }

// recursive dot product for inline expansion
template <int N>
__device__ inline Scalar dot_(const Scalar* a, const Scalar* b)
{
	return dot_<N - 1>(a, b) + a[N - 1] * b[N - 1];
}

template <>
__device__ inline Scalar dot_<1>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }

// recursive dot product for inline expansion (strided access pattern)
template <int N, int S1, int S2>
__device__ inline Scalar dot_stride_(const Scalar* a, const Scalar* b)
{
	static_assert(S1 == PDIM || S1 == LDIM, "S1 must be PDIM or LDIM");
	static_assert(S2 == 1 || S2 == PDIM || S2 == LDIM, "S2 must be 1 or PDIM or LDIM");
	return dot_stride_<N - 1, S1, S2>(a, b) + a[S1 * (N - 1)] * b[S2 * (N - 1)];
}

template <>
__device__ inline Scalar dot_stride_<1, PDIM, 1>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }
template <>
__device__ inline Scalar dot_stride_<1, LDIM, 1>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }
template <>
__device__ inline Scalar dot_stride_<1, PDIM, PDIM>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }
template <>
__device__ inline Scalar dot_stride_<1, LDIM, LDIM>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }

// matrix(tansposed)-vector product: b = AT*x
template <int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatTMulVec(const Scalar* A, const Scalar* x, Scalar* b, Scalar omega)
{
#pragma unroll
	for (int i = 0; i < M; i++)
		OP(b + i, omega * dot_<N>(A + i * N, x));
}

// matrix(tansposed)-matrix product: C = AT*B
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatTMulMat(const Scalar* A, const Scalar* B, Scalar* C, Scalar omega)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatTMulVec<L, M, OP>(A, B + i * M, C + i * L, omega);
}

// ---------------------------------------------------------------------------
// Deterministic (fixed-point int64) variants of MatTMulVec / MatTMulMat.
// Used by Option 4 Phase 2+ when the caller supplies a parallel int64 storage
// buffer (see `deterministic_atomics.cuh`). The output address is a pointer
// into a `long long` buffer sized identically to the corresponding `Scalar`
// buffer, and every accumulation is performed via
// `deterministic::atomicAccumDet`, which uses `atomicAdd(unsigned long long*,
// ...)` on a fixed-point quantum so the final bit-pattern is independent of
// SM execution order.
// ---------------------------------------------------------------------------
template <int M, int N>
__device__ inline void MatTMulVecDet(const Scalar* A, const Scalar* x,
	long long* b_int, Scalar omega)
{
#pragma unroll
	for (int i = 0; i < M; i++)
		deterministic::atomicAccumDet(b_int + i, omega * dot_<N>(A + i * N, x));
}

template <int L, int M, int N>
__device__ inline void MatTMulMatDet(const Scalar* A, const Scalar* B,
	long long* C_int, Scalar omega)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatTMulVecDet<L, M>(A, B + i * M, C_int + i * L, omega);
}

// matrix-vector product: b = A*x
template <int M, int N, int S = 1, AssignOP OP = ASSIGN>
__device__ inline void MatMulVec(const Scalar* A, const Scalar* x, Scalar* b)
{
#pragma unroll
	for (int i = 0; i < M; i++)
		OP(b + i, dot_stride_<N, M, S>(A + i, x));
}

// matrix-matrix product: C = A*B
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatMulMat(const Scalar* A, const Scalar* B, Scalar* C)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatMulVec<L, M, 1, OP>(A, B + i * M, C + i * L);
}

// matrix-matrix(tansposed) product: C = A*BT
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatMulMatT(const Scalar* A, const Scalar* B, Scalar* C)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatMulVec<L, M, N, OP>(A, B + i, C + i * L);
}

// ---------------------------------------------------------------------------
// Deterministic (fixed-point int64) variants of MatMulVec / MatMulMatT.
// Used by Option 4 Phase 3f (Schur-complement update in computeBschureKernel /
// computeHschureKernel). The legacy call sites use DEACCUM_ATOMIC, so these
// deterministic variants take a signed `sign` multiplier (-1 replays the
// legacy DEACCUM behavior); the dot product is scaled and accumulated via
// `deterministic::atomicAccumDet` on an int64 buffer whose layout mirrors
// the original Scalar output buffer (bsc: numP*PDIM; Hsc: nnz*PDIM*PDIM).
// After kernel completion the caller invokes `convertFixedPoint*` helpers
// to propagate the int64 increments additively back into the double buffer.
// ---------------------------------------------------------------------------
template <int M, int N, int S = 1>
__device__ inline void MatMulVecDet(const Scalar* A, const Scalar* x,
	long long* b_int, Scalar sign)
{
#pragma unroll
	for (int i = 0; i < M; i++)
		deterministic::atomicAccumDet(b_int + i, sign * dot_stride_<N, M, S>(A + i, x));
}

template <int L, int M, int N>
__device__ inline void MatMulMatTDet(const Scalar* A, const Scalar* B,
	long long* C_int, Scalar sign)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatMulVecDet<L, M, N>(A, B + i, C_int + i * L, sign);
}

// squared L2 norm
template <int N>
__device__ inline Scalar squaredNorm(const Scalar* x) { return dot_<N>(x, x); }
template <int N>
__device__ inline Scalar squaredNorm(const Vecxd<N>& x) { return squaredNorm<N>(x.data); }

// L2 norm
template <int N>
__device__ inline Scalar norm(const Scalar* x) { return sqrt(squaredNorm<N>(x)); }
template <int N>
__device__ inline Scalar norm(const Vecxd<N>& x) { return norm<N>(x.data); }

////////////////////////////////////////////////////////////////////////////////////
// Device functions
////////////////////////////////////////////////////////////////////////////////////
__device__ static inline void cross(const Vec4d& a, const Vec3d& b, Vec3d& c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void rotate(const Vec4d& q, const Vec3d& Xw, Vec3d& Xc)
{
	Vec3d tmp1, tmp2;

	cross(q, Xw, tmp1);

	tmp1[0] += tmp1[0];
	tmp1[1] += tmp1[1];
	tmp1[2] += tmp1[2];

	cross(q, tmp1, tmp2);

	Xc[0] = Xw[0] + q[3] * tmp1[0] + tmp2[0];
	Xc[1] = Xw[1] + q[3] * tmp1[1] + tmp2[1];
	Xc[2] = Xw[2] + q[3] * tmp1[2] + tmp2[2];
}

__device__ inline void projectW2C(const Vec4d& q, const Vec3d& t, const Vec3d& Xw, Vec3d& Xc)
{
	rotate(q, Xw, Xc);
	Xc[0] += t[0];
	Xc[1] += t[1];
	Xc[2] += t[2];
}

__device__ inline void applyExtrinsics(const Vec4d& q_ext, const Vec3d& t_ext, const Vec3d& Xc_body, Vec3d& Xc)
{
	rotate(q_ext, Xc_body, Xc);
	Xc[0] += t_ext[0];
	Xc[1] += t_ext[1];
	Xc[2] += t_ext[2];
}

template <int MDIM>
__device__ inline void projectC2I(const Vec3d& Xc, Vecxd<MDIM>& p, CameraParamView camera)
{
}

// Kannala-Brandt equidistant projection model.
// theta = atan2(sqrt(X^2+Y^2), Z)
// theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
// u = fx * (theta_d/r) * X + cx    where r = sqrt(X^2+Y^2)
// v = fy * (theta_d/r) * Y + cy
__device__ inline void projectC2I_equidistant(const Vec3d& Xc, Vec2d& p,
	CameraParamView camera, const Scalar* distortion)
{
	const Scalar X = Xc[0];
	const Scalar Y = Xc[1];
	const Scalar Z = Xc[2];
	const Scalar r = sqrt(X * X + Y * Y);
	const Scalar eps = Scalar(1e-10);

	if (r < eps) {
		// Degenerate case: point on optical axis, use pinhole approximation
		const Scalar invZ = 1 / Z;
		p[0] = camera.fx() * invZ * X + camera.cx();
		p[1] = camera.fy() * invZ * Y + camera.cy();
		return;
	}

	const Scalar theta = atan2(r, Z);
	const Scalar k1 = distortion[0];
	const Scalar k2 = distortion[1];
	const Scalar k3 = distortion[2];
	const Scalar k4 = distortion[3];
	const Scalar theta2 = theta * theta;
	const Scalar theta3 = theta2 * theta;
	const Scalar theta_d = theta + k1 * theta3 + k2 * theta2 * theta3
		+ k3 * theta3 * theta3 * theta + k4 * theta3 * theta3 * theta3;
	const Scalar scale = theta_d / r;
	p[0] = camera.fx() * scale * X + camera.cx();
	p[1] = camera.fy() * scale * Y + camera.cy();
}

template <>
__device__ inline void projectC2I<2>(const Vec3d& Xc, Vec2d& p, CameraParamView camera)
{
	const Scalar invZ = 1 / Xc[2];
	p[0] = camera.fx() * invZ * Xc[0] + camera.cx();
	p[1] = camera.fy() * invZ * Xc[1] + camera.cy();
}

template <>
__device__ inline void projectC2I<3>(const Vec3d& Xc, Vec3d& p, CameraParamView camera)
{
	const Scalar invZ = 1 / Xc[2];
	p[0] = camera.fx() * invZ * Xc[0] + camera.cx();
	p[1] = camera.fy() * invZ * Xc[1] + camera.cy();
	p[2] = p[0] - camera.bf() * invZ;
}

__device__ inline void quaternionToRotationMatrix(const Vec4d& q, MatView3x3d R)
{
	const Scalar x = q[0];
	const Scalar y = q[1];
	const Scalar z = q[2];
	const Scalar w = q[3];

	const Scalar tx = 2 * x;
	const Scalar ty = 2 * y;
	const Scalar tz = 2 * z;
	const Scalar twx = tx * w;
	const Scalar twy = ty * w;
	const Scalar twz = tz * w;
	const Scalar txx = tx * x;
	const Scalar txy = ty * x;
	const Scalar txz = tz * x;
	const Scalar tyy = ty * y;
	const Scalar tyz = tz * y;
	const Scalar tzz = tz * z;

	R(0, 0) = 1 - (tyy + tzz);
	R(0, 1) = txy - twz;
	R(0, 2) = txz + twy;
	R(1, 0) = txy + twz;
	R(1, 1) = 1 - (txx + tzz);
	R(1, 2) = tyz - twx;
	R(2, 0) = txz - twy;
	R(2, 1) = tyz + twx;
	R(2, 2) = 1 - (txx + tyy);
}

// Exact Jacobians accounting for per-edge extrinsics R_ext in the chain rule.
// Xc: camera-optical-frame point (post-extrinsics).
// Xc_body: body-frame point (pre-extrinsics).
// q: body pose quaternion.
// q_ext/t_ext: camera_optical_from_body extrinsics.
template <int MDIM>
__device__ void computeJacobiansExact(const Vec3d& Xc, const Vec3d& Xc_body, const Vec4d& q,
	const Vec4d& q_ext, const Vec3d& t_ext,
	MatView<Scalar, MDIM, PDIM> JP, MatView<Scalar, MDIM, LDIM> JL, CameraParamView camera)
{
}

template <>
__device__ void computeJacobiansExact<2>(const Vec3d& Xc, const Vec3d& Xc_body, const Vec4d& q,
	const Vec4d& q_ext, const Vec3d& t_ext,
	MatView2x6d JP, MatView2x3d JL, CameraParamView camera)
{
	const Scalar X = Xc[0];
	const Scalar Y = Xc[1];
	const Scalar Z = Xc[2];
	const Scalar invZ = 1 / Z;
	const Scalar fu = camera.fx();
	const Scalar fv = camera.fy();

	// J_pi: 2×3 projection Jacobian d(proj)/d(Xc)
	// row 0: [-fu/Z,    0, fu*X/Z^2]
	// row 1: [   0, -fv/Z, fv*Y/Z^2]
	const Scalar invZZ = invZ * invZ;
	Scalar Jpi[2][3];
	Jpi[0][0] = -fu * invZ;  Jpi[0][1] = 0;            Jpi[0][2] = fu * X * invZZ;
	Jpi[1][0] = 0;           Jpi[1][1] = -fv * invZ;    Jpi[1][2] = fv * Y * invZZ;

	// R_ext: extrinsics rotation matrix
	Matx<Scalar, 3, 3> Re;
	quaternionToRotationMatrix(q_ext, Re);

	// J_pi_ext = J_pi * R_ext  (2×3)
	Scalar Jpe[2][3];
	for (int r = 0; r < 2; r++)
		for (int c = 0; c < 3; c++)
			Jpe[r][c] = Jpi[r][0] * Re(0, c) + Jpi[r][1] * Re(1, c) + Jpi[r][2] * Re(2, c);

	// R_body
	Matx<Scalar, 3, 3> Rb;
	quaternionToRotationMatrix(q, Rb);

	// JL = J_pi_ext * R_body  (2×3)
	for (int r = 0; r < 2; r++)
		for (int c = 0; c < 3; c++)
			JL(r, c) = Jpe[r][0] * Rb(0, c) + Jpe[r][1] * Rb(1, c) + Jpe[r][2] * Rb(2, c);

	// JP rotation columns [0:3]: J_pi_ext * [-Xc_body×]
	// [-Xc_body×] = [[0, Zb, -Yb], [-Zb, 0, Xb], [Yb, -Xb, 0]]
	const Scalar Xb = Xc_body[0];
	const Scalar Yb = Xc_body[1];
	const Scalar Zb = Xc_body[2];
	JP(0, 0) = Jpe[0][1] * (-Zb) + Jpe[0][2] * Yb;
	JP(0, 1) = Jpe[0][0] * Zb + Jpe[0][2] * (-Xb);
	JP(0, 2) = Jpe[0][0] * (-Yb) + Jpe[0][1] * Xb;
	JP(1, 0) = Jpe[1][1] * (-Zb) + Jpe[1][2] * Yb;
	JP(1, 1) = Jpe[1][0] * Zb + Jpe[1][2] * (-Xb);
	JP(1, 2) = Jpe[1][0] * (-Yb) + Jpe[1][1] * Xb;

	// JP translation columns [3:6]: J_pi_ext
	JP(0, 3) = Jpe[0][0];
	JP(0, 4) = Jpe[0][1];
	JP(0, 5) = Jpe[0][2];
	JP(1, 3) = Jpe[1][0];
	JP(1, 4) = Jpe[1][1];
	JP(1, 5) = Jpe[1][2];
}

// Exact Jacobians for Kannala-Brandt equidistant projection model.
// Uses chain rule: d(u,v)/d(Xc) for equidistant, then same extrinsics/pose chain as pinhole.
__device__ void computeJacobiansExact_equidistant(const Vec3d& Xc, const Vec3d& Xc_body, const Vec4d& q,
	const Vec4d& q_ext, const Vec3d& t_ext,
	MatView2x6d JP, MatView2x3d JL, CameraParamView camera, const Scalar* distortion)
{
	const Scalar X = Xc[0];
	const Scalar Y = Xc[1];
	const Scalar Z = Xc[2];
	const Scalar fu = camera.fx();
	const Scalar fv = camera.fy();
	const Scalar r = sqrt(X * X + Y * Y);
	const Scalar eps = Scalar(1e-10);

	// Jpi_equi: 2x3 projection Jacobian for equidistant model.
	// NOTE: Following the existing codebase convention, Jpi stores the NEGATED
	// Jacobian: Jpi = -d(proj)/d(Xc). This is consistent with the pinhole path.
	Scalar Jpi[2][3];

	if (r < eps) {
		// Degenerate case: fall back to pinhole Jacobian (negated convention)
		const Scalar invZ = 1 / Z;
		const Scalar invZZ = invZ * invZ;
		Jpi[0][0] = -fu * invZ;  Jpi[0][1] = 0;            Jpi[0][2] = fu * X * invZZ;
		Jpi[1][0] = 0;           Jpi[1][1] = -fv * invZ;    Jpi[1][2] = fv * Y * invZZ;
	}
	else {
		const Scalar r2 = r * r;
		const Scalar r2_plus_Z2 = r2 + Z * Z;

		const Scalar theta = atan2(r, Z);
		const Scalar k1 = distortion[0];
		const Scalar k2 = distortion[1];
		const Scalar k3 = distortion[2];
		const Scalar k4 = distortion[3];
		const Scalar theta2 = theta * theta;
		const Scalar theta4 = theta2 * theta2;
		const Scalar theta6 = theta4 * theta2;
		const Scalar theta8 = theta4 * theta4;
		const Scalar theta3 = theta2 * theta;
		const Scalar theta5 = theta4 * theta;
		const Scalar theta7 = theta6 * theta;
		const Scalar theta9 = theta8 * theta;
		const Scalar theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

		// d(theta_d)/d(theta)
		const Scalar dtheta_d_dtheta = 1 + 3 * k1 * theta2 + 5 * k2 * theta4
			+ 7 * k3 * theta6 + 9 * k4 * theta8;

		// Partial derivatives of theta w.r.t. X, Y, Z
		// theta = atan2(r, Z), r = sqrt(X^2 + Y^2)
		// d(theta)/d(X) = X*Z / (r * (r^2 + Z^2))
		// d(theta)/d(Y) = Y*Z / (r * (r^2 + Z^2))
		// d(theta)/d(Z) = -r / (r^2 + Z^2)
		const Scalar dtheta_dX = X * Z / (r * r2_plus_Z2);
		const Scalar dtheta_dY = Y * Z / (r * r2_plus_Z2);
		const Scalar dtheta_dZ = -r / r2_plus_Z2;

		// Partial derivatives of theta_d w.r.t. X, Y, Z
		const Scalar dthetad_dX = dtheta_d_dtheta * dtheta_dX;
		const Scalar dthetad_dY = dtheta_d_dtheta * dtheta_dY;
		const Scalar dthetad_dZ = dtheta_d_dtheta * dtheta_dZ;

		// Partial derivatives of r w.r.t. X, Y, Z
		const Scalar dr_dX = X / r;
		const Scalar dr_dY = Y / r;
		// dr_dZ = 0

		// s = theta_d / r
		// ds/dX = (dthetad_dX * r - theta_d * dr_dX) / r^2
		// ds/dY = (dthetad_dY * r - theta_d * dr_dY) / r^2
		// ds/dZ = dthetad_dZ / r   (since dr_dZ = 0)
		const Scalar inv_r2 = 1 / r2;
		const Scalar ds_dX = (dthetad_dX * r - theta_d * dr_dX) * inv_r2;
		const Scalar ds_dY = (dthetad_dY * r - theta_d * dr_dY) * inv_r2;
		const Scalar ds_dZ = dthetad_dZ / r;

		const Scalar s = theta_d / r;

		// True derivatives (then negate to match convention):
		// du/dX = fu * (s + X * ds_dX)
		// du/dY = fu * X * ds_dY
		// du/dZ = fu * X * ds_dZ
		// Negated for consistency with pinhole path:
		Jpi[0][0] = -(fu * (s + X * ds_dX));
		Jpi[0][1] = -(fu * X * ds_dY);
		Jpi[0][2] = -(fu * X * ds_dZ);

		// dv/dX = fv * Y * ds_dX
		// dv/dY = fv * (s + Y * ds_dY)
		// dv/dZ = fv * Y * ds_dZ
		Jpi[1][0] = -(fv * Y * ds_dX);
		Jpi[1][1] = -(fv * (s + Y * ds_dY));
		Jpi[1][2] = -(fv * Y * ds_dZ);
	}

	// The rest is identical to the pinhole path: chain through extrinsics and pose.

	// R_ext: extrinsics rotation matrix
	Matx<Scalar, 3, 3> Re;
	quaternionToRotationMatrix(q_ext, Re);

	// Jpe = Jpi * R_ext  (2x3)
	Scalar Jpe[2][3];
	for (int row = 0; row < 2; row++)
		for (int col = 0; col < 3; col++)
			Jpe[row][col] = Jpi[row][0] * Re(0, col) + Jpi[row][1] * Re(1, col) + Jpi[row][2] * Re(2, col);

	// R_body
	Matx<Scalar, 3, 3> Rb;
	quaternionToRotationMatrix(q, Rb);

	// JL = Jpe * R_body  (2x3)
	for (int row = 0; row < 2; row++)
		for (int col = 0; col < 3; col++)
			JL(row, col) = Jpe[row][0] * Rb(0, col) + Jpe[row][1] * Rb(1, col) + Jpe[row][2] * Rb(2, col);

	// JP rotation columns [0:3]: Jpe * [-Xc_body x]
	const Scalar Xb = Xc_body[0];
	const Scalar Yb = Xc_body[1];
	const Scalar Zb = Xc_body[2];
	JP(0, 0) = Jpe[0][1] * (-Zb) + Jpe[0][2] * Yb;
	JP(0, 1) = Jpe[0][0] * Zb + Jpe[0][2] * (-Xb);
	JP(0, 2) = Jpe[0][0] * (-Yb) + Jpe[0][1] * Xb;
	JP(1, 0) = Jpe[1][1] * (-Zb) + Jpe[1][2] * Yb;
	JP(1, 1) = Jpe[1][0] * Zb + Jpe[1][2] * (-Xb);
	JP(1, 2) = Jpe[1][0] * (-Yb) + Jpe[1][1] * Xb;

	// JP translation columns [3:6]: Jpe
	JP(0, 3) = Jpe[0][0];
	JP(0, 4) = Jpe[0][1];
	JP(0, 5) = Jpe[0][2];
	JP(1, 3) = Jpe[1][0];
	JP(1, 4) = Jpe[1][1];
	JP(1, 5) = Jpe[1][2];
}

template <>
__device__ void computeJacobiansExact<3>(const Vec3d& Xc, const Vec3d& Xc_body, const Vec4d& q,
	const Vec4d& q_ext, const Vec3d& t_ext,
	MatView3x6d JP, MatView3x3d JL, CameraParamView camera)
{
	const Scalar X = Xc[0];
	const Scalar Y = Xc[1];
	const Scalar Z = Xc[2];
	const Scalar invZ = 1 / Z;
	const Scalar invZZ = invZ * invZ;
	const Scalar fu = camera.fx();
	const Scalar fv = camera.fy();
	const Scalar bf = camera.bf();

	// J_pi: 3×3 projection Jacobian for stereo
	// row 0 (left u):  [-fu/Z,    0, fu*X/Z^2]
	// row 1 (left v):  [   0, -fv/Z, fv*Y/Z^2]
	// row 2 (right u): [-fu/Z,    0, (fu*X+bf)/Z^2]
	Scalar Jpi[3][3];
	Jpi[0][0] = -fu * invZ;  Jpi[0][1] = 0;            Jpi[0][2] = fu * X * invZZ;
	Jpi[1][0] = 0;           Jpi[1][1] = -fv * invZ;    Jpi[1][2] = fv * Y * invZZ;
	Jpi[2][0] = -fu * invZ;  Jpi[2][1] = 0;            Jpi[2][2] = (fu * X + bf) * invZZ;

	// R_ext
	Matx<Scalar, 3, 3> Re;
	quaternionToRotationMatrix(q_ext, Re);

	// J_pi_ext = J_pi * R_ext  (3×3)
	Scalar Jpe[3][3];
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			Jpe[r][c] = Jpi[r][0] * Re(0, c) + Jpi[r][1] * Re(1, c) + Jpi[r][2] * Re(2, c);

	// R_body
	Matx<Scalar, 3, 3> Rb;
	quaternionToRotationMatrix(q, Rb);

	// JL = J_pi_ext * R_body  (3×3)
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			JL(r, c) = Jpe[r][0] * Rb(0, c) + Jpe[r][1] * Rb(1, c) + Jpe[r][2] * Rb(2, c);

	// JP rotation columns [0:3]: J_pi_ext * [-Xc_body×]
	const Scalar Xb = Xc_body[0];
	const Scalar Yb = Xc_body[1];
	const Scalar Zb = Xc_body[2];
	for (int r = 0; r < 3; r++)
	{
		JP(r, 0) = Jpe[r][1] * (-Zb) + Jpe[r][2] * Yb;
		JP(r, 1) = Jpe[r][0] * Zb + Jpe[r][2] * (-Xb);
		JP(r, 2) = Jpe[r][0] * (-Yb) + Jpe[r][1] * Xb;
	}

	// JP translation columns [3:6]: J_pi_ext
	for (int r = 0; r < 3; r++)
	{
		JP(r, 3) = Jpe[r][0];
		JP(r, 4) = Jpe[r][1];
		JP(r, 5) = Jpe[r][2];
	}
}

__device__ inline void Sym3x3Inv(ConstMatView3x3d A, MatView3x3d B)
{
	const Scalar A00 = A(0, 0);
	const Scalar A01 = A(0, 1);
	const Scalar A11 = A(1, 1);
	const Scalar A02 = A(2, 0);
	const Scalar A12 = A(1, 2);
	const Scalar A22 = A(2, 2);

	const Scalar det
		= A00 * A11 * A22
		+ A01 * A12 * A02
		+ A02 * A01 * A12
		- A00 * A12 * A12
		- A02 * A11 * A02
		- A01 * A01 * A22;

	const Scalar invDet = 1 / det;

	const Scalar B00 = invDet * (A11 * A22 - A12 * A12);
	const Scalar B01 = invDet * (A02 * A12 - A01 * A22);
	const Scalar B11 = invDet * (A00 * A22 - A02 * A02);
	const Scalar B02 = invDet * (A01 * A12 - A02 * A11);
	const Scalar B12 = invDet * (A02 * A01 - A00 * A12);
	const Scalar B22 = invDet * (A00 * A11 - A01 * A01);

	B(0, 0) = B00;
	B(0, 1) = B01;
	B(0, 2) = B02;
	B(1, 0) = B01;
	B(1, 1) = B11;
	B(1, 2) = B12;
	B(2, 0) = B02;
	B(2, 1) = B12;
	B(2, 2) = B22;
}

__device__ inline void skew1(Scalar x, Scalar y, Scalar z, MatView3x3d M)
{
	M(0, 0) = +0; M(0, 1) = -z; M(0, 2) = +y;
	M(1, 0) = +z; M(1, 1) = +0; M(1, 2) = -x;
	M(2, 0) = -y; M(2, 1) = +x; M(2, 2) = +0;
}

__device__ inline void skew2(Scalar x, Scalar y, Scalar z, MatView3x3d M)
{
	const Scalar xx = x * x;
	const Scalar yy = y * y;
	const Scalar zz = z * z;

	const Scalar xy = x * y;
	const Scalar yz = y * z;
	const Scalar zx = z * x;

	M(0, 0) = -yy - zz; M(0, 1) = +xy;      M(0, 2) = +zx;
	M(1, 0) = +xy;      M(1, 1) = -zz - xx; M(1, 2) = +yz;
	M(2, 0) = +zx;      M(2, 1) = +yz;      M(2, 2) = -xx - yy;
}

__device__ inline void addOmega(Scalar a1, ConstMatView3x3d O1, Scalar a2, ConstMatView3x3d O2,
	MatView3x3d R)
{
	R(0, 0) = 1 + a1 * O1(0, 0) + a2 * O2(0, 0);
	R(1, 0) = 0 + a1 * O1(1, 0) + a2 * O2(1, 0);
	R(2, 0) = 0 + a1 * O1(2, 0) + a2 * O2(2, 0);

	R(0, 1) = 0 + a1 * O1(0, 1) + a2 * O2(0, 1);
	R(1, 1) = 1 + a1 * O1(1, 1) + a2 * O2(1, 1);
	R(2, 1) = 0 + a1 * O1(2, 1) + a2 * O2(2, 1);

	R(0, 2) = 0 + a1 * O1(0, 2) + a2 * O2(0, 2);
	R(1, 2) = 0 + a1 * O1(1, 2) + a2 * O2(1, 2);
	R(2, 2) = 1 + a1 * O1(2, 2) + a2 * O2(2, 2);
}

__device__ inline void rotationMatrixToQuaternion(ConstMatView3x3d R, Vec4d& q)
{
	Scalar t = R(0, 0) + R(1, 1) + R(2, 2);
	if (t > 0)
	{
		t = sqrt(t + 1);
		q[3] = Scalar(0.5) * t;
		t = Scalar(0.5) / t;
		q[0] = (R(2, 1) - R(1, 2)) * t;
		q[1] = (R(0, 2) - R(2, 0)) * t;
		q[2] = (R(1, 0) - R(0, 1)) * t;
	}
	else
	{
		int i = 0;
		if (R(1, 1) > R(0, 0))
			i = 1;
		if (R(2, 2) > R(i, i))
			i = 2;
		int j = (i + 1) % 3;
		int k = (j + 1) % 3;

		t = sqrt(R(i, i) - R(j, j) - R(k, k) + 1);
		q[i] = Scalar(0.5) * t;
		t = Scalar(0.5) / t;
		q[3] = (R(k, j) - R(j, k)) * t;
		q[j] = (R(j, i) + R(i, j)) * t;
		q[k] = (R(k, i) + R(i, k)) * t;
	}
}

__device__ inline void multiplyQuaternion(const Vec4d& a, const Vec4d& b, Vec4d& c)
{
	c[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
	c[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0];
}

__device__ inline void normalizeQuaternion(const Vec4d& a, Vec4d& b)
{
	Scalar invn = 1 / norm(a);
	if (a[3] < 0)
		invn = -invn;

	for (int i = 0; i < 4; i++)
		b[i] = invn * a[i];
}

__device__ inline Scalar pow2(Scalar x)
{
	return x * x;
}

__device__ inline Scalar pow3(Scalar x)
{
	return x * x * x;
}

__device__ inline void updateExp(const Scalar* update, Vec4d& q, Vec3d& t)
{
	Vec3d omega(update);
	Vec3d upsilon(update + 3);

	const Scalar theta = norm(omega);

	Matx<Scalar, 3, 3> O1, O2;
	skew1(omega[0], omega[1], omega[2], O1);
	skew2(omega[0], omega[1], omega[2], O2);

	Scalar R[9], V[9];
	if (theta < Scalar(0.00001))
	{
		addOmega(Scalar(1.0), O1, Scalar(0.5), O2, R);
		addOmega(Scalar(0.5), O1, Scalar(1)/6, O2, V);
	}
	else
	{
		const Scalar a1 = sin(theta) / theta;
		const Scalar a2 = (1 - cos(theta)) / (theta * theta);
		const Scalar a3 = (theta - sin(theta)) / pow3(theta);
		addOmega(a1, O1, a2, O2, R);
		addOmega(a2, O1, a3, O2, V);
	}

	rotationMatrixToQuaternion(R, q);
	MatMulVec<3, 3>(V, upsilon.data, t.data);
}

__device__ inline void updatePose(const Vec4d& q1, const Vec3d& t1, Vec4d& q2, Vec3d& t2)
{
	Vec3d u;
	rotate(q1, t2, u);

	for (int i = 0; i < 3; i++)
		t2[i] = t1[i] + u[i];

	Vec4d r;
	multiplyQuaternion(q1, q2, r);
	normalizeQuaternion(r, q2);
}

template <int N>
__device__ inline void copy(const Scalar* src, Scalar* dst)
{
	for (int i = 0; i < N; i++)
		dst[i] = src[i];
}

__device__ inline Vec3i makeVec3i(int i, int j, int k)
{
	Vec3i  vec;
	vec[0] = i;
	vec[1] = j;
	vec[2] = k;
	return vec;
}

__device__ inline  void solveSym3x3(const Scalar* H, const Scalar* b, Scalar* x)
{
	Scalar invH[LDIM * LDIM];
	Sym3x3Inv(H, invH);
	MatMulVec<3, 3>(invH, b, x);
}

__device__ inline  void solveSym6x6(const Scalar* _H, const Scalar* _b, Scalar* _x)
{
	using Mat3x3d = Matx<Scalar, 3, 3>;
	using Mat3x1d = Matx<Scalar, 3, 1>;
	using Mat6x1d = Matx<Scalar, 6, 1>;

	ConstMatView6x6d H(_H);
	ConstMatView6x1d b(_b);
	ConstMatView3x1d bp(b.data);
	ConstMatView3x1d bl(b.data + 3);

	Scalar buf1[LDIM * LDIM], buf2[LDIM * LDIM], buf3[LDIM * LDIM], buf4[LDIM];

	MatView3x3d Hpl(buf1);
	MatView3x3d Hll(buf2);
	for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) Hpl(i, j) = H(i + 0, j + 3);
	for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) Hll(i, j) = H(i + 3, j + 3);

	Mat6x1d x;
	MatView3x1d xp(x.data);
	MatView3x1d xl(x.data + 3);
	MatView3x3d invHll(buf3), Hpl_invHll(buf2);

	// Hsc = Hpp - Hpl*Hll^-1*HplT
	Mat3x3d Hsc;
	for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) Hsc(i, j) = H(i, j);
	Sym3x3Inv(Hll.data, invHll.data);
	MatMulMat<3, 3, 3>(Hpl.data, invHll.data, Hpl_invHll.data);
	MatMulMatT<3, 3, 3, DEACCUM>(Hpl_invHll.data, Hpl.data, Hsc.data);

	// bsc = -bp + Hpl*Hll^-1*bl
	MatView3x1d bsc(buf4);
	copy<3>(bp.data, bsc.data);
	MatMulVec<3, 3, 1, DEACCUM>(Hpl_invHll.data, bl.data, bsc.data);

	// Hsc*Δxp = bsc
	MatView3x3d invHsc(buf2);
	Sym3x3Inv(Hsc, invHsc);
	MatMulVec<3, 3>(invHsc.data, bsc.data, xp.data);

	// Hll*Δxl = -bl - HplT*Δxp
	MatView3x1d cl(buf4);
	copy<3>(bl.data, cl.data);
	MatTMulVec<3, 3, DEACCUM>(Hpl.data, xp.data, cl.data, 1);
	MatMulVec<3, 3>(invHll.data, cl.data, xl.data);

	copy<6>(x.data, _x);
}

////////////////////////////////////////////////////////////////////////////////////
// Robust kernels
////////////////////////////////////////////////////////////////////////////////////
enum RobustKernelType
{
	NONE  = 0,
	HUBER = 1,
	TUKEY = 2,
};

template <int TYPE>
struct RobustKernelFunc
{
	__device__ inline RobustKernelFunc(Scalar delta) {}
	__device__ inline Scalar robustify(Scalar x) const { return x; }
	__device__ inline Scalar derivative(Scalar x) const { return 1; }
};

template <>
struct RobustKernelFunc<RobustKernelType::NONE>
{
	__device__ inline RobustKernelFunc(Scalar delta) {}
	__device__ inline Scalar robustify(Scalar x) const { return x; }
	__device__ inline Scalar derivative(Scalar x) const { return 1; }
};

template <>
struct RobustKernelFunc<RobustKernelType::HUBER>
{
	__device__ inline RobustKernelFunc(Scalar delta) : delta(delta), deltaSq(delta * delta) {}

	__device__ inline Scalar robustify(Scalar x) const
	{
		return x <= deltaSq ? x : (2 * sqrt(x) * delta - deltaSq);
	}

	__device__ inline Scalar derivative(Scalar x) const
	{
		return x <= deltaSq ? 1 : (delta / sqrt(x));
	}

	Scalar delta, deltaSq;
};

template <>
struct RobustKernelFunc<RobustKernelType::TUKEY>
{
	__device__ inline RobustKernelFunc(Scalar delta) : delta(delta), deltaSq(delta * delta) {}

	__device__ inline Scalar robustify(Scalar x) const
	{
		const Scalar maxv = (Scalar(1) / 3) * deltaSq;
		return x <= deltaSq ? maxv * (1 - pow3(1 - x / deltaSq)) : maxv;
	}

	__device__ inline Scalar derivative(Scalar x) const
	{
		return x <= deltaSq ? pow2(1 - x / deltaSq) : 0;
	}

	Scalar delta, deltaSq;
};

////////////////////////////////////////////////////////////////////////////////////
// Kernel functions
////////////////////////////////////////////////////////////////////////////////////
template <int MDIM, int RK_TYPE>
__global__ void computeActiveErrorsKernel(int nedges, const Vec4d* qs, const Vec3d* ts, const Vec5d* cameras,
	const Vec3d* Xws, const Vecxd<MDIM>* measurements, const Scalar* omegas, const Vec2i* edge2PL,
	const Vec4d* q_exts, const Vec3d* t_exts, const Vec4d* distortions,
	RobustKernelFunc<RK_TYPE> robustKernel, Vecxd<MDIM>* errors, Vec3d* Xcs, Scalar* chi,
	// Option 4 Phase 3g: when non-null, the final reduction writes into a
	// fixed-point int64 accumulator instead of the double `chi` buffer. This
	// eliminates the last `atomicAdd(double*)` site on the LM error-evaluation
	// path so `rho = (F - Fhat) / scale` becomes fully bit-identical.
	long long* chi_int)
{
	using Vecmd = Vecxd<MDIM>;

	const int sharedIdx = threadIdx.x;
	__shared__ Scalar cache[BLOCK_ACTIVE_ERRORS];

	Scalar sumchi = 0;
	for (int iE = blockIdx.x * blockDim.x + threadIdx.x; iE < nedges; iE += gridDim.x * blockDim.x)
	{
		const Vec2i index = edge2PL[iE];
		const int iP = index[0];
		const int iL = index[1];

		const Vec4d& q = qs[iP];
		const Vec3d& t = ts[iP];
		const Vec5d& camera = cameras[iP];
		const Vec3d& Xw = Xws[iL];
		const Vecmd& measurement = measurements[iE];

		// project world to body frame
		Vec3d Xc_body;
		projectW2C(q, t, Xw, Xc_body);

		// apply per-edge extrinsics: Xc = R_ext * Xc_body + t_ext
		Vec3d Xc;
		applyExtrinsics(q_exts[iE], t_exts[iE], Xc_body, Xc);

		// project camera to image: equidistant if distortion is non-zero (2D only)
		Vecmd proj;
		if (MDIM == 2 && distortions != nullptr)
		{
			const Scalar* dist = distortions[iE].data;
			const bool use_equidistant = (dist[0] != 0 || dist[1] != 0 || dist[2] != 0 || dist[3] != 0);
			if (use_equidistant) {
				// Only valid for MDIM==2 (monocular); cast through pointer to satisfy compiler
				projectC2I_equidistant(Xc, *reinterpret_cast<Vec2d*>(&proj), camera, dist);
			} else {
				projectC2I(Xc, proj, camera);
			}
		}
		else
		{
			projectC2I(Xc, proj, camera);
		}

		// compute residual
		Vecmd error;
		for (int i = 0; i < MDIM; i++)
			error[i] = proj[i] - measurement[i];

		errors[iE] = error;
		Xcs[iE] = Xc;

		sumchi += robustKernel.robustify(omegas[iE] * squaredNorm(error));
	}

	cache[sharedIdx] = sumchi;
	__syncthreads();

	for (int stride = BLOCK_ACTIVE_ERRORS / 2; stride > 0; stride >>= 1)
	{
		if (sharedIdx < stride)
			cache[sharedIdx] += cache[sharedIdx + stride];
		__syncthreads();
	}

	if (sharedIdx == 0)
	{
		if (chi_int != nullptr)
			deterministic::atomicAccumDet(chi_int, cache[0]);
		else
			atomicAdd(chi, cache[0]);
	}
}

template <int MDIM, int RK_TYPE>
__global__ void constructQuadraticFormKernel(int nedges, const Vec3d* Xcs, const Vec4d* qs, const Vec5d* cameras, const Vecxd<MDIM>* errors,
	const Scalar* omegas, const Vec2i* edge2PL, const int* edge2Hpl, const int* edge2HplExt, const int* edge2ExtIP, const int* edge2HscPE,
	const uint8_t* flags, RobustKernelFunc<RK_TYPE> robustKernel,
	PxPBlockPtr Hpp, Px1BlockPtr bp, LxLBlockPtr Hll, Lx1BlockPtr bl, PxLBlockPtr Hpl, PxPBlockPtr HscDirect,
	const Vec4d* q_exts, const Vec3d* t_exts, const Vec4d* distortions,
	// Option 4 Phase 2+: when non-null, the kernel routes the body/ext/landmark
	// accumulations for `Hpp`, `bp`, `Hll`, `bl`, `HscDirect`, and the
	// ext-slot portion of `Hpl` into these fixed-point int64 mirror buffers
	// instead of the double buffers, so the final values are bit-identical
	// across runs for the covered slots.
	//   Phase 2 : `Hpp.at(iPExt)`         — ext-range slot in [numBody, numBody+numExt).
	//   Phase 3a: `bp.at(iPExt)`          — same iPExt slot (PDIM scalars).
	//   Phase 3b: `Hpp.at(iP)` / `bp.at(iP)` — body-range slots in [0, numBody).
	//   Phase 3c: `Hll.at(iL)` / `bl.at(iL)` — landmark-range slots in [0, numL).
	//   Phase 3d: `HscDirect.at(hscPESlot)` — direct body×ext Hsc cross blocks
	//             (PDIM×PDIM). Layout matches `HscDirect.values()`; slot
	//             indexing is dense across `d_HscDirect_.nnz()` blocks.
	//   Phase 3e: `Hpl.at(hplExtSlot)` — ext×landmark Hpl cross blocks
	//             (PDIM×LDIM). Layout matches `Hpl.values()` (stride
	//             `PDIM*LDIM` per nnz slot). Only ext-range slots are written
	//             via atomic accumulation; body Hpl slots use ASSIGN and are
	//             not touched by the int64 path. The converter only propagates
	//             the ext-dedup slots back to `Hpl.values()`.
	// After the kernel, the caller propagates the covered ranges back to the
	// double buffers via the matching `convertFixedPoint*Range` helpers.
	// Remaining legacy atomic sites (Schur update) continue to use
	// `atomicAdd(double*)` and will be migrated in Phase 3f+.
	long long* Hpp_int_ext_raw = nullptr,
	long long* bp_int_ext_raw = nullptr,
	long long* Hll_int_raw = nullptr,
	long long* bl_int_raw = nullptr,
	long long* HscDirect_int_raw = nullptr,
	long long* Hpl_ext_int_raw = nullptr)
{
	using Vecmd = Vecxd<MDIM>;

	const int iE = blockIdx.x * blockDim.x + threadIdx.x;
	if (iE >= nedges)
		return;

	const int iP = edge2PL[iE][0];
	const int iL = edge2PL[iE][1];
	const int flag = flags[iE];

	const Vec4d& q = qs[iP];
	const Vec5d& camera = cameras[iP];
	const Vec3d& Xc = Xcs[iE];
	const Vecmd& error = errors[iE];

	// Robust kernel derivative
	const Scalar e = squaredNorm(error) * omegas[iE];
	const Scalar rho1 = robustKernel.derivative(e);
	const Scalar omega = omegas[iE] * rho1;

	// Exact Jacobians: per-edge extrinsics を反映した正確な Jacobian。
	// Xc_body を Xc と extrinsics の逆変換で復元する。
	// Xc = R_ext * Xc_body + t_ext → Xc_body = R_ext^T * (Xc - t_ext)
	const Vec4d& q_ext = q_exts[iE];
	const Vec3d& t_ext = t_exts[iE];
	Vec3d Xc_body;
	{
		// Inverse extrinsics: R_ext^T * (Xc - t_ext)
		Vec3d diff;
		diff[0] = Xc[0] - t_ext[0];
		diff[1] = Xc[1] - t_ext[1];
		diff[2] = Xc[2] - t_ext[2];
		// q_ext_inv = conjugate of q_ext (for unit quaternion)
		Vec4d q_ext_inv;
		q_ext_inv[0] = -q_ext[0]; q_ext_inv[1] = -q_ext[1]; q_ext_inv[2] = -q_ext[2]; q_ext_inv[3] = q_ext[3];
		rotate(q_ext_inv, diff, Xc_body);
	}
	Scalar JP[MDIM * PDIM];
	Scalar JL[MDIM * LDIM];

	// Select Jacobian: equidistant for 2D edges with non-zero distortion, pinhole otherwise
	if (MDIM == 2 && distortions != nullptr)
	{
		const Scalar* dist = distortions[iE].data;
		const bool use_equidistant = (dist[0] != 0 || dist[1] != 0 || dist[2] != 0 || dist[3] != 0);
		if (use_equidistant) {
			computeJacobiansExact_equidistant(Xc, Xc_body, q, q_ext, t_ext,
				MatView2x6d(JP), MatView2x3d(JL), camera, dist);
		} else {
			computeJacobiansExact<MDIM>(Xc, Xc_body, q, q_ext, t_ext, JP, JL, camera);
		}
	}
	else
	{
		computeJacobiansExact<MDIM>(Xc, Xc_body, q, q_ext, t_ext, JP, JL, camera);
	}

	if (!(flag & EDGE_FLAG_FIXED_P))
	{
		// Option 4 Phase 3b: deterministic body-range Hpp[iP] + bp[iP] when
		// the int64 mirror buffers are supplied. `iP` is a body-range index
		// in [0, numBody); Phase 2/3a ext-range writes use a separate iPExt
		// in the disjoint range [numBody, numBody+numExt), so body and ext
		// slots never collide. When the buffers are null we fall back to the
		// legacy `atomicAdd(double*)` path.
		if (Hpp_int_ext_raw != nullptr)
		{
			MatTMulMatDet<PDIM, MDIM, PDIM>(JP, JP,
				Hpp_int_ext_raw + iP * PDIM * PDIM, omega);
		}
		else
		{
			// Hpp += = JPT*Omega*JP
			MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JP, JP, Hpp.at(iP), omega);
		}
		if (bp_int_ext_raw != nullptr)
		{
			MatTMulVecDet<PDIM, MDIM>(JP, error.data,
				bp_int_ext_raw + iP * PDIM, omega);
		}
		else
		{
			// bp += = JPT*Omega*r
			MatTMulVec<PDIM, MDIM, ACCUM_ATOMIC>(JP, error.data, bp.at(iP), omega);
		}
	}
	if (!(flag & EDGE_FLAG_FIXED_L))
	{
		// Option 4 Phase 3c: deterministic landmark-range Hll[iL] + bl[iL].
		// Edge-to-landmark fanout is typically the highest in BA (each edge
		// writes to exactly one iL), so this site has historically been a
		// dominant source of atomicAdd non-determinism. LDIM=3, so
		// MatTMulMatDet<3, MDIM, 3> and MatTMulVecDet<3, MDIM> instantiate
		// cleanly from the existing primitives.
		if (Hll_int_raw != nullptr)
		{
			MatTMulMatDet<LDIM, MDIM, LDIM>(JL, JL,
				Hll_int_raw + iL * LDIM * LDIM, omega);
		}
		else
		{
			// Hll += = JLT*Omega*JL
			MatTMulMat<LDIM, MDIM, LDIM, ACCUM_ATOMIC>(JL, JL, Hll.at(iL), omega);
		}
		if (bl_int_raw != nullptr)
		{
			MatTMulVecDet<LDIM, MDIM>(JL, error.data,
				bl_int_raw + iL * LDIM, omega);
		}
		else
		{
			// bl += = JLT*Omega*r
			MatTMulVec<LDIM, MDIM, ACCUM_ATOMIC>(JL, error.data, bl.at(iL), omega);
		}
	}
	if (!(flag & (EDGE_FLAG_FIXED_P | EDGE_FLAG_FIXED_L)))
	{
		// Hpl += = JPT*Omega*JL (body). Unique per edge, ASSIGN is safe.
		MatTMulMat<PDIM, MDIM, LDIM, ASSIGN>(JP, JL, Hpl.at(edge2Hpl[iE]), omega);
	}

	// Joint ext contribution. JE has the same shape as JP_body but is anchored
	// on the ext vertex. JE = [Jpe · (-[Xc×]), Jpe] where Xc is the point in
	// the camera-optical frame (post-extrinsics). The existing
	// computeJacobiansExact*() kernels already produce Jpe internally; recompute
	// here using Xc directly. JE drops Rb from the chain because the ext Jacobian
	// is w.r.t. perturbation of camera_from_body, not body_from_world.
		if (!(flag & EDGE_FLAG_FIXED_E))
		{
			const int iPExt = edge2ExtIP[iE];
			const int hplExtSlot = edge2HplExt[iE];
			const int hscPESlot = edge2HscPE[iE];

		// Recompute Jpe locally. We need the camera-frame projection Jacobian
		// J_pi (2x3 or 3x3) multiplied by R_ext (3x3). The same quantities are
		// derived inside computeJacobiansExact, but we recompute them here to
		// keep the ext path self-contained and avoid changing the existing
		// Jacobian kernels' return signature.
		Scalar JE[MDIM * PDIM];

		// Build Jpi based on projection model.
		Scalar Jpi[MDIM * 3];
		const Scalar X = Xc[0];
		const Scalar Y = Xc[1];
		const Scalar Z = Xc[2];
		const Scalar fu = camera.data[0];
		const Scalar fv = camera.data[1];
		const Scalar bf = camera.data[4];
		bool use_equi = false;
		const Scalar* dist_ptr = nullptr;
		if (MDIM == 2 && distortions != nullptr)
		{
			dist_ptr = distortions[iE].data;
			use_equi = (dist_ptr[0] != 0 || dist_ptr[1] != 0 || dist_ptr[2] != 0 || dist_ptr[3] != 0);
		}

		if (MDIM == 2 && use_equi)
		{
			// Equidistant Jpi (negated to match convention downstream).
			const Scalar r = sqrt(X * X + Y * Y);
			const Scalar eps = Scalar(1e-10);
			if (r < eps)
			{
				const Scalar invZ = 1 / Z;
				const Scalar invZZ = invZ * invZ;
				Jpi[0 * 3 + 0] = -fu * invZ;  Jpi[0 * 3 + 1] = 0;          Jpi[0 * 3 + 2] = fu * X * invZZ;
				Jpi[1 * 3 + 0] = 0;           Jpi[1 * 3 + 1] = -fv * invZ;  Jpi[1 * 3 + 2] = fv * Y * invZZ;
			}
			else
			{
				const Scalar r2 = r * r;
				const Scalar r2_plus_Z2 = r2 + Z * Z;
				const Scalar theta = atan2(r, Z);
				const Scalar k1 = dist_ptr[0];
				const Scalar k2 = dist_ptr[1];
				const Scalar k3 = dist_ptr[2];
				const Scalar k4 = dist_ptr[3];
				const Scalar theta2 = theta * theta;
				const Scalar theta3 = theta2 * theta;
				const Scalar theta4 = theta2 * theta2;
				const Scalar theta5 = theta4 * theta;
				const Scalar theta6 = theta4 * theta2;
				const Scalar theta7 = theta6 * theta;
				const Scalar theta8 = theta4 * theta4;
				const Scalar theta9 = theta8 * theta;
				const Scalar theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;
				const Scalar dtheta_d_dtheta = 1 + 3 * k1 * theta2 + 5 * k2 * theta4
					+ 7 * k3 * theta6 + 9 * k4 * theta8;
				const Scalar dtheta_dX = X * Z / (r * r2_plus_Z2);
				const Scalar dtheta_dY = Y * Z / (r * r2_plus_Z2);
				const Scalar dtheta_dZ = -r / r2_plus_Z2;
				const Scalar dthetad_dX = dtheta_d_dtheta * dtheta_dX;
				const Scalar dthetad_dY = dtheta_d_dtheta * dtheta_dY;
				const Scalar dthetad_dZ = dtheta_d_dtheta * dtheta_dZ;
				const Scalar inv_r2 = 1 / r2;
				const Scalar ds_dX = (dthetad_dX * r - theta_d * (X / r)) * inv_r2;
				const Scalar ds_dY = (dthetad_dY * r - theta_d * (Y / r)) * inv_r2;
				const Scalar ds_dZ = dthetad_dZ / r;
				const Scalar s = theta_d / r;
				Jpi[0 * 3 + 0] = -(fu * (s + X * ds_dX));
				Jpi[0 * 3 + 1] = -(fu * X * ds_dY);
				Jpi[0 * 3 + 2] = -(fu * X * ds_dZ);
				Jpi[1 * 3 + 0] = -(fv * Y * ds_dX);
				Jpi[1 * 3 + 1] = -(fv * (s + Y * ds_dY));
				Jpi[1 * 3 + 2] = -(fv * Y * ds_dZ);
			}
		}
		else if (MDIM == 2)
		{
			const Scalar invZ = 1 / Z;
			const Scalar invZZ = invZ * invZ;
			Jpi[0 * 3 + 0] = -fu * invZ;  Jpi[0 * 3 + 1] = 0;          Jpi[0 * 3 + 2] = fu * X * invZZ;
			Jpi[1 * 3 + 0] = 0;           Jpi[1 * 3 + 1] = -fv * invZ;  Jpi[1 * 3 + 2] = fv * Y * invZZ;
		}
		else  // MDIM == 3: pinhole stereo
		{
			const Scalar invZ = 1 / Z;
			const Scalar invZZ = invZ * invZ;
			Jpi[0 * 3 + 0] = -fu * invZ;  Jpi[0 * 3 + 1] = 0;          Jpi[0 * 3 + 2] = fu * X * invZZ;
			Jpi[1 * 3 + 0] = 0;           Jpi[1 * 3 + 1] = -fv * invZ;  Jpi[1 * 3 + 2] = fv * Y * invZZ;
			Jpi[2 * 3 + 0] = -fu * invZ;  Jpi[2 * 3 + 1] = 0;          Jpi[2 * 3 + 2] = (fu * X + bf) * invZZ;
		}

		// JE rotation columns [0:3]: Jpi * (-[Xc×])  (Xc in camera-optical frame)
		// Note: unlike the body JP, here the cross is taken w.r.t. Xc (post-extrinsics)
		// because the ext perturbation acts directly on camera_from_body.
		MatView<Scalar, MDIM, PDIM> JEview(JE);
		for (int r = 0; r < MDIM; r++)
		{
			JEview(r, 0) = Jpi[r * 3 + 1] * (-Z) + Jpi[r * 3 + 2] * Y;
			JEview(r, 1) = Jpi[r * 3 + 0] * Z + Jpi[r * 3 + 2] * (-X);
			JEview(r, 2) = Jpi[r * 3 + 0] * (-Y) + Jpi[r * 3 + 1] * X;
			JEview(r, 3) = Jpi[r * 3 + 0];
			JEview(r, 4) = Jpi[r * 3 + 1];
			JEview(r, 5) = Jpi[r * 3 + 2];
		}

			if (iPExt >= 0)
			{
				if (Hpp_int_ext_raw != nullptr)
				{
					// Option 4 Phase 2: deterministic accumulation for the
					// `Hpp.at(iPExt)` block. The int64 buffer is laid out
					// identically to `Hpp` (PDIM * PDIM scalars per slot), so
					// `iPExt * PDIM * PDIM` gives the start of this slot.
					MatTMulMatDet<PDIM, MDIM, PDIM>(JE, JE,
						Hpp_int_ext_raw + iPExt * (PDIM * PDIM), omega);
				}
				else
				{
					MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JE, JE, Hpp.at(iPExt), omega);
				}
				if (bp_int_ext_raw != nullptr)
				{
					// Option 4 Phase 3a: deterministic accumulation for the
					// `bp.at(iPExt)` vector. The int64 buffer is laid out
					// identically to `bp` (PDIM scalars per slot), so
					// `iPExt * PDIM` gives the start of this slot.
					MatTMulVecDet<PDIM, MDIM>(JE, error.data,
						bp_int_ext_raw + iPExt * PDIM, omega);
				}
				else
				{
					MatTMulVec<PDIM, MDIM, ACCUM_ATOMIC>(JE, error.data, bp.at(iPExt), omega);
				}
			}
			if (!(flag & EDGE_FLAG_FIXED_P) && hscPESlot >= 0)
			{
				// Option 4 Phase 3d: deterministic accumulation for the direct
				// body×ext Schur cross-block `HscDirect.at(hscPESlot)`. The
				// int64 buffer is laid out identically to `HscDirect.values()`
				// (PDIM*PDIM scalars per slot), so `hscPESlot * (PDIM*PDIM)`
				// gives the start of this slot. Multiple edges may share the
				// same (iP, iPExt) slot when they project to the same
				// keyframe / camera pair.
				if (HscDirect_int_raw != nullptr)
				{
					MatTMulMatDet<PDIM, MDIM, PDIM>(JP, JE,
						HscDirect_int_raw + hscPESlot * (PDIM * PDIM), omega);
				}
				else
				{
					MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JP, JE, HscDirect.at(hscPESlot), omega);
				}
			}
			if (hplExtSlot >= 0 && !(flag & EDGE_FLAG_FIXED_L))
			{
				// Multiple edges may share the same (iP_ext, iL) Hpl slot.
				// Option 4 Phase 3e: route this accumulation through the int64
				// fixed-point path when `Hpl_ext_int_raw` is non-null. The
				// mirror buffer shares the full `Hpl.values()` layout
				// (`nnz * PDIM * LDIM` scalars) so `hplExtSlot * PDIM * LDIM`
				// gives the slot offset. Body Hpl slots are handled by the
				// ASSIGN branch above and are never touched through this
				// pointer. The caller invokes
				// `convertFixedPointHplExtSlots()` after the accumulation
				// kernel to write only the ext dedup slots back to the
				// double `Hpl` buffer, leaving body ASSIGN values intact.
				if (Hpl_ext_int_raw != nullptr)
				{
					MatTMulMatDet<PDIM, MDIM, LDIM>(JE, JL,
						Hpl_ext_int_raw + hplExtSlot * (PDIM * LDIM), omega);
				}
				else
				{
					MatTMulMat<PDIM, MDIM, LDIM, ACCUM_ATOMIC>(JE, JL, Hpl.at(hplExtSlot), omega);
				}
			}
	}
}

template <int MDIM>
__global__ void computeChiSquaresKernel(int nedges, const Vec4d* qs, const Vec3d* ts, const Vec5d* cameras,
	const Vec3d* Xws, const Vecxd<MDIM>* measurements, const Scalar* omegas, const Vec2i* edge2PL,
	const Vec4d* q_exts, const Vec3d* t_exts, const Vec4d* distortions, Scalar* chiSqs)
{
	using Vecmd = Vecxd<MDIM>;

	const int iE = blockIdx.x * blockDim.x + threadIdx.x;
	if (iE >= nedges)
		return;

	const Vec2i index = edge2PL[iE];
	const int iP = index[0];
	const int iL = index[1];

	const Vec4d& q = qs[iP];
	const Vec3d& t = ts[iP];
	const Vec5d& camera = cameras[iP];
	const Vec3d& Xw = Xws[iL];
	const Vecmd& measurement = measurements[iE];

	// project world to body frame
	Vec3d Xc_body;
	projectW2C(q, t, Xw, Xc_body);

	// apply per-edge extrinsics
	Vec3d Xc;
	applyExtrinsics(q_exts[iE], t_exts[iE], Xc_body, Xc);

	// project camera to image: equidistant if distortion is non-zero (2D only)
	Vecmd proj;
	if (MDIM == 2 && distortions != nullptr)
	{
		const Scalar* dist = distortions[iE].data;
		const bool use_equidistant = (dist[0] != 0 || dist[1] != 0 || dist[2] != 0 || dist[3] != 0);
		if (use_equidistant) {
			projectC2I_equidistant(Xc, *reinterpret_cast<Vec2d*>(&proj), camera, dist);
		} else {
			projectC2I(Xc, proj, camera);
		}
	}
	else
	{
		projectC2I(Xc, proj, camera);
	}

	// compute residual
	Vecmd error;
	for (int i = 0; i < MDIM; i++)
		error[i] = proj[i] - measurement[i];

	chiSqs[iE] = omegas[iE] * squaredNorm(error);
}

template <int DIM>
__global__ void maxDiagonalKernel(int size, const Scalar* D, Scalar* maxD)
{
	const int sharedIdx = threadIdx.x;
	__shared__ Scalar cache[BLOCK_MAX_DIAGONAL];

	Scalar maxVal = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
	{
		const int j = i / DIM;
		const int k = i % DIM;
		const Scalar* ptrBlock = D + j * DIM * DIM;
		maxVal = max(maxVal, ptrBlock[k * DIM + k]);
	}

	cache[sharedIdx] = maxVal;
	__syncthreads();

	for (int stride = BLOCK_MAX_DIAGONAL / 2; stride > 0; stride >>= 1)
	{
		if (sharedIdx < stride)
			cache[sharedIdx] = max(cache[sharedIdx], cache[sharedIdx + stride]);
		__syncthreads();
	}

	if (sharedIdx == 0)
		maxD[blockIdx.x] = cache[0];
}

template <int DIM>
__global__ void addLambdaKernel(int size, Scalar* D, Scalar lambda, Scalar* backup)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	const int j = i / DIM;
	const int k = i % DIM;
	Scalar* ptrBlock = D + j * DIM * DIM;
	backup[i] = ptrBlock[k * DIM + k];
	ptrBlock[k * DIM + k] += lambda;
}

template <int DIM>
__global__ void restoreDiagonalKernel(int size, Scalar* D, const Scalar* backup)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	const int j = i / DIM;
	const int k = i % DIM;
	Scalar* ptrBlock = D + j * DIM * DIM;
	ptrBlock[k * DIM + k] = backup[i];
}

__global__ void computeBschureKernel(int cols, LxLBlockPtr Hll, LxLBlockPtr invHll,
	Lx1BlockPtr bl, PxLBlockPtr Hpl, const int* HplColPtr, const int* HplRowInd,
	Px1BlockPtr bsc, PxLBlockPtr Hpl_invHll,
	long long* bsc_int_raw)
{
	const int colId = blockIdx.x * blockDim.x + threadIdx.x;
	if (colId >= cols)
		return;

	Scalar iHll[LDIM * LDIM];
	Scalar Hpl_iHll[PDIM * LDIM];

	Sym3x3Inv(Hll.at(colId), iHll);
	copy<LDIM * LDIM>(iHll, invHll.at(colId));

	for (int i = HplColPtr[colId]; i < HplColPtr[colId + 1]; i++)
	{
		MatMulMat<6, 3, 3>(Hpl.at(i), iHll, Hpl_iHll);
		if (bsc_int_raw != nullptr)
		{
			// Phase 3f: deterministic DEACCUM via fixed-point int64 atomics.
			// The caller zeros `bsc_int_raw` before this launch and invokes
			// `convertFixedPointBsc` afterwards to add the accumulated
			// decrement back into the already-initialized double `bsc`.
			MatMulVecDet<6, 3, 1>(Hpl_iHll, bl.at(colId),
				bsc_int_raw + HplRowInd[i] * PDIM, Scalar(-1));
		}
		else
		{
			MatMulVec<6, 3, 1, DEACCUM_ATOMIC>(Hpl_iHll, bl.at(colId), bsc.at(HplRowInd[i]));
		}
		copy<PDIM * LDIM>(Hpl_iHll, Hpl_invHll.at(i));
	}
}

__global__ void initializeHschurKernel(int rows, PxPBlockPtr Hpp, PxPBlockPtr Hsc, const int* HscRowPtr)
{
	const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowId >= rows)
		return;

	copy<PDIM * PDIM>(Hpp.at(rowId), Hsc.at(HscRowPtr[rowId]));
}

__global__ void computeHschureKernel(int size, const Vec3i* mulBlockIds,
	PxLBlockPtr Hpl_invHll, PxLBlockPtr Hpl, PxPBlockPtr Hschur,
	long long* Hschur_int_raw)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size)
		return;

	const Vec3i index = mulBlockIds[tid];
	// Sentinel set by findHschureMulBlockIndicesKernel when the Hpl (iP1, iP2)
	// pair does not have a corresponding Hschur entry (joint-ext structural mismatch).
	if (index[2] < 0)
		return;
	Scalar A[PDIM * LDIM];
	Scalar B[PDIM * LDIM];
	copy<PDIM * LDIM>(Hpl_invHll.at(index[0]), A);
	copy<PDIM * LDIM>(Hpl.at(index[1]), B);
	if (Hschur_int_raw != nullptr)
	{
		// Phase 3f: deterministic DEACCUM via fixed-point int64 atomics.
		// Caller zeros `Hschur_int_raw` before this launch and invokes
		// `convertFixedPointHsc` afterwards to add the accumulated
		// decrement back into the already-initialized double `Hschur`.
		MatMulMatTDet<6, 3, 6>(A, B,
			Hschur_int_raw + index[2] * (PDIM * PDIM), Scalar(-1));
	}
	else
	{
		MatMulMatT<6, 3, 6, DEACCUM_ATOMIC>(A, B, Hschur.at(index[2]));
	}
}

__global__ void findHschureMulBlockIndicesKernel(int cols, const int* HplColPtr, const int* HplRowInd,
	const int* HscRowPtr, const int* HscColInd, Vec3i* mulBlockIds, int* nindices,
	int mulBlockCapacity, int* overflow)
{
	const int colId = blockIdx.x * blockDim.x + threadIdx.x;
	if (colId >= cols)
		return;

	const int i0 = HplColPtr[colId];
	const int i1 = HplColPtr[colId + 1];
	for (int i = i0; i < i1; i++)
	{
		const int iP1 = HplRowInd[i];
		const int kRowEnd = HscRowPtr[iP1 + 1];
		int k = HscRowPtr[iP1];
		for (int j = i; j < i1; j++)
		{
			const int iP2 = HplRowInd[j];
			// Bounded walk: stop at the row boundary to prevent reading into the
			// next row's column data (cudaErrorInvalidAddressSpace in joint-ext
			// mode when Hpl gains ext iP rows whose Hschur entries are not strict
			// supersets of body cross-blocks on every landmark column).
			while (k < kRowEnd && HscColInd[k] < iP2) k++;
			if (k >= kRowEnd || HscColInd[k] != iP2)
			{
				// No matching Hschur entry. Emit a sentinel index so downstream
				// computeHschureKernel can skip this multiplication instead of
				// scattering into an unrelated block.
				const int pos = atomicAdd(nindices, 1);
				if (pos < mulBlockCapacity)
				{
					mulBlockIds[pos] = makeVec3i(i, j, -1);
				}
				else
				{
					atomicExch(overflow, 1);
				}
			}
			else
			{
				const int pos = atomicAdd(nindices, 1);
				if (pos < mulBlockCapacity)
				{
					mulBlockIds[pos] = makeVec3i(i, j, k);
				}
				else
				{
					atomicExch(overflow, 1);
				}
			}
		}
	}
}

__global__ void permuteNnzPerRowKernel(int size, const int* srcRowPtr, const int* P, int* nnzPerRow)
{
	const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowId >= size)
		return;

	nnzPerRow[P[rowId]] = srcRowPtr[rowId + 1] - srcRowPtr[rowId];
}

__global__ void permuteColIndKernel(int size, const int* srcRowPtr, const int* srcColInd, const int* P,
	int* dstColInd, int* dstMap, int* nnzPerRow)
{
	const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowId >= size)
		return;

	const int i0 = srcRowPtr[rowId];
	const int i1 = srcRowPtr[rowId + 1];
	const int permRowId = P[rowId];
	for (int srck = i0; srck < i1; srck++)
	{
		const int dstk = nnzPerRow[permRowId]++;
		dstColInd[dstk] = P[srcColInd[srck]];
		dstMap[dstk] = srck;
	}
}

__global__ void schurComplementPostKernel(int cols, LxLBlockPtr invHll, Lx1BlockPtr bl, PxLBlockPtr Hpl,
	const int* HplColPtr, const int* HplRowInd, Px1BlockPtr xp, Lx1BlockPtr xl)
{
	const int colId = blockIdx.x * blockDim.x + threadIdx.x;
	if (colId >= cols)
		return;

	Scalar cl[LDIM];
	copy<LDIM>(bl.at(colId), cl);

	for (int i = HplColPtr[colId]; i < HplColPtr[colId + 1]; i++)
		MatTMulVec<3, 6, DEACCUM>(Hpl.at(i), xp.at(HplRowInd[i]), cl, 1);

	MatMulVec<3, 3>(invHll.at(colId), cl, xl.at(colId));
}

__global__ void updatePosesKernel(int size, Px1BlockPtr xp, Vec4d* qs, Vec3d* ts)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	Vec4d expq;
	Vec3d expt;
	updateExp(xp.at(i), expq, expt);
	updatePose(expq, expt, qs[i], ts[i]);
}

// Joint-mode variant with per-slot clamping for ext vertices (i >= num_body).
// Reads the 6D delta from xp, clamps rotation/translation separately to the
// given bounds, and then retracts onto SE(3) using the existing updateExp /
// updatePose primitives. Body slots (i < num_body) take the unclamped path.
__global__ void updatePosesKernelJoint(int size, int num_body, Scalar max_trans, Scalar max_rot,
	Px1BlockPtr xp, Vec4d* qs, Vec3d* ts)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	Scalar delta[6];
	const Scalar* xpi = xp.at(i);
	for (int k = 0; k < 6; k++) delta[k] = xpi[k];

	if (i >= num_body)
	{
		// Rotation in first 3, translation in last 3 (matches updateExp layout).
		const Scalar rn = sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
		if (rn > max_rot && rn > 0)
		{
			const Scalar s = max_rot / rn;
			delta[0] *= s; delta[1] *= s; delta[2] *= s;
		}
		const Scalar tn = sqrt(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);
		if (tn > max_trans && tn > 0)
		{
			const Scalar s = max_trans / tn;
			delta[3] *= s; delta[4] *= s; delta[5] *= s;
		}
	}

	Vec4d expq;
	Vec3d expt;
	updateExp(delta, expq, expt);
	updatePose(expq, expt, qs[i], ts[i]);
}

// Joint-mode: add prior diagonal to Hpp[iP] for iP in [num_body, num_body + num_ext).
// Rotation DOFs [0..3) get lambda_rot, translation DOFs [3..6) get lambda_trans.
__global__ void addExtPriorKernel(int num_ext, int num_body, Scalar lambda_rot, Scalar lambda_trans,
	PxPBlockPtr Hpp)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = num_ext * PDIM;
	if (tid >= total)
		return;
	const int localIdx = tid / PDIM;
	const int dof = tid % PDIM;
	const int iP = num_body + localIdx;
	const Scalar lambda = (dof < 3) ? lambda_rot : lambda_trans;
	Scalar* block = Hpp.at(iP);
	block[dof * PDIM + dof] += lambda;
}

// Joint-mode: build per-edge ext Hpl slot by dereferencing the dedup slot ids.
//   out[iE] = (dedup[iE] < 0) ? -1 : edge2Hpl[nedges + dedup[iE]]
// Default OFF / no ext dedup slots: caller invokes fillEdge2HplExtSentinel instead.
__global__ void buildEdgeExtHplKernel(int nedges, int nedges_total, const int* dedup_slot_per_edge,
	const int* edge2Hpl, int* edge2HplExt)
{
	const int iE = blockIdx.x * blockDim.x + threadIdx.x;
	if (iE >= nedges)
		return;
	const int slot = dedup_slot_per_edge[iE];
	edge2HplExt[iE] = (slot < 0) ? -1 : edge2Hpl[nedges_total + slot];
}

__global__ void fillIntSentinelKernel(int n, int sentinel, int* data)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	data[i] = sentinel;
}

// Scatter current ext solution (qs[iPExt] / ts[iPExt]) into per-edge q_exts / t_exts
// for edges whose extrinsics vertex participates in the joint solve.
// Edges with edge2ExtIP[iE] < 0 retain their existing q_exts/t_exts (set once at init).
__global__ void syncExtSolutionToPerEdgeKernel(int nedges, const Vec4d* qs, const Vec3d* ts,
	const int* edge2ExtIP, Vec4d* q_exts, Vec3d* t_exts)
{
	const int iE = blockIdx.x * blockDim.x + threadIdx.x;
	if (iE >= nedges) return;
	const int iPExt = edge2ExtIP[iE];
	if (iPExt < 0) return;
	q_exts[iE] = qs[iPExt];
	t_exts[iE] = ts[iPExt];
}

__global__ void updateLandmarksKernel(int size, Lx1BlockPtr xl, Vec3d* Xws)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	const Scalar* dXw = xl.at(i);
	Vec3d& Xw = Xws[i];
	Xw[0] += dXw[0];
	Xw[1] += dXw[1];
	Xw[2] += dXw[2];
}

__global__ void computeScaleKernel(const Scalar* x, const Scalar* b, Scalar* scale, Scalar lambda, int size,
	// Option 4 Phase 3g: deterministic fixed-point accumulator mirror, same
	// shape as `chi_int` in computeActiveErrorsKernel. When non-null, the
	// final block-reduced scalar is routed through `atomicAccumDet` instead
	// of `atomicAdd(double*)` so the LM denominator (`scale + 1e-3`) becomes
	// bit-identical across runs.
	long long* scale_int)
{
	const int sharedIdx = threadIdx.x;
	__shared__ Scalar cache[BLOCK_COMPUTE_SCALE];

	Scalar sum = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
		sum += x[i] * (lambda * x[i] + b[i]);

	cache[sharedIdx] = sum;
	__syncthreads();

	for (int stride = BLOCK_COMPUTE_SCALE / 2; stride > 0; stride >>= 1)
	{
		if (sharedIdx < stride)
			cache[sharedIdx] += cache[sharedIdx + stride];
		__syncthreads();
	}

	if (sharedIdx == 0)
	{
		if (scale_int != nullptr)
			deterministic::atomicAccumDet(scale_int, cache[0]);
		else
			atomicAdd(scale, cache[0]);
	}
}

__global__ void convertBSRToCSRKernel(int size, const Scalar* src, Scalar* dst, const int* map)
{
	const int dstk = blockIdx.x * blockDim.x + threadIdx.x;
	if (dstk >= size)
		return;

	dst[dstk] = src[map[dstk]];
}

__global__ void nnzPerColKernel(const Vec3i* blockpos, int nblocks, int* nnzPerCol)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nblocks)
		return;

	const int colId = blockpos[i][1];
	atomicAdd(&nnzPerCol[colId], 1);
}

__global__ void setRowIndKernel(const Vec3i* blockpos, int nblocks, int* rowInd, int* indexPL)
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= nblocks)
		return;

	const int rowId = blockpos[k][0];
	const int edgeId = blockpos[k][2];
	rowInd[k] = rowId;
	indexPL[edgeId] = k;
}

__global__ void solveDiagonalSystemKernel(int size, LxLBlockPtr Hll, Lx1BlockPtr bl, Lx1BlockPtr xl)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	solveSym3x3(Hll.at(i), bl.at(i), xl.at(i));
}

__global__ void solveDiagonalSystemKernel(int size, PxPBlockPtr Hpp, Px1BlockPtr bp, Px1BlockPtr xp)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	solveSym6x6(Hpp.at(i), bp.at(i), xp.at(i));
}

////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

void waitForKernelCompletion()
{
	prepareCudaThreadContext();
	CUDA_CHECK(cudaDeviceSynchronize());
}

void exclusiveScan(const int* src, int* dst, int size)
{
	prepareCudaThreadContext();
	auto ptrSrc = thrust::device_pointer_cast(src);
	auto ptrDst = thrust::device_pointer_cast(dst);
	thrust::exclusive_scan(ptrSrc, ptrSrc + size, ptrDst);
}

void buildHplStructure(GpuVec3i& blockpos, GpuHplBlockMat& Hpl, GpuVec1i& indexPL, GpuVec1i& nnzPerCol)
{
	prepareCudaThreadContext();
	const int nblocks = Hpl.nnz();
	const int block = 1024;
	const int grid = divUp(nblocks, block);
	int* colPtr = Hpl.outerIndices();
	int* rowInd = Hpl.innerIndices();

	auto ptrBlockPos = thrust::device_pointer_cast(blockpos.data());
	// Option 4 Phase 5: duplicate Hpl slots are valid when multiple edges contribute
	// to the same (row, col) block. Sort by a total (col, row, edgeId) key so the
	// per-edge `indexPL[edgeId] = k` mapping does not depend on Thrust's handling of
	// equivalent keys or on the launch order that produced the block list.
	thrust::stable_sort(ptrBlockPos, ptrBlockPos + nblocks, LessColId());

	CUDA_CHECK(cudaMemset(nnzPerCol, 0, sizeof(int) * (Hpl.cols() + 1)));
	nnzPerColKernel<<<grid, block>>>(blockpos, nblocks, nnzPerCol);
	exclusiveScan(nnzPerCol, colPtr, Hpl.cols() + 1);
	setRowIndKernel<<<grid, block>>>(blockpos, nblocks, rowInd, indexPL);
}

void findHschureMulBlockIndices(const GpuHplBlockMat& Hpl, const GpuHscBlockMat& Hsc,
	GpuVec3i& mulBlockIds)
{
	prepareCudaThreadContext();
	const int mulBlockCapacity = mulBlockIds.ssize();
	if (Hpl.cols() <= 0 || mulBlockCapacity <= 0)
	{
		return;
	}
	const int block = 1024;
	const int grid = divUp(Hpl.cols(), block);

	DeviceBuffer<int> nindices(1);
	nindices.fillZero();
	DeviceBuffer<int> overflow(1);
	overflow.fillZero();

	// Initialize every Vec3i slot to (-1, -1, -1) before the kernel populates a
	// prefix of the buffer via atomicAdd(nindices, 1). The caller sizes the buffer
	// from the actual Hpl row-slot pair enumeration per landmark column, not from
	// Hsc_.nmulBlocks(), because Hsc stores unique Schur row pairs while Hpl can
	// contain duplicate row slots introduced by multi-camera / joint-ext edges.
	// Writing 0xFF bytes across the entire buffer sets each int to -1, which the
	// sentinel check in computeHschureKernel (index[2] < 0) skips.
	CUDA_CHECK(cudaMemset(mulBlockIds.data(), 0xFF,
		sizeof(Vec3i) * mulBlockIds.size()));

	findHschureMulBlockIndicesKernel<<<grid, block>>>(Hpl.cols(), Hpl.outerIndices(), Hpl.innerIndices(),
		Hsc.outerIndices(), Hsc.innerIndices(), mulBlockIds, nindices, mulBlockCapacity, overflow);
	CUDA_CHECK(cudaGetLastError());
	int hostNindices = 0;
	int hostOverflow = 0;
	nindices.download(&hostNindices);
	overflow.download(&hostOverflow);
	if (hostOverflow != 0 || hostNindices > mulBlockCapacity)
	{
		throw std::runtime_error(
			"findHschureMulBlockIndices overflow: writes=" + std::to_string(hostNindices) +
			", capacity=" + std::to_string(mulBlockCapacity));
	}

	auto ptrSrc = thrust::device_pointer_cast(mulBlockIds.data());
	// Option 4 Phase 5: use a total (row, col, hplPairSlot) order. The kernel writes
	// the prefix through atomicAdd(nindices, 1), so preserving equivalent-key prefix
	// order is not sufficient for deterministic Schur construction.
	thrust::stable_sort(ptrSrc, ptrSrc + mulBlockIds.size(), LessRowId());
}

template <int MDIM, int RK_TYPE = 0>
Scalar computeActiveErrors_(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVecAny& _measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions,
	Scalar robustDelta,
	const GpuVecAny& _errors, GpuVec3d& Xcs, Scalar* chi, long long* chi_int)
{
	prepareCudaThreadContext();
	const auto& measurements = _measurements.getCRef<Vecxd<MDIM>>();
	auto& errors = _errors.getRef<Vecxd<MDIM>>();
	const RobustKernelFunc<RK_TYPE> robustKernel(robustDelta);

	const int nedges = measurements.ssize();
	const int block = BLOCK_ACTIVE_ERRORS;
	const int grid = 16;

	if (nedges <= 0)
		return 0;

	// Pass distortion pointer only for 2D edges that have data uploaded
	const Vec4d* d_dist_ptr = (MDIM == 2 && distortions.size() > 0)
		? static_cast<const Vec4d*>(distortions) : nullptr;

	// Option 4 Phase 3g: when `chi_int` is provided, route the final reduction
	// through the deterministic int64 accumulator. The legacy `chi` double
	// buffer is still zeroed/read for callers that haven't migrated (and as a
	// no-op guard in case the kernel hits the double branch).
	if (chi_int != nullptr)
		CUDA_CHECK(cudaMemset(chi_int, 0, sizeof(long long)));
	CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
	computeActiveErrorsKernel<MDIM, RK_TYPE><<<grid, block>>>(nedges, qs, ts, cameras, Xws, measurements, omegas,
		edge2PL, q_exts, t_exts, d_dist_ptr, robustKernel, errors, Xcs, chi, chi_int);
	CUDA_CHECK(cudaGetLastError());

	if (chi_int != nullptr)
	{
		long long h_chi_int = 0;
		CUDA_CHECK(cudaMemcpy(&h_chi_int, chi_int, sizeof(long long), cudaMemcpyDeviceToHost));
		return deterministic::fromFixedPoint(h_chi_int);
	}

	Scalar h_chi = 0;
	CUDA_CHECK(cudaMemcpy(&h_chi, chi, sizeof(Scalar), cudaMemcpyDeviceToHost));

	return h_chi;
}

using ComputeActiveErrorsFunc = Scalar(*)(const GpuVec4d&, const GpuVec3d&, const GpuVec5d&, const GpuVec3d&,
	const GpuVecAny&, const GpuVec1d&, const GpuVec2i&, const GpuVec4d&, const GpuVec3d&, const GpuVec4d&,
	Scalar, const GpuVecAny&, GpuVec3d&, Scalar*, long long*);

static ComputeActiveErrorsFunc computeActiveErrorsFuncs[6] =
{
	computeActiveErrors_<2, 0>,
	computeActiveErrors_<2, 1>,
	computeActiveErrors_<2, 2>,
	computeActiveErrors_<3, 0>,
	computeActiveErrors_<3, 1>,
	computeActiveErrors_<3, 2>
};

Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec2d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions,
	const RobustKernel& kernel,
	GpuVec2d& errors, GpuVec3d& Xcs, Scalar* chi, long long* chi_int)
{
	auto func = computeActiveErrorsFuncs[0 + kernel.type];
	return func(qs, ts, cameras, Xws, measurements, omegas, edge2PL, q_exts, t_exts, distortions, kernel.delta, errors, Xcs, chi, chi_int);
}

Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec3d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts,
	const RobustKernel& kernel,
	GpuVec3d& errors, GpuVec3d& Xcs, Scalar* chi, long long* chi_int)
{
	// 3D (stereo) path: no distortion support, pass empty GpuVec4d
	static GpuVec4d empty_distortions;
	auto func = computeActiveErrorsFuncs[3 + kernel.type];
	return func(qs, ts, cameras, Xws, measurements, omegas, edge2PL, q_exts, t_exts, empty_distortions, kernel.delta, errors, Xcs, chi, chi_int);
}

template <int MDIM, int RK_TYPE = 0>
void constructQuadraticForm_(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec5d& cameras, const GpuVecAny& _errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1i& edge2HplExt,
	const GpuVec1i& edge2ExtIP, const GpuVec1i& edge2HscPE, const GpuVec1b& flags,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions,
	Scalar robustDelta,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl,
	GpuHscBlockMat& HscDirect,
	long long* Hpp_int_ext_raw,
	long long* bp_int_ext_raw,
	long long* Hll_int_raw,
	long long* bl_int_raw,
	long long* HscDirect_int_raw,
	long long* Hpl_ext_int_raw)
{
	const auto& errors = _errors.getRef<Vecxd<MDIM>>();
	const RobustKernelFunc<RK_TYPE> robustKernel(robustDelta);

	const int nedges = errors.ssize();
	const int block = 512;
	const int grid = divUp(nedges, block);

	if (nedges <= 0)
		return;

	// Pass distortion pointer only for 2D edges that have data uploaded
	const Vec4d* d_dist_ptr = (MDIM == 2 && distortions.size() > 0)
		? static_cast<const Vec4d*>(distortions) : nullptr;

	constructQuadraticFormKernel<MDIM, RK_TYPE><<<grid, block>>>(nedges, Xcs, qs, cameras, errors, omegas,
		edge2PL, edge2Hpl, edge2HplExt, edge2ExtIP, edge2HscPE, flags, robustKernel, Hpp, bp, Hll, bl, Hpl, HscDirect,
		q_exts, t_exts, d_dist_ptr, Hpp_int_ext_raw, bp_int_ext_raw, Hll_int_raw, bl_int_raw, HscDirect_int_raw, Hpl_ext_int_raw);
	CUDA_CHECK(cudaGetLastError());
}

using ConstructQuadraticFormFunc = void(*)(const GpuVec3d&, const GpuVec4d&, const GpuVec5d&, const GpuVecAny&,
	const GpuVec1d&, const GpuVec2i&, const GpuVec1i&, const GpuVec1i&, const GpuVec1i&, const GpuVec1i&, const GpuVec1b&,
	const GpuVec4d&, const GpuVec3d&, const GpuVec4d&, Scalar,
	GpuPxPBlockVec&, GpuPx1BlockVec&, GpuLxLBlockVec&, GpuLx1BlockVec&, GpuHplBlockMat&, GpuHscBlockMat&,
	long long*, long long*, long long*, long long*, long long*, long long*);

static ConstructQuadraticFormFunc constructQuadraticFormFuncs[6] =
{
	constructQuadraticForm_<2, 0>,
	constructQuadraticForm_<2, 1>,
	constructQuadraticForm_<2, 2>,
	constructQuadraticForm_<3, 0>,
	constructQuadraticForm_<3, 1>,
	constructQuadraticForm_<3, 2>
};

void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec5d& cameras, const GpuVec2d& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1i& edge2HplExt,
	const GpuVec1i& edge2ExtIP, const GpuVec1i& edge2HscPE, const GpuVec1b& flags,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions,
	const RobustKernel& kernel,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl,
	GpuHscBlockMat& HscDirect,
	long long* Hpp_int_ext_raw,
	long long* bp_int_ext_raw,
	long long* Hll_int_raw,
	long long* bl_int_raw,
	long long* HscDirect_int_raw,
	long long* Hpl_ext_int_raw)
{
	auto func = constructQuadraticFormFuncs[0 + kernel.type];
	func(Xcs, qs, cameras, errors, omegas, edge2PL, edge2Hpl, edge2HplExt, edge2ExtIP, edge2HscPE, flags, q_exts, t_exts, distortions, kernel.delta, Hpp, bp, Hll, bl, Hpl, HscDirect, Hpp_int_ext_raw, bp_int_ext_raw, Hll_int_raw, bl_int_raw, HscDirect_int_raw, Hpl_ext_int_raw);
}

void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec5d& cameras, const GpuVec3d& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1i& edge2HplExt,
	const GpuVec1i& edge2ExtIP, const GpuVec1i& edge2HscPE, const GpuVec1b& flags,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts,
	const RobustKernel& kernel,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl,
	GpuHscBlockMat& HscDirect,
	long long* Hpp_int_ext_raw,
	long long* bp_int_ext_raw,
	long long* Hll_int_raw,
	long long* bl_int_raw,
	long long* HscDirect_int_raw,
	long long* Hpl_ext_int_raw)
{
	// 3D (stereo) path: no distortion support, pass empty GpuVec4d
	static GpuVec4d empty_distortions;
	auto func = constructQuadraticFormFuncs[3 + kernel.type];
	func(Xcs, qs, cameras, errors, omegas, edge2PL, edge2Hpl, edge2HplExt, edge2ExtIP, edge2HscPE, flags, q_exts, t_exts, empty_distortions, kernel.delta, Hpp, bp, Hll, bl, Hpl, HscDirect, Hpp_int_ext_raw, bp_int_ext_raw, Hll_int_raw, bl_int_raw, HscDirect_int_raw, Hpl_ext_int_raw);
}

template <int MDIM>
void computeChiSquares_(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVecAny& _measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions, GpuVec1d& chiSqs)
{
	prepareCudaThreadContext();
	using Vecmd = Vecxd<MDIM>;

	const GpuVec<Vecmd>& measurements = _measurements.getCRef<Vecmd>();

	const int nedges = measurements.ssize();
	const int block = 512;
	const int grid = divUp(nedges, block);

	if (nedges <= 0)
		return;

	// Pass distortion pointer only for 2D edges that have data uploaded
	const Vec4d* d_dist_ptr = (MDIM == 2 && distortions.size() > 0)
		? static_cast<const Vec4d*>(distortions) : nullptr;

	computeChiSquaresKernel<MDIM><<<grid, block>>>(nedges, qs, ts, cameras, Xws, measurements, omegas, edge2PL,
		q_exts, t_exts, d_dist_ptr, chiSqs);
	CUDA_CHECK(cudaGetLastError());
}

void computeChiSquares(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec2d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, const GpuVec4d& distortions, GpuVec1d& chiSqs)
{
	computeChiSquares_<2>(qs, ts, cameras, Xws, measurements, omegas, edge2PL, q_exts, t_exts, distortions, chiSqs);
}

void computeChiSquares(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec5d& cameras, const GpuVec3d& Xws,
	const GpuVec3d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	const GpuVec4d& q_exts, const GpuVec3d& t_exts, GpuVec1d& chiSqs)
{
	// 3D (stereo) path: no distortion support
	static GpuVec4d empty_distortions;
	computeChiSquares_<3>(qs, ts, cameras, Xws, measurements, omegas, edge2PL, q_exts, t_exts, empty_distortions, chiSqs);
}

template <typename T, int DIM>
Scalar maxDiagonal_(const DeviceBlockVector<T, DIM, DIM>& D, Scalar* maxD)
{
	if (!D.size())
		return 0;

	constexpr int block = BLOCK_MAX_DIAGONAL;
	constexpr int grid = 4;
	const int size = D.size() * DIM;

	maxDiagonalKernel<DIM><<<grid, block>>>(size, D.values(), maxD);
	CUDA_CHECK(cudaGetLastError());

	Scalar tmpMax[grid];
	CUDA_CHECK(cudaMemcpy(tmpMax, maxD, sizeof(Scalar) * grid, cudaMemcpyDeviceToHost));

	Scalar maxv = 0;
	for (int i = 0; i < grid; i++)
		maxv = std::max(maxv, tmpMax[i]);

	return maxv;
}

Scalar maxDiagonal(const GpuPxPBlockVec& Hpp, Scalar* maxD)
{
	return maxDiagonal_(Hpp, maxD);
}

Scalar maxDiagonal(const GpuLxLBlockVec& Hll, Scalar* maxD)
{
	return maxDiagonal_(Hll, maxD);
}

template <typename T, int DIM>
void addLambda_(DeviceBlockVector<T, DIM, DIM>& D, Scalar lambda, DeviceBlockVector<T, DIM, 1>& backup)
{
	if (!D.size())
		return;

	const int size = D.size() * DIM;
	const int block = 1024;
	const int grid = divUp(size, block);
	addLambdaKernel<DIM><<<grid, block>>>(size, D.values(), lambda, backup.values());
	CUDA_CHECK(cudaGetLastError());
}

void addLambda(GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup)
{
	addLambda_(Hpp, lambda, backup);
}

void addLambda(GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup)
{
	addLambda_(Hll, lambda, backup);
}

template <typename T, int DIM>
void restoreDiagonal_(DeviceBlockVector<T, DIM, DIM>& D, const DeviceBlockVector<T, DIM, 1>& backup)
{
	if (!D.size())
		return;

	const int size = D.size() * DIM;
	const int block = 1024;
	const int grid = divUp(size, block);
	restoreDiagonalKernel<DIM><<<grid, block>>>(size, D.values(), backup.values());
	CUDA_CHECK(cudaGetLastError());
}

void restoreDiagonal(GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup)
{
	restoreDiagonal_(Hpp, backup);
}

void restoreDiagonal(GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup)
{
	restoreDiagonal_(Hll, backup);
}

void computeBschure(const GpuPx1BlockVec& bp, const GpuHplBlockMat& Hpl, const GpuLxLBlockVec& Hll,
	const GpuLx1BlockVec& bl, GpuPx1BlockVec& bsc, GpuLxLBlockVec& invHll, GpuPxLBlockVec& Hpl_invHll,
	long long* bsc_int_raw)
{
	prepareCudaThreadContext();
	const int cols = Hll.size();
	const int block = 256;
	const int grid = divUp(cols, block);

	// Phase 3f note: When the deterministic path is active, the caller must
	// zero `bsc_int_raw` prior to this call. `bp.copyTo(bsc)` initializes the
	// double buffer with the initial bp contribution; the kernel accumulates
	// only the Hpl*invHll*bl decrement into `bsc_int_raw`, and the matching
	// `convertFixedPointBsc` call adds the fixed-point decrement back into
	// `bsc` (additive propagate).
	bp.copyTo(bsc);
	computeBschureKernel<<<grid, block>>>(cols, Hll, invHll, bl, Hpl, Hpl.outerIndices(), Hpl.innerIndices(),
		bsc, Hpl_invHll, bsc_int_raw);
	CUDA_CHECK(cudaGetLastError());
}

void computeHschure(const GpuPxPBlockVec& Hpp, const GpuHscBlockMat& HscDirect, const GpuPxLBlockVec& Hpl_invHll,
	const GpuHplBlockMat& Hpl, const GpuVec3i& mulBlockIds, GpuHscBlockMat& Hsc,
	long long* Hschur_int_raw)
{
	prepareCudaThreadContext();
	const int nmulBlocks = mulBlockIds.ssize();
	const int block = 256;
	const int grid1 = divUp(Hsc.rows(), block);
	const int grid2 = divUp(nmulBlocks, block);

	// Phase 3f note: The pre-kernel initialization of `Hsc` (cudaMemcpy from
	// HscDirect, then Hpp-diagonal add) is deterministic on its own (no
	// atomics on overlapping addresses). Only the subsequent Schur complement
	// update via `computeHschureKernel` uses DEACCUM_ATOMIC; when
	// `Hschur_int_raw` is non-null, that update is routed through int64
	// accumulation and the paired `convertFixedPointHsc` helper adds the
	// fixed-point decrement back into `Hsc` after the kernel completes.
	CUDA_CHECK(cudaMemcpy(Hsc.values(), HscDirect.values(),
		sizeof(Scalar) * Hsc.nnz() * GpuHscBlockMat::BLOCK_AREA,
		cudaMemcpyDeviceToDevice));
	initializeHschurKernel<<<grid1, block>>>(Hsc.rows(), Hpp, Hsc, Hsc.outerIndices());
	computeHschureKernel<<<grid2, block>>>(nmulBlocks, mulBlockIds, Hpl_invHll, Hpl, Hsc, Hschur_int_raw);
	CUDA_CHECK(cudaGetLastError());
}

// Phase 3f: additive propagation from fixed-point int64 buffer back into the
// double `bsc` vector. The double `bsc` already holds `bp.copyTo(bsc)` at
// call time, so this helper performs `bsc[i] += fromFixedPoint(src_int[i])`
// over `numP*PDIM` entries to complete the deterministic DEACCUM.
__global__ void fixedPointAddToDoubleBscKernel(int n,
	const long long* src_int, Scalar* dst)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	dst[i] += deterministic::fromFixedPoint(src_int[i]);
}

void convertFixedPointBsc(const long long* src_int, GpuPx1BlockVec& bsc, int numP)
{
	if (numP <= 0) return;
	const int n = numP * PDIM;
	const int block = 1024;
	const int grid = divUp(n, block);
	fixedPointAddToDoubleBscKernel<<<grid, block>>>(n, src_int, bsc.values());
	CUDA_CHECK(cudaGetLastError());
}

// Phase 3f: additive propagation from fixed-point int64 buffer back into the
// double `Hsc` block matrix. `Hsc.values()` already holds `HscDirect + Hpp`
// diagonal at call time; this helper adds the accumulated Schur-complement
// decrement `-Hpl*invHll*HplT` over `nnz*PDIM*PDIM` entries.
__global__ void fixedPointAddToDoubleHscKernel(int n,
	const long long* src_int, Scalar* dst)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	dst[i] += deterministic::fromFixedPoint(src_int[i]);
}

void convertFixedPointHsc(const long long* src_int, GpuHscBlockMat& Hsc, int nnz)
{
	if (nnz <= 0) return;
	const int n = nnz * GpuHscBlockMat::BLOCK_AREA;
	const int block = 1024;
	const int grid = divUp(n, block);
	fixedPointAddToDoubleHscKernel<<<grid, block>>>(n, src_int, Hsc.values());
	CUDA_CHECK(cudaGetLastError());
}

void convertHschureBSRToCSR(const GpuHscBlockMat& HscBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HscCSR)
{
	const int size = HscCSR.ssize();
	const int block = 1024;
	const int grid = divUp(size, block);
	convertBSRToCSRKernel<<<grid, block>>>(size, HscBSR.values(), HscCSR, BSR2CSR);
}

void twistCSR(int size, int nnz, const int* srcRowPtr, const int* srcColInd, const int* P,
	int* dstRowPtr, int* dstColInd, int* dstMap, int* nnzPerRow)
{
	const int block = 512;
	const int grid = divUp(size, block);

	permuteNnzPerRowKernel<<<grid, block>>>(size, srcRowPtr, P, nnzPerRow);
	exclusiveScan(nnzPerRow, dstRowPtr, size + 1);
	CUDA_CHECK(cudaMemcpy(nnzPerRow, dstRowPtr, sizeof(int) * (size + 1), cudaMemcpyDeviceToDevice));
	permuteColIndKernel<<<grid, block>>>(size, srcRowPtr, srcColInd, P, dstColInd, dstMap, nnzPerRow);
}

void permute(int size, const Scalar* src, Scalar* dst, const int* P)
{
	prepareCudaThreadContext();
	auto ptrSrc = thrust::device_pointer_cast(src);
	auto ptrDst = thrust::device_pointer_cast(dst);
	auto ptrMap = thrust::device_pointer_cast(P);
	thrust::gather(ptrMap, ptrMap + size, ptrSrc, ptrDst);
}

void schurComplementPost(const GpuLxLBlockVec& invHll, const GpuLx1BlockVec& bl,
	const GpuHplBlockMat& Hpl, const GpuPx1BlockVec& xp, GpuLx1BlockVec& xl)
{
	prepareCudaThreadContext();
	const int block = 1024;
	const int grid = divUp(Hpl.cols(), block);

	schurComplementPostKernel<<<grid, block>>>(Hpl.cols(), invHll, bl, Hpl,
		Hpl.outerIndices(), Hpl.innerIndices(),xp, xl);
	CUDA_CHECK(cudaGetLastError());
}

void updatePoses(const GpuPx1BlockVec& xp, GpuVec4d& qs, GpuVec3d& ts)
{
	if (!xp.size())
		return;

	const int block = 256;
	const int grid = divUp(xp.size(), block);
	updatePosesKernel<<<grid, block>>>(xp.size(), xp, qs, ts);
	CUDA_CHECK(cudaGetLastError());
}

void updatePoses(const GpuPx1BlockVec& xp, GpuVec4d& qs, GpuVec3d& ts, int num_body,
	Scalar max_ext_trans, Scalar max_ext_rot)
{
	if (!xp.size())
		return;

	const int block = 256;
	const int grid = divUp(xp.size(), block);
	updatePosesKernelJoint<<<grid, block>>>(xp.size(), num_body, max_ext_trans, max_ext_rot, xp, qs, ts);
	CUDA_CHECK(cudaGetLastError());
}

void addExtPrior(GpuPxPBlockVec& Hpp, int numBody, int numExt, Scalar lambda_rot, Scalar lambda_trans)
{
	if (numExt <= 0)
		return;
	prepareCudaThreadContext();
	const int total = numExt * PDIM;
	const int block = 256;
	const int grid = divUp(total, block);
	addExtPriorKernel<<<grid, block>>>(numExt, numBody, lambda_rot, lambda_trans, Hpp);
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 2: convert the ext-range slots of a fixed-point int64 Hpp
// buffer into the double `Hpp` buffer. Slots `[numBody, numBody + numExt)` are
// targeted; each slot spans `PDIM * PDIM` scalars so the element range is
// `[numBody * PDIM * PDIM, (numBody + numExt) * PDIM * PDIM)`.
//
// Preconditions:
//   * `src_int` is sized identically to `Hpp.values()` and was zeroed before
//     the last accumulation kernel.
//   * Double slots in the ext range were not written during accumulation
//     (guaranteed because the kernel routes ext writes to the int64 buffer
//     when the pointer is non-null).
//
// The conversion overwrites `Hpp.values()` in the ext range, which is safe
// because the double buffer was left at zero there by the kernel.
void convertFixedPointHppExtRange(const long long* src_int,
	GpuPxPBlockVec& Hpp, int numBody, int numExt)
{
	if (numExt <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_AREA = PDIM * PDIM;
	const int n = numExt * BLOCK_AREA;
	const int offset = numBody * BLOCK_AREA;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int + offset, Hpp.values() + offset);
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3a: copy the ext-range slots of a fixed-point int64 buffer
// (`src_int`, sized identically to `bp.values()`, i.e. PDIM scalars per slot)
// into the corresponding double slots of `bp`. Only elements in
// `[numBody, numBody + numExt)` are touched; body-pose slots are left
// untouched. Mirrors `convertFixedPointHppExtRange` but for the PDIM-sized
// gradient vector.
void convertFixedPointBpExtRange(const long long* src_int,
	GpuPx1BlockVec& bp, int numBody, int numExt)
{
	if (numExt <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_SIZE = PDIM;
	const int n = numExt * BLOCK_SIZE;
	const int offset = numBody * BLOCK_SIZE;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int + offset, bp.values() + offset);
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3b: copy the body-range slots of a fixed-point int64 buffer
// (`src_int`, sized identically to `Hpp.values()`) into the corresponding
// double slots of `Hpp`. Only elements in `[0, numBody)` are touched;
// ext-range slots are left untouched (they are handled separately by
// `convertFixedPointHppExtRange`).
void convertFixedPointHppBodyRange(const long long* src_int,
	GpuPxPBlockVec& Hpp, int numBody)
{
	if (numBody <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_SIZE = PDIM * PDIM;
	const int n = numBody * BLOCK_SIZE;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int, Hpp.values());
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3b: copy the body-range slots of a fixed-point int64 buffer
// (`src_int`, sized identically to `bp.values()`) into the corresponding
// double slots of `bp`. Only elements in `[0, numBody)` are touched;
// ext-range slots are left untouched.
void convertFixedPointBpBodyRange(const long long* src_int,
	GpuPx1BlockVec& bp, int numBody)
{
	if (numBody <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_SIZE = PDIM;
	const int n = numBody * BLOCK_SIZE;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int, bp.values());
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3c: copy the full landmark-range slots of a fixed-point int64
// buffer into the double `Hll` buffer. Each slot spans `LDIM * LDIM` scalars,
// so the element range is `[0, numL * LDIM * LDIM)`. `src_int` must be sized
// identically to `Hll.values()` and zeroed prior to accumulation; the double
// slots must not have been written directly during accumulation (kernel
// branches to int64 when pointer is non-null).
void convertFixedPointHllRange(const long long* src_int,
	GpuLxLBlockVec& Hll, int numL)
{
	if (numL <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_AREA = LDIM * LDIM;
	const int n = numL * BLOCK_AREA;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int, Hll.values());
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3c: copy the full landmark-range slots of a fixed-point int64
// buffer into the double `bl` buffer. Each slot spans `LDIM` scalars, so the
// element range is `[0, numL * LDIM)`. Mirrors `convertFixedPointHllRange`.
void convertFixedPointBlRange(const long long* src_int,
	GpuLx1BlockVec& bl, int numL)
{
	if (numL <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_SIZE = LDIM;
	const int n = numL * BLOCK_SIZE;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int, bl.values());
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3d: copy the full non-zero slot range of a fixed-point int64
// buffer into the double `HscDirect` block matrix. HscDirect only receives
// writes through the atomic accumulation path (there is no ASSIGN path), so
// the entire `[0, nnz * PDIM * PDIM)` range is converted unconditionally.
// `src_int` must be sized identically to `HscDirect.values()` and zeroed
// prior to accumulation.
void convertFixedPointHscDirect(const long long* src_int,
	GpuHscBlockMat& HscDirect, int nnz)
{
	if (nnz <= 0 || src_int == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_AREA = PDIM * PDIM;
	const int n = nnz * BLOCK_AREA;
	const int block = 256;
	const int grid = divUp(n, block);
	deterministic::fixedPointToDoubleKernel<<<grid, block>>>(
		n, src_int, HscDirect.values());
	CUDA_CHECK(cudaGetLastError());
}

// Option 4 Phase 3e: per-slot conversion for the ext-range portion of `Hpl`.
// `Hpl` is a mixed-mode sparse block matrix: body edges write their unique
// slot via `ASSIGN` (no atomic), while ext dedup slots accumulate via atomic
// add from multiple edges. The deterministic path moves only the ext
// accumulations into an int64 mirror (`src_int`, sized identically to
// `Hpl.values()`, i.e. `nnz * PDIM * LDIM` scalars). After accumulation we
// must only copy back the ext dedup slots — copying the full nnz range would
// clobber the body ASSIGN values with the (zero) int64 content.
//
// `extSlotGlobalIds` is a device pointer to `nExtDedupSlots` int32 values,
// each giving the global nnz slot index (into `Hpl`) for the ext dedup slot
// at that position. This matches `d_edge2Hpl_.data() + nedges_total` in the
// `CudaBlockSolver` bookkeeping.
__global__ void fixedPointToDoubleHplExtSlotsKernel(
	int nExtSlots, int scalarsPerSlot,
	const int* extSlotGlobalIds,
	const long long* src_int, Scalar* dst_double)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = nExtSlots * scalarsPerSlot;
	if (tid >= total) return;
	const int extIdx = tid / scalarsPerSlot;
	const int scalarOffset = tid % scalarsPerSlot;
	const int globalSlot = extSlotGlobalIds[extIdx];
	const int idx = globalSlot * scalarsPerSlot + scalarOffset;
	dst_double[idx] = deterministic::fromFixedPoint(src_int[idx]);
}

void convertFixedPointHplExtSlots(const long long* src_int,
	GpuHplBlockMat& Hpl, const int* extSlotGlobalIds, int nExtDedupSlots)
{
	if (nExtDedupSlots <= 0 || src_int == nullptr || extSlotGlobalIds == nullptr)
		return;
	prepareCudaThreadContext();
	constexpr int BLOCK_AREA = PDIM * LDIM;
	const int n = nExtDedupSlots * BLOCK_AREA;
	const int block = 256;
	const int grid = divUp(n, block);
	fixedPointToDoubleHplExtSlotsKernel<<<grid, block>>>(
		nExtDedupSlots, BLOCK_AREA, extSlotGlobalIds, src_int, Hpl.values());
	CUDA_CHECK(cudaGetLastError());
}

void buildEdgeExtHpl(const int* h_dedup_slot_per_edge, int nedges_total, int nExtDedupSlots,
	const GpuVec1i& edge2Hpl, GpuVec1i& edge2HplExt)
{
	if (nedges_total <= 0)
		return;
	prepareCudaThreadContext();

	// Upload dedup_slot per edge to a device buffer local to this call.
	GpuVec1i d_dedup;
	d_dedup.assign(static_cast<size_t>(nedges_total), h_dedup_slot_per_edge);

	const int block = 256;
	const int grid = divUp(nedges_total, block);
	buildEdgeExtHplKernel<<<grid, block>>>(nedges_total, nedges_total, d_dedup.data(), edge2Hpl.data(), edge2HplExt.data());
	CUDA_CHECK(cudaGetLastError());
	// Wait for kernel completion before d_dedup is destroyed when this function returns,
	// otherwise the kernel may read freed device memory.
	CUDA_CHECK(cudaDeviceSynchronize());
	// Suppress unused-parameter warning for nExtDedupSlots; retained in the
	// public signature for potential sanity checks at a later revision.
	(void)nExtDedupSlots;
}

void fillEdge2HplExtSentinel(GpuVec1i& edge2HplExt, int nedges_total)
{
	if (nedges_total <= 0)
		return;
	prepareCudaThreadContext();
	const int block = 256;
	const int grid = divUp(nedges_total, block);
	fillIntSentinelKernel<<<grid, block>>>(nedges_total, -1, edge2HplExt.data());
	CUDA_CHECK(cudaGetLastError());
}

void syncExtSolutionToPerEdge(const GpuVec4d& qs, const GpuVec3d& ts,
	const GpuVec1i& edge2ExtIP, GpuVec4d& q_exts, GpuVec3d& t_exts, int nedges)
{
	if (nedges <= 0)
		return;
	prepareCudaThreadContext();
	const int block = 256;
	const int grid = divUp(nedges, block);
	syncExtSolutionToPerEdgeKernel<<<grid, block>>>(nedges, qs.data(), ts.data(),
		edge2ExtIP.data(), q_exts.data(), t_exts.data());
	CUDA_CHECK(cudaGetLastError());
}

void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& Xws)
{
	if (!xl.size())
		return;

	const int block = 1024;
	const int grid = divUp(xl.size(), block);
	updateLandmarksKernel<<<grid, block>>>(xl.size(), xl, Xws);
	CUDA_CHECK(cudaGetLastError());
}

Scalar computeScale(const GpuVec1d& x, const GpuVec1d& b, Scalar* scale, Scalar lambda, long long* scale_int)
{
	prepareCudaThreadContext();
	const int block = BLOCK_COMPUTE_SCALE;
	const int grid = 4;

	// Option 4 Phase 3g: mirror the pattern from computeActiveErrors_ — when
	// `scale_int` is provided, route the reduction through the deterministic
	// int64 accumulator and read the result back to host, converting via
	// `fromFixedPoint`. Otherwise preserve the legacy double path.
	if (scale_int != nullptr)
		CUDA_CHECK(cudaMemset(scale_int, 0, sizeof(long long)));
	CUDA_CHECK(cudaMemset(scale, 0, sizeof(Scalar)));
	computeScaleKernel<<<grid, block>>>(x, b, scale, lambda, x.ssize(), scale_int);
	CUDA_CHECK(cudaGetLastError());

	if (scale_int != nullptr)
	{
		long long h_scale_int = 0;
		CUDA_CHECK(cudaMemcpy(&h_scale_int, scale_int, sizeof(long long), cudaMemcpyDeviceToHost));
		return deterministic::fromFixedPoint(h_scale_int);
	}

	Scalar h_scale = 0;
	CUDA_CHECK(cudaMemcpy(&h_scale, scale, sizeof(Scalar), cudaMemcpyDeviceToHost));
	return h_scale;
}

void solveDiagonalSystem(const GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuLx1BlockVec& xl)
{
	const int size = Hll.size();
	const int block = 1024;
	const int grid = divUp(size, block);
	solveDiagonalSystemKernel<<<grid, block>>>(size, Hll, bl, xl);
	CUDA_CHECK(cudaGetLastError());
}

void solveDiagonalSystem(const GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuPx1BlockVec& xp)
{
	const int size = Hpp.size();
	const int block = 512;
	const int grid = divUp(size, block);
	solveDiagonalSystemKernel<<<grid, block>>>(size, Hpp, bp, xp);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace cuba
