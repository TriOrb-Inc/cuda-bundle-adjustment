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

#include "cuda_linear_solver.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "device_buffer.h"
#include "cuda_block_solver.h"

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

enum class OrderingMethod
{
	kNone,
	kMetis,
	kSymRcm,
	kSymAmd,
	kSymMdq,
};

OrderingMethod resolve_ordering_method()
{
	const char* env_ordering = std::getenv("TRIORB_CUDA_BA_ORDERING");
	if (env_ordering != nullptr)
	{
		std::string value(env_ordering);
		for (char& ch : value)
			ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

		if (value == "none" || value == "off")
			return OrderingMethod::kNone;
		if (value == "symrcm" || value == "rcm")
			return OrderingMethod::kSymRcm;
		if (value == "symamd" || value == "amd")
			return OrderingMethod::kSymAmd;
		if (value == "symmdq" || value == "mdq")
			return OrderingMethod::kSymMdq;
		return OrderingMethod::kMetis;
	}

	const char* env_is_jetson = std::getenv("TRIORB_IS_JETSON");
	if (env_is_jetson != nullptr && std::string(env_is_jetson) == "1")
	{
		return OrderingMethod::kNone;
	}

	// Option 4 Phase 5: METIS nested-dissection uses a randomized initial
	// partition (default seed is time-based), so the permutation `P` differs
	// across runs and Cholesky factor `L` floats a few ULPs in spite of the
	// deterministic int64 atomic accumulation. When the caller asks for
	// deterministic BA, fall back to `symrcm`, which is a deterministic
	// graph ordering (reverse Cuthill–McKee). Callers can still opt back
	// into METIS by setting `TRIORB_CUDA_BA_ORDERING=metis` explicitly.
	const char* env_det_accum = std::getenv("TRIORB_CUDA_BA_DETERMINISTIC_ACCUM");
	if (env_det_accum != nullptr && std::string(env_det_accum) == "1")
	{
		return OrderingMethod::kSymRcm;
	}

	return OrderingMethod::kMetis;
}

const char* ordering_method_to_cstr(const OrderingMethod method)
{
	switch (method)
	{
		case OrderingMethod::kNone:
			return "none";
		case OrderingMethod::kMetis:
			return "metis";
		case OrderingMethod::kSymRcm:
			return "symrcm";
		case OrderingMethod::kSymAmd:
			return "symamd";
		case OrderingMethod::kSymMdq:
			return "symmdq";
		default:
			return "unknown";
	}
}

enum class LinearSolverBackend
{
	kSparse,
	kDense,
};

LinearSolverBackend resolve_linear_solver_backend()
{
	const char* env_backend = std::getenv("TRIORB_CUDA_BA_LINEAR_SOLVER");
	if (env_backend != nullptr)
	{
		std::string value(env_backend);
		for (char& ch : value)
			ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

		if (value == "dense")
			return LinearSolverBackend::kDense;
		if (value == "sparse")
			return LinearSolverBackend::kSparse;
	}

	const char* env_is_jetson = std::getenv("TRIORB_IS_JETSON");
	const char* env_jetson_board = std::getenv("TRIORB_JETSON_BOARD");
	const char* env_jetson_model = std::getenv("TRIORB_JETSON_MODEL");
	if (env_is_jetson != nullptr && std::string(env_is_jetson) == "1" &&
		env_jetson_board != nullptr && env_jetson_model != nullptr)
	{
		std::string board(env_jetson_board);
		std::string model(env_jetson_model);
		for (char& ch : board)
			ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
		for (char& ch : model)
			ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

		if (board == "t186ref" && model.find("orin") != std::string::npos)
			return LinearSolverBackend::kDense;
	}

	return LinearSolverBackend::kSparse;
}

const char* linear_solver_backend_to_cstr(const LinearSolverBackend backend)
{
	switch (backend)
	{
		case LinearSolverBackend::kSparse:
			return "sparse";
		case LinearSolverBackend::kDense:
			return "dense";
		default:
			return "unknown";
	}
}

bool check_cusolver_status(const cusolverStatus_t status, const char* const context)
{
	if (status == CUSOLVER_STATUS_SUCCESS)
		return true;

	std::cerr
		<< "CUDA BA cusolver error: context=" << context
		<< " status=" << static_cast<int>(status)
		<< std::endl;
	return false;
}

int cusolver_potrf_buffer_size(
	cusolverDnHandle_t handle,
	const cublasFillMode_t uplo,
	const int n,
	float* const A,
	const int lda)
{
	int lwork = 0;
	check_cusolver_status(
		cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, &lwork),
		"spotrf_buffer_size");
	return lwork;
}

int cusolver_potrf_buffer_size(
	cusolverDnHandle_t handle,
	const cublasFillMode_t uplo,
	const int n,
	double* const A,
	const int lda)
{
	int lwork = 0;
	check_cusolver_status(
		cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, &lwork),
		"dpotrf_buffer_size");
	return lwork;
}

cusolverStatus_t cusolver_potrf(
	cusolverDnHandle_t handle,
	const cublasFillMode_t uplo,
	const int n,
	float* const A,
	const int lda,
	float* const workspace,
	const int lwork,
	int* const dev_info)
{
	return cusolverDnSpotrf(handle, uplo, n, A, lda, workspace, lwork, dev_info);
}

cusolverStatus_t cusolver_potrf(
	cusolverDnHandle_t handle,
	const cublasFillMode_t uplo,
	const int n,
	double* const A,
	const int lda,
	double* const workspace,
	const int lwork,
	int* const dev_info)
{
	return cusolverDnDpotrf(handle, uplo, n, A, lda, workspace, lwork, dev_info);
}

cusolverStatus_t cusolver_potrs(
	cusolverDnHandle_t handle,
	const cublasFillMode_t uplo,
	const int n,
	const int nrhs,
	const float* const A,
	const int lda,
	float* const B,
	const int ldb,
	int* const dev_info)
{
	return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, dev_info);
}

cusolverStatus_t cusolver_potrs(
	cusolverDnHandle_t handle,
	const cublasFillMode_t uplo,
	const int n,
	const int nrhs,
	const double* const A,
	const int lda,
	double* const B,
	const int ldb,
	int* const dev_info)
{
	return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, dev_info);
}

} // namespace

template <typename T>
static constexpr bool is_value_type_32f() { return std::is_same_v<T, float>; }
template <typename T>
static constexpr bool is_value_type_64f() { return std::is_same_v<T, double>; }

struct CusparseHandle
{
	CusparseHandle() { init(); }
	~CusparseHandle() { destroy(); }
	void init()
	{
		trace_cuda_ba("cusparseCreate begin");
		cusparseCreate(&handle);
		trace_cuda_ba("cusparseCreate end");
	}
	void destroy() { cusparseDestroy(handle); }
	operator cusparseHandle_t() const { return handle; }
	CusparseHandle(const CusparseHandle&) = delete;
	CusparseHandle& operator=(const CusparseHandle&) = delete;
	cusparseHandle_t handle;
};

struct CusolverHandle
{
	CusolverHandle() { init(); }
	~CusolverHandle() { destroy(); }
	void init()
	{
		trace_cuda_ba("cusolverSpCreate begin");
		cusolverSpCreate(&handle);
		trace_cuda_ba("cusolverSpCreate end");
	}
	void destroy() { cusolverSpDestroy(handle); }
	operator cusolverSpHandle_t() const { return handle; }
	CusolverHandle(const CusolverHandle&) = delete;
	CusolverHandle& operator=(const CusolverHandle&) = delete;
	cusolverSpHandle_t handle;
};

struct CusolverDnHandle
{
	CusolverDnHandle() { init(); }
	~CusolverDnHandle() { destroy(); }
	void init()
	{
		trace_cuda_ba("cusolverDnCreate begin");
		cusolverDnCreate(&handle);
		trace_cuda_ba("cusolverDnCreate end");
	}
	void destroy() { cusolverDnDestroy(handle); }
	operator cusolverDnHandle_t() const { return handle; }
	CusolverDnHandle(const CusolverDnHandle&) = delete;
	CusolverDnHandle& operator=(const CusolverDnHandle&) = delete;
	cusolverDnHandle_t handle;
};

struct CusparseMatDescriptor
{
	CusparseMatDescriptor() { init(); }
	~CusparseMatDescriptor() { destroy(); }

	void init()
	{
		cusparseCreateMatDescr(&desc);
		cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
	}

	void destroy() { cusparseDestroyMatDescr(desc); }
	operator cusparseMatDescr_t() const { return desc; }
	CusparseMatDescriptor(const CusparseMatDescriptor&) = delete;
	CusparseMatDescriptor& operator=(const CusparseMatDescriptor&) = delete;
	cusparseMatDescr_t desc;
};

template <typename T>
class SparseSquareMatrixCSR
{
public:

	SparseSquareMatrixCSR() : size_(0), nnz_(0) {}

	void resize(int size)
	{
		size_ = size;
		rowPtr_.resize(size + 1);
	}

	void resizeNonZeros(int nnz)
	{
		nnz_ = nnz;
		values_.resize(nnz);
		colInd_.resize(nnz);
	}

	void upload(const T* values = nullptr, const int* rowPtr = nullptr, const int* colInd = nullptr)
	{
		if (values)
			values_.upload(values);
		if (rowPtr)
			rowPtr_.upload(rowPtr);
		if (colInd)
			colInd_.upload(colInd);
	}

	void download(T* values = nullptr, int* rowPtr = nullptr, int* colInd = nullptr) const
	{
		if (values)
			values_.download(values);
		if (rowPtr)
			rowPtr_.download(rowPtr);
		if (colInd)
			colInd_.download(colInd);
	}

	T* val() { return values_.data(); }
	int* rowPtr() { return rowPtr_.data(); }
	int* colInd() { return colInd_.data(); }

	const T* val() const { return values_.data(); }
	const int* rowPtr() const { return rowPtr_.data(); }
	const int* colInd() const { return colInd_.data(); }

	int size() const { return size_; }
	int nnz() const { return nnz_; }

	cusparseMatDescr_t desc() const { return desc_; }

private:

	DeviceBuffer<T> values_;
	DeviceBuffer<int> rowPtr_;
	DeviceBuffer<int> colInd_;
	int size_, nnz_;
	CusparseMatDescriptor desc_;
};

template <typename T>
class SparseCholesky
{
public:

	void init(cusolverSpHandle_t handle)
	{
		handle_ = handle;

		// create info
		cusolverSpCreateCsrcholInfo(&info_);
	}

	void allocateBuffer(const SparseSquareMatrixCSR<T>& A)
	{
		size_t internalData, workSpace;

		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholBufferInfo(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, &internalData, &workSpace);

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholBufferInfo(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, &internalData, &workSpace);

		buffer_.resize(workSpace);
	}

	bool hasZeroPivot(int* position = nullptr) const
	{
		const T tol = static_cast<T>(1e-14);
		int singularity = -1;

		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholZeroPivot(handle_, info_, tol, &singularity);

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholZeroPivot(handle_, info_, tol, &singularity);

		if (position)
			*position = singularity;
		return singularity >= 0;
	}

	bool analyze(const SparseSquareMatrixCSR<T>& A)
	{
		cusolverSpXcsrcholAnalysis(handle_, A.size(), A.nnz(), A.desc(), A.rowPtr(), A.colInd(), info_);
		allocateBuffer(A);
		return true;
	}

	bool factorize(SparseSquareMatrixCSR<T>& A)
	{
		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholFactor(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, buffer_.data());

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholFactor(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, buffer_.data());

		return !hasZeroPivot();
	}

	void solve(int size, const T* b, T* x)
	{
		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholSolve(handle_, size, b, x, info_, buffer_.data());

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholSolve(handle_, size, b, x, info_, buffer_.data());
	}

	void destroy()
	{
		cusolverSpDestroyCsrcholInfo(info_);
	}

	~SparseCholesky() { destroy(); }

private:

	cusolverSpHandle_t handle_;
	csrcholInfo_t info_;
	DeviceBuffer<unsigned char> buffer_;
};

template <typename T>
class CuSparseCholeskySolver
{
public:

	enum Info
	{
		SUCCESS,
		NUMERICAL_ISSUE
	};

	CuSparseCholeskySolver(int size = 0)
	{
		init();

		if (size > 0)
			resize(size);
	}

	void init()
	{
		trace_cuda_ba("CuSparseCholeskySolver::init begin");
		cholesky.init(cusolver);
		doOrdering = false;
		information = Info::SUCCESS;
		trace_cuda_ba("CuSparseCholeskySolver::init end");
	}

	void resize(int size)
	{
		Acsr.resize(size);
		d_y.resize(size);
		d_z.resize(size);
	}

	void setPermutaion(int size, const int* P)
	{
		h_PT.resize(size);
		for (int i = 0; i < size; i++)
			h_PT[P[i]] = i;

		d_P.assign(size, P);
		d_PT.assign(size, h_PT.data());
		doOrdering = true;
	}

	void analyze(int nnz, const int* csrRowPtr, const int* csrColInd)
	{
		const int size = Acsr.size();
		trace_cuda_ba("linear analyze begin: size=" + std::to_string(size) +
			", nnz=" + std::to_string(nnz) +
			", ordered=" + std::string(doOrdering ? "true" : "false"));
		Acsr.resizeNonZeros(nnz);

		if (doOrdering)
		{
			d_tmpRowPtr.assign(size + 1, csrRowPtr);
			d_tmpColInd.assign(nnz, csrColInd);
			d_nnzPerRow.resize(size + 1);
			d_map.resize(nnz);

			gpu::twistCSR(size, nnz, d_tmpRowPtr, d_tmpColInd, d_PT,
				Acsr.rowPtr(), Acsr.colInd(), d_map, d_nnzPerRow);
		}
		else
		{
			Acsr.upload(nullptr, csrRowPtr, csrColInd);
		}

		cholesky.analyze(Acsr);
		trace_cuda_ba("linear analyze end");
	}

	void factorize(const T* d_A)
	{
		trace_cuda_ba("linear factorize begin");
		if (doOrdering)
		{
			permute(Acsr.nnz(), d_A, Acsr.val(), d_map);
		}
		else
		{
			cudaMemcpy(Acsr.val(), d_A, sizeof(Scalar) * Acsr.nnz(), cudaMemcpyDeviceToDevice);
		}

		// M = L * LT
		if (!cholesky.factorize(Acsr))
			information = Info::NUMERICAL_ISSUE;
		trace_cuda_ba("linear factorize end");
	}

	void solve(const T* d_b, T* d_x)
	{
		trace_cuda_ba("linear solve begin");
		if (doOrdering)
		{
			// y = P * b
			permute(Acsr.size(), d_b, d_y, d_P);

			// solve A * z = y
			cholesky.solve(Acsr.size(), d_y, d_z);

			// x = PT * z
			permute(Acsr.size(), d_z, d_x, d_PT);
		}
		else
		{
			// solve A * x = b
			cholesky.solve(Acsr.size(), d_b, d_x);
		}
		trace_cuda_ba("linear solve end");
	}

	void permute(int size, const T* src, T* dst, const int* P)
	{
		gpu::permute(size, src, dst, P);
	}

	bool reordering(
		int size,
		int nnz,
		const int* csrRowPtr,
		const int* csrColInd,
		int* P,
		OrderingMethod method) const
	{
		if (method == OrderingMethod::kNone)
			return false;

		cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
		switch (method)
		{
			case OrderingMethod::kSymRcm:
				status = cusolverSpXcsrsymrcmHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
				break;
			case OrderingMethod::kSymAmd:
				status = cusolverSpXcsrsymamdHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
				break;
			case OrderingMethod::kSymMdq:
				status = cusolverSpXcsrsymmdqHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
				break;
			case OrderingMethod::kMetis:
				status = cusolverSpXcsrmetisndHost(
					cusolver,
					size,
					nnz,
					Acsr.desc(),
					csrRowPtr,
					csrColInd,
					nullptr,
					P);
				break;
			case OrderingMethod::kNone:
			default:
				return false;
		}

		if (status != CUSOLVER_STATUS_SUCCESS)
		{
			std::cerr
				<< "CUDA BA reordering failed: method=" << ordering_method_to_cstr(method)
				<< " status=" << static_cast<int>(status)
				<< std::endl;
			return false;
		}

		return true;
	}

	Info info() const
	{
		return information;
	}

	void downloadCSR(int* csrRowPtr, int* csrColInd)
	{
		Acsr.download(nullptr, csrRowPtr, csrColInd);
	}

private:

	SparseSquareMatrixCSR<T> Acsr;
	DeviceBuffer<T> d_y, d_z, d_tmp;
	DeviceBuffer<int> d_P, d_PT, d_map;
	DeviceBuffer<int> d_tmpRowPtr, d_tmpColInd, d_nnzPerRow;

	CusparseHandle cusparse;
	CusolverHandle cusolver;

	SparseCholesky<T> cholesky;

	std::vector<int> h_PT;

	Info information;
	bool doOrdering;
};

class SparseLinearSolverImpl : public SparseLinearSolver
{
public:

	using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
	using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
	using Cholesky = CuSparseCholeskySolver<Scalar>;

	void initialize(const HschurSparseBlockMatrix& Hsc) override
	{
		const int size = Hsc.rows();
		const int nnz = Hsc.nnzSymm();

		cholesky_.resize(size);

		// Jetson では metis host ordering が固まる run があるため、既定では無効化し、
		// 必要なら TRIORB_CUDA_BA_ORDERING で明示 override する。
		const OrderingMethod ordering_method = resolve_ordering_method();
		trace_cuda_ba("linear initialize begin: size=" + std::to_string(size) +
			", nnz=" + std::to_string(nnz) +
			", ordering=" + ordering_method_to_cstr(ordering_method));
		if (ordering_method != OrderingMethod::kNone)
		{
			P_.resize(size);
			if (cholesky_.reordering(
				size,
				nnz,
				Hsc.rowPtr(),
				Hsc.colInd(),
				P_.data(),
				ordering_method))
			{
				cholesky_.setPermutaion(size, P_.data());
			}
			else
			{
				std::cerr
					<< "CUDA BA falls back to unordered factorization because reordering did not complete"
					<< std::endl;
				P_.clear();
			}
		}

		// analyze
		cholesky_.analyze(nnz, Hsc.rowPtr(), Hsc.colInd());
		trace_cuda_ba("linear initialize end");
	}

	bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) override
	{
		trace_cuda_ba("sparse linear solver solve begin");
		cholesky_.factorize(d_A);
		trace_cuda_ba("sparse linear solver factorize returned");

		if (cholesky_.info() != Cholesky::SUCCESS)
		{
			std::cerr << "factorize failed" << std::endl;
			return false;
		}

		cholesky_.solve(d_b, d_x);
		trace_cuda_ba("sparse linear solver solve end");

		return true;
	}

private:

	std::vector<int> P_;
	Cholesky cholesky_;
};

class DenseLinearSolverImpl : public SparseLinearSolver
{
public:

	void initialize(const HschurSparseBlockMatrix& Hsc) override
	{
		this->size_ = Hsc.rows();
		this->lda_ = Hsc.rows();
		this->nnz_ = Hsc.nnzSymm();

		trace_cuda_ba(
			"dense linear initialize begin: size=" + std::to_string(this->size_) +
			", nnz=" + std::to_string(this->nnz_));

		this->h_row_ptr_.assign(Hsc.rowPtr(), Hsc.rowPtr() + this->size_ + 1);
		this->h_col_ind_.assign(Hsc.colInd(), Hsc.colInd() + this->nnz_);
		this->h_csr_values_.resize(this->nnz_);
		this->h_dense_matrix_.resize(static_cast<size_t>(this->size_) * static_cast<size_t>(this->size_));

		if (this->size_ <= 0)
		{
			trace_cuda_ba("dense linear initialize end: empty system");
			return;
		}

		this->d_dense_matrix_.resize(static_cast<size_t>(this->size_) * static_cast<size_t>(this->size_));
		this->d_rhs_.resize(this->size_);
		this->d_dev_info_.resize(1);

		this->workspace_size_ = cusolver_potrf_buffer_size(
			this->cusolver_dn_,
			CUBLAS_FILL_MODE_LOWER,
			this->size_,
			this->d_dense_matrix_.data(),
			this->lda_);
		if (this->workspace_size_ > 0)
			this->d_workspace_.resize(this->workspace_size_);

		trace_cuda_ba(
			"dense linear initialize end: workspace=" + std::to_string(this->workspace_size_));
	}

	bool solve(const Scalar* const d_A, const Scalar* const d_b, Scalar* const d_x) override
	{
		trace_cuda_ba("dense linear solver solve begin");
		if (this->size_ <= 0)
			return true;

		if (this->workspace_size_ <= 0)
		{
			std::cerr << "CUDA BA dense solver has no workspace" << std::endl;
			return false;
		}

		CUDA_CHECK(cudaMemcpy(
			this->h_csr_values_.data(),
			d_A,
			sizeof(Scalar) * static_cast<size_t>(this->nnz_),
			cudaMemcpyDeviceToHost));

		std::fill(this->h_dense_matrix_.begin(), this->h_dense_matrix_.end(), Scalar(0));
		for (int row = 0; row < this->size_; ++row)
		{
			for (int index = this->h_row_ptr_[row]; index < this->h_row_ptr_[row + 1]; ++index)
			{
				const int col = this->h_col_ind_[index];
				this->h_dense_matrix_[static_cast<size_t>(col) * static_cast<size_t>(this->size_) + static_cast<size_t>(row)] =
					this->h_csr_values_[index];
			}
		}

		this->d_dense_matrix_.upload(this->h_dense_matrix_.data());
		CUDA_CHECK(cudaMemcpy(
			this->d_rhs_.data(),
			d_b,
			sizeof(Scalar) * static_cast<size_t>(this->size_),
			cudaMemcpyDeviceToDevice));

		const cusolverStatus_t status = cusolver_potrf(
			this->cusolver_dn_,
			CUBLAS_FILL_MODE_LOWER,
			this->size_,
			this->d_dense_matrix_.data(),
			this->lda_,
			this->d_workspace_.data(),
			this->workspace_size_,
			this->d_dev_info_.data());

		if (!check_cusolver_status(status, "potrf"))
			return false;

		int dev_info = 0;
		this->d_dev_info_.download(&dev_info);
		if (dev_info != 0)
		{
			std::cerr << "CUDA BA dense potrf failed: dev_info=" << dev_info << std::endl;
			return false;
		}

		const cusolverStatus_t status_potrs = cusolver_potrs(
			this->cusolver_dn_,
			CUBLAS_FILL_MODE_LOWER,
			this->size_,
			1,
			this->d_dense_matrix_.data(),
			this->lda_,
			this->d_rhs_.data(),
			this->size_,
			this->d_dev_info_.data());

		if (!check_cusolver_status(status_potrs, "potrs"))
			return false;

		this->d_dev_info_.download(&dev_info);
		if (dev_info != 0)
		{
			std::cerr << "CUDA BA dense potrs failed: dev_info=" << dev_info << std::endl;
			return false;
		}

		CUDA_CHECK(cudaMemcpy(
			d_x,
			this->d_rhs_.data(),
			sizeof(Scalar) * static_cast<size_t>(this->size_),
			cudaMemcpyDeviceToDevice));
		trace_cuda_ba("dense linear solver solve end");
		return true;
	}

private:

	int size_ = 0;
	int lda_ = 0;
	int nnz_ = 0;
	int workspace_size_ = 0;
	std::vector<int> h_row_ptr_;
	std::vector<int> h_col_ind_;
	std::vector<Scalar> h_csr_values_;
	std::vector<Scalar> h_dense_matrix_;
	DeviceBuffer<Scalar> d_dense_matrix_;
	DeviceBuffer<Scalar> d_rhs_;
	DeviceBuffer<Scalar> d_workspace_;
	DeviceBuffer<int> d_dev_info_;
	CusolverDnHandle cusolver_dn_;
};

SparseLinearSolver::Ptr SparseLinearSolver::create()
{
	trace_cuda_ba("SparseLinearSolver::create begin");
	const LinearSolverBackend backend = resolve_linear_solver_backend();
	trace_cuda_ba(
		"SparseLinearSolver::create backend=" + std::string(linear_solver_backend_to_cstr(backend)));

	if (backend == LinearSolverBackend::kDense)
	{
		auto solver = std::make_unique<DenseLinearSolverImpl>();
		trace_cuda_ba("SparseLinearSolver::create end");
		return solver;
	}

	auto solver = std::make_unique<SparseLinearSolverImpl>();
	trace_cuda_ba("SparseLinearSolver::create end");
	return solver;
}

SparseLinearSolver::~SparseLinearSolver()
{
}

} // namespace cuba
