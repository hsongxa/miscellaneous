#include "cuda_backend.h"

#include "cuda_kernels.cuh"

#include <thrust/inner_product.h>

#include <cassert>

#define BLOCK_SIZE 32

void cuda_backend::initialize()
{
	assert(s_cusparse_handle == nullptr);
	cusparseCreate(&s_cusparse_handle);
}

void cuda_backend::finalize()
{
	assert(s_cusparse_handle != nullptr);
	cusparseDestroy(s_cusparse_handle);
}

void cuda_backend::memcpy_from_frondend(const double* source, int n, double* target)
{
	cudaMemcpy(target, source, n * sizeof(double), cudaMemcpyHostToDevice);
}

void cuda_backend::memcpy_to_frondend(const double* source, int n, double* target)
{
	cudaMemcpy(target, source, n * sizeof(double), cudaMemcpyDeviceToHost);
}

scalar_vector* cuda_backend::create_vector(int n)
{
	using value_type = scalar_vector::value_type;
	value_type* values;
	cudaMalloc(&values, n * sizeof(value_type));

	scalar_vector* ret = new scalar_vector();
	ret->values = values;
	ret->n = n;
	return ret;
}

void cuda_backend::destroy_vector(scalar_vector* vector)
{
	assert(vector != nullptr);

	cudaFree(vector->values);
	delete vector;
}

csr_matrix* cuda_backend::create_matrix(int n, int nnz)
{
	using value_type = csr_matrix::value_type;
	value_type* values;
	cudaMalloc(&values, nnz * sizeof(value_type));
	int* row_pointers;
	cudaMalloc(&row_pointers, (n + 1) * sizeof(int));
	int* columns;
	cudaMalloc(&columns, nnz * sizeof(int));

	csr_matrix* ret = new csr_matrix();
	ret->values = values;
	ret->n = n;
	ret->row_pointers = row_pointers;
	ret->nnz = nnz;
	ret->columns = columns;

	return ret;
}

void cuda_backend::destroy_matrix(csr_matrix* matrix)
{
	assert(matrix != nullptr);

	cudaFree(matrix->values);
	cudaFree(matrix->row_pointers);
	cudaFree(matrix->columns);
	delete matrix;
}

void cuda_backend::copy_vector(const scalar_vector& x, scalar_vector& y)
{
	assert(x.n == y.n);

	const double* source = x.values;
	double* target = y.values;
	cudaMemcpy(target, source, x.n * sizeof(double), cudaMemcpyDeviceToDevice);
}

void cuda_backend::axpby(double a, const scalar_vector& x, double b, scalar_vector& y)
{
	const int NUM_BLOCKS = (x.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const double* source = x.values;
	double* target = y.values;
	kernel_axpby<<< NUM_BLOCKS, BLOCK_SIZE >>>(a, source, x.n, b, target);
}

void cuda_backend::dot(const scalar_vector& x, const scalar_vector& y, double& ret)
{
	const double* src1 = x.values;
	const double* src2 = y.values;
	ret = thrust::inner_product(src1, src1 + x.n, src2, 0.0);
}

void cuda_backend::spmv(double a, const csr_matrix& A, const scalar_vector& x, double b, scalar_vector& y)
{
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void*                dBuffer = NULL;
	size_t               bufferSize = 0;

	// Create sparse matrix A in CSR format
	cusparseCreateCsr(&matA, A.n, A.n, A.nnz, A.row_pointers, A.columns, A.values,
		              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	// Create dense vectors
	cusparseCreateDnVec(&vecX, x.n, x.values, CUDA_R_64F);
	cusparseCreateDnVec(&vecY, y.n, y.values, CUDA_R_64F);

	// allocate an external buffer if needed
	cusparseSpMV_bufferSize(s_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &a, matA, vecX, &b, vecY,
		                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
	cudaMalloc(&dBuffer, bufferSize);

	// execute SpMV
	cusparseSpMV(s_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &a, matA, vecX, &b, vecY,
		     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

	// destroy matrix/vector descriptors
	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
}
