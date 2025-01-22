#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "basic_data_structs.h"

#include "cusparse_v2.h"

#include <cstddef>

// This class is an instance of the "backend" concept.
// Another instance could be omp_backend, for example.
//
// TODO: Generalize member functions to function templates,
// TODO: with parameters of value_type (double), matrix_type
// TODO: (csr_matrix), vector_type (scalar_vector), ..., etc.
//
// These functions work with matrices and vectors on CUDA device.
//
class cuda_backend
{
public:
	using vector_type = scalar_vector;
	using matrix_type = csr_matrix;

public:
	cuda_backend() = delete;

	static void initialize();

	static void finalize();

	static void memcpy_from_frondend(const double* source, int n, double* target);

	static void memcpy_to_frondend(const double* source, int n, double* target);

	static scalar_vector* create_vector(int n);

	static void destroy_vector(scalar_vector* vector);

	static csr_matrix* create_matrix(int n, int nnz);

	static void destroy_matrix(csr_matrix* matrix);

	static void copy_vector(const scalar_vector& x, scalar_vector& y);

	// y = a * x + y
	static void axpy(double a, const scalar_vector& x, scalar_vector& y)
	{ axpby(a, x, 1.0, y); }

	// y = a * x + b * y
	static void axpby(double a, const scalar_vector& x, double b, scalar_vector& y);

	// ret = x * y
	static void dot(const scalar_vector& x, const scalar_vector& y, double& ret);

	// y = A * x
	static void spmv(const csr_matrix& A, const scalar_vector& x, scalar_vector& y)
	{ spmv(1.0, A, x, 1.0, y); }

	// y = a * A * x + b * y
	static void spmv(double a, const csr_matrix& A, const scalar_vector& x, double b, scalar_vector& y);

private:
	static inline cusparseHandle_t s_cusparse_handle = nullptr;
};

#endif

