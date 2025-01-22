#include "cuda_backend.h"
#include "cg_solver.h"

#include <vector>
#include <iostream>

int main()
{
    using BK = cuda_backend;
    using vector_type = BK::vector_type;
    using matrix_type = BK::matrix_type;

    BK::initialize();

    // TODO: test BK spmv, dot, ..., etc.
    const int N = 10;
    std::vector<double> host_vec(N);// { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    vector_type* b = BK::create_vector(N);
    BK::memcpy_from_frondend(host_vec.data(), host_vec.size(), b->values);

    cg_solver<BK> solver;


    BK::destroy_vector(b);

    BK::finalize();
    return 0;
}

