#include "cuda_backend.h"
#include "cg_solver.h"

#include <vector>
#include <iostream>
#include <cmath>

int main()
{
    using BK = cuda_backend;
    using vector_type = BK::vector_type;
    using matrix_type = BK::matrix_type;

    BK::initialize();

    // Test SpMV
    const int N = 4;
    int       h_A_row_pointers[] = { 0, 3, 4, 7, 9 };
    int       h_A_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    double    h_A_values[]  = { 1.0, 2.0, 3.0, 4.0, 5.0,
                                6.0, 7.0, 8.0, 9.0 };
    double    h_X[] = { 1.0, 2.0, 3.0, 4.0 };
    double    h_Y[] = { 0.0, 0.0, 0.0, 0.0 };
    double    h_Y_result[] = { 19.0, 8.0, 51.0, 52.0 };

    int NNZ = h_A_row_pointers[N];
    matrix_type* A = BK::create_matrix(N, NNZ);
    BK::memcpy_from_frondend(h_A_row_pointers, N + 1, A->row_pointers);
    BK::memcpy_from_frondend(h_A_columns, NNZ, A->columns);
    BK::memcpy_from_frondend(h_A_values, NNZ, A->values);

    vector_type* x = BK::create_vector(N);
    BK::memcpy_from_frondend(h_X, N, x->values);
    vector_type* y = BK::create_vector(N);
    //BK::copy_vector(*x, *y);
    BK::memcpy_from_frondend(h_Y, N, y->values);

    BK::spmv(*A, *x, *y);
    BK::memcpy_to_frondend<double>(y->values, N, h_Y);

    bool pass = true;
    for (int i = 0; i < N; ++i)
        if (std::fabs(h_Y[i] - h_Y_result[i]) > 1.0e-10)
        {
            pass = false;
            break;
        }
    std::cout << "SpMV test: pass = " << pass << std::endl;

    // Test dot
    double dot;
    BK::dot(*x, *y, dot);
    std::cout << "dot test: dot = " << dot << std::endl;

    // Test axpby
    BK::axpby(2.0, *x, 1.0, *y);
    BK::memcpy_to_frondend<double>(y->values, N, h_Y_result);

    std::cout << "axpby: " << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << "   " << h_Y_result[i] << std::endl;

    BK::destroy_matrix(A);
    BK::destroy_vector(x);
    BK::destroy_vector(y);

    // Test the cg solver:
    // A = {  1 -1  2  0 }  b = { 1.0 }
    //     { -1  4 -1  1 }      { 2.0 }
    //     {  2 -1  6 -2 }      { 3.0 }
    //     {  0  1 -2  4 }      { 4.0 }
    int       A_row_pointers[] = { 0, 3, 7, 11, 14 };
    int       A_columns[] = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3 };
    double    A_values[] = { 1.0, -1.0, 2.0, -1.0, 4.0, -1.0, 1.0,
                             2.0, -1.0, 6.0, -2.0, 1.0, -2.0, 4.0 };
    double    B[] = { 1.0, 2.0, 3.0, 4.0 };
    double    X[] = { 0.0, 0.0, 0.0, 0.0 };
    double    X_result[] = { -25.0, -5.0, 10.5, 7.5 };

    NNZ = A_row_pointers[N];
    A = BK::create_matrix(N, NNZ);
    BK::memcpy_from_frondend(A_row_pointers, N + 1, A->row_pointers);
    BK::memcpy_from_frondend(A_columns, NNZ, A->columns);
    BK::memcpy_from_frondend(A_values, NNZ, A->values);

    x = BK::create_vector(N);
    BK::memcpy_from_frondend(X, N, x->values);
    vector_type* b = BK::create_vector(N);
    BK::memcpy_from_frondend(B, N, b->values);

    cg_solver<BK> solver;
    bool solved;
    int num_iters;
    solver.set_up(*A);
    std::tie(solved, num_iters) = solver.solve(*A, *b, *x);
    BK::memcpy_to_frondend<double>(x->values, N, X);

    std::cout << "cg soler: solved = " << solved << std::endl;
    std::cout << "cg soler: # iters = " << num_iters << std::endl;
    pass = (solved && num_iters <= solver.max_iters()) ? true : false;
    if (pass)
        for (int i = 0; i < N; ++i)
            if (std::fabs(X[i] - X_result[i]) > solver.tolerance())
            {
                pass = false;
                break;
            }
    std::cout << "cg soler: pass = " << pass << std::endl;

    BK::destroy_matrix(A);
    BK::destroy_vector(x);
    BK::destroy_vector(b);

    BK::finalize();
    return 0;
}

