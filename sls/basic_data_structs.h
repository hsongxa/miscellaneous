#ifndef BASIC_DATA_STRUCTS_H
#define BASIC_DATA_STRUCTS_H

struct scalar_vector {
	using value_type = double;

	value_type* values;
	int         n;
};

struct csr_matrix {
	using value_type = double;

	value_type* values;
	int*        row_pointers;
	int*        columns;

	int         n;
	int         nnz;
};

// block vectors, matrices, ..., etc.

#endif

