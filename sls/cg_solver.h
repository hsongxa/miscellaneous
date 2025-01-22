#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include <cmath>

template<typename BK>
class cg_solver {
public:
	using vector_type = BK::vector_type;
	using matrix_type = BK::matrix_type;

	cg_solver() : Ap(nullptr), r(nullptr), p(nullptr) {}

	void set_up(const matrix_type& A) {
		int new_n = A.n;

		// prepare the workspapce
		if (Ap == nullptr) Ap = BK::create_vector(new_n);
		else if (Ap->n != new_n) {
			BK::destroy_vector(Ap);
			Ap = BK::create_vector(new_n);
		}
		if (r == nullptr) r = BK::create_vector(new_n);
		else if (r->n != new_n) {
			BK::destroy_vector(r);
			r = BK::create_vector(new_n);
		}
		if (p == nullptr) p = BK::create_vector(new_n);
		else if (p->n != new_n) {
			BK::destroy_vector(p);
			p = BK::create_vector(new_n);
		}
	}

	void solve(const matrix_type& A, const vector_type& b, vector_type& x) {
		BK::copy_vector(b, *r);

		// r = b - Ax
		BK::spmv(A, x, *Ap);
		BK::axpy(-1.0, *Ap, *r);

		// p = r
		BK::copy_vector(*r, *p);

		// save (r_old * r_old) and compute r_norm
		double dot_old;
		BK::dot(*r, *r, &dot_old);
		double r_norm = std::sqrt(dot_old);

		int k = 0;
		while (r_norm > tolerance() && k < max_iters())
		{
			double dot_new; // first used for (p * Ap), then (r * r)

			// alpha = (r_old * r_old)/(p * Ap)
			BK::spmv(A, *p, *Ap);
			BK::dot(*p, *Ap, &dot_new);
			double alpha = dot_old / dot_new;

			// x = x + alpha * p
			// r = r - alpha * Ap
			BK::axpy(alpha, *p, x);
			BK::axpy(-alpha, *Ap, *r);

			// beta = (r * r)/(r_old * r_old)
			BK::dot(*r, *r, &dot_new);
			double beta = dot_new / dot_old;

			// p = r + beta * p
			BK::axpby(1.0, *r, beta, *p);

			// update
			dot_old = dot_new;
			r_norm = std::sqrt(dot_old);
			++k;
		}
	}

	~cg_solver() {
		if (Ap != nullptr) BK::destroy_vector(Ap);
		if (r != nullptr) BK::destroy_vector(r);
		if (p != nullptr) BK::destroy_vector(p);
	}
	 
private:
	vector_type* Ap;
	vector_type* r;
	vector_type* p;

	// TODO: introduce solver configuration
	int max_iters() const { return 10000; }
	double tolerance() const { return 1.0e-4; }
};

#endif

