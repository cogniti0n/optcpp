#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/solvers/gradient_descent.hpp"
#include "optlib/solvers/newton.hpp"
#include "optlib/solvers/momentum.hpp"
#include "optlib/solvers/nesterov.hpp"

using namespace optlib;

class Quadratic : public Objective
{
public:
    Quadratic(const Matrix &A_, const Vector &b_) : A(A_), b(b_) {}
    Scalar value(const Vector &x) const override
    {
        return 0.5 * x.transpose() * A * x + b.dot(x);
    }

    Vector gradient(const Vector &x) const override
    {
        return A * x + b;
    }

    Matrix hessian(const Vector &x) const override { return A; }

private:
    Matrix A;
    Vector b;
};

int main()
{
    int d = 5;
    Matrix A = Matrix::Identity(d, d);
    Vector b = Vector::Ones(d);

    Quadratic obj(A, b);

    Vector x0 = Vector::Zero(d);

    // GDOptions options;
    // NewtonOptions options;
    // MomentumOptions options;
    // options.alpha = 0.1; options.beta = 0.1;
    NesterovOptions options;
    options.alpha = 0.1;

    // SolverResult res = gradient_descent(obj, x0, options);
    // SolverResult res = newton(obj, x0, options);
    // SolverResult res = momentum(obj, x0, options);
    SolverResult res = nesterov(obj, x0, options);

    std::cout << "Solution:\n"
              << res.x << std::endl;
    std::cout << "Gradient norm: " << res.grad_norm << std::endl;
    std::cout << "Iteration number: " << res.iters << std::endl;

    return 0;
}