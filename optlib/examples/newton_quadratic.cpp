#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/solvers/newton.hpp"

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

    NewtonParams ntparams;
    ntparams.step_size = 0.1;

    Vector x_star = newton(obj, x0, ntparams, true);

    std::cout << "Solution:\n"
              << x_star << std::endl;
    std::cout << "Gradient norm: " << obj.gradient(x_star).norm() << std::endl;

    return 0;
}