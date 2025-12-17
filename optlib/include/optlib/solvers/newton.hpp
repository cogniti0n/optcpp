#pragma once

#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/stepsizes/backtracking.hpp"

using namespace optlib;

struct NewtonParams
{
    int max_iters = 1000;
    Scalar grad_tol = 1e-6;
};

inline Vector newton(
    const Objective &obj,
    const Vector &x0,
    const NewtonParams &ntparams,
    bool record = false)
{
    Vector x = x0;
    BacktrackingStepsize bt_stepsize(0.5, 0.5);

    for (int k = 0; k < ntparams.max_iters; ++k)
    {
        Vector grad = obj.gradient(x);
        if (grad.norm() < ntparams.grad_tol)
        {
            break;
        }
        if (record)
        {
            std::cout << "iter " << k + 1 << " | gradient norm : " << grad.norm() << "\n ";
        }
        Vector p = -obj.hessian(x).ldlt().solve(grad); // TODO: ldlt might fail!
        Scalar t = bt_stepsize.choose(obj, x, p);
        x = x + t * p;
    }
    return x;
}