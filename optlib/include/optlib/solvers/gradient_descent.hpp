#pragma once

#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/stepsizes/backtracking.hpp"

using namespace optlib;

struct GDParams
{
    int max_iters = 1000;
    Scalar grad_tol = 1e-6;
};

inline Vector gradient_descent(
    const Objective &obj,
    const Vector &x0,
    const GDParams &params,
    bool record = false)
{
    Vector x = x0;
    BacktrackingStepsize bt_stepsize(0.5, 0.5);

    for (int k = 0; k < params.max_iters; ++k)
    {
        Vector grad = obj.gradient(x);
        if (grad.norm() < params.grad_tol)
        {
            break;
        }
        if (record)
        {
            std::cout << "iter " << k + 1 << " | gradient norm: " << grad.norm() << "\n";
        }
        Scalar t = bt_stepsize.choose(obj, x, -grad);
        x = x - t * grad;
    }
    return x;
}