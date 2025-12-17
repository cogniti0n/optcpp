#pragma once

#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/stepsizes/backtracking.hpp"

using namespace optlib;

struct GDParams
{
    int max_iters = 1000;
    Scalar step_size = 1e-2;
    Scalar grad_tol = 1e-6;
};

inline Vector gradient_descent(
    const Objective &obj,
    const Vector &x0,
    const GDParams &params,
    bool record = false)
{
    Vector x = x0;
    BacktrackParams btparams;

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
        Scalar bt_stepsize = backtrack_search(obj, x, -grad, btparams);
        x -= bt_stepsize * grad;
    }
    return x;
}