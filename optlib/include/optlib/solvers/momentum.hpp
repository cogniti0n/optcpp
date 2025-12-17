#pragma once

#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/stepsizes/backtracking.hpp"

using namespace optlib;

struct MomentumParams
{
    int max_iters = 1000;
    Scalar step_size = 1e-2;
    Scalar grad_tol = 1e-6;
};

inline Vector momentum(
    const Objective &obj,
    const Vector &x0,
    const MomentumParams &mtparams,
    bool record = false)
{
    Vector x = x0;
    Vector y = x0;

    for (int k = 0; k < mtparams.max_iters; ++k)
    {
        Vector prev = x;

        Vector grad_y = obj.gradient(y);
        if (grad_y.norm() < mtparams.grad_tol)
        {
            break;
        }
        if (record)
        {
            std::cout << "iter " << k + 1 << " | gradient norm : " << grad_y.norm() << "\n ";
        }
        x = y - mtparams.step_size * grad_y;
        y = x + k / (k + 3) * (x - prev);
    }
    return x;
}