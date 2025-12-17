#pragma once

#include <iostream>
#include "optlib/core/objectives.hpp"
#include "optlib/stepsizes/backtracking.hpp"

using namespace optlib;

struct NewtonParams
{
    int max_iters = 1000;
    Scalar step_size = 1e-2;
    Scalar grad_tol = 1e-6;
};

inline Vector newton(
    const Objective &obj,
    const Vector &x0,
    const NewtonParams &ntparams,
    bool record = false)
{
    Vector x = x0;
    BacktrackParams btparams;

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
        Vector dx = -obj.hessian(x).ldlt().solve(grad); // TODO: ldlt might fail!
        Scalar bt_stepsize = backtrack_search(obj, x, dx, btparams);
        x += bt_stepsize * dx;
    }
    return x;
}