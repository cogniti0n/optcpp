#pragma once

#include "optlib/core/objectives.hpp"
#include "optlib/core/options.hpp"
#include "optlib/core/store_results.hpp"
#include "optlib/stepsizes/backtracking.hpp"

namespace optlib
{
    struct GDOptions
    {
        SolverOptions base;
    };

    inline SolverResult gradient_descent(
        const Objective &obj,
        const Vector &x0,
        const GDOptions &options,
        bool store_result = false)
    {
        SolverResult res;
        Vector x = x0;
        BacktrackingStepsize bt_stepsize(0.5, 0.5);

        int k;
        for (k = 0; k < options.base.max_iters; ++k)
        {
            Vector grad = obj.gradient(x);
            if (grad.norm() < options.base.grad_tol)
            {
                break;
            }
            Scalar t = bt_stepsize.choose(obj, x, -grad);
            x = x - t * grad;

            if (store_result)
            {
                res.history.push_back({k, obj.value(x), grad.norm(), t});
            }
        }
        res.x = x;
        res.f = obj.value(x);
        res.grad_norm = obj.gradient(x).norm();
        res.iters = k;
        return res;
    }
}