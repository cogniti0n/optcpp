#pragma once

#include "optlib/core/objectives.hpp"
#include "optlib/core/options.hpp"
#include "optlib/core/store_results.hpp"
#include "optlib/stepsizes/backtracking.hpp"

namespace optlib
{
    struct NesterovOptions
    {
        SolverOptions base;
        Scalar alpha;
    };

    inline SolverResult nesterov(
        const Objective &obj,
        const Vector &x0,
        const NesterovOptions &options,
        bool store_result = false)
    {
        SolverResult res;
        Vector x = x0;
        Vector y = x0;

        int k;
        for (k = 0; k < options.base.max_iters; ++k)
        {
            Vector prev = x;

            Vector grad_y = obj.gradient(y);
            if (grad_y.norm() < options.base.grad_tol)
            {
                break;
            }
            x = y - options.alpha * grad_y;
            Scalar tmp = static_cast<Scalar>(k);
            y = x + tmp / (tmp + 3) * (x - prev);
            if (store_result)
            {
                res.history.push_back({k, obj.value(x), grad_y.norm(), options.alpha});
            }
        }
        res.x = x;
        res.f = obj.value(x);
        res.grad_norm = obj.gradient(x).norm();
        res.iters = k;
        return res;
    }
}