#pragma once

#include "optlib/core/objectives.hpp"
#include "optlib/core/options.hpp"
#include "optlib/core/store_results.hpp"
#include "optlib/stepsizes/backtracking.hpp"

namespace optlib
{
    struct MomentumOptions
    {
        SolverOptions base;
        Scalar alpha;
        Scalar beta;
    };

    inline SolverResult momentum(
        const Objective &obj,
        const Vector &x0,
        const MomentumOptions &options,
        bool store_result = false)
    {
        SolverResult res;
        Vector x = x0;
        Vector v = Vector::Zero(x0.size());

        int k;
        for (k = 0; k < options.base.max_iters; ++k)
        {
            Vector prev = x;

            Vector grad = obj.gradient(x);
            if (grad.norm() < options.base.grad_tol)
            {
                break;
            }
            v = options.beta * v - options.alpha * grad;
            x = x + v;

            if (store_result)
            {
                res.history.push_back({k, obj.value(x), grad.norm(), options.alpha});
            }
        }
        res.x = x;
        res.f = obj.value(x);
        res.grad_norm = obj.gradient(x).norm();
        res.iters = k;
        return res;
    }
}