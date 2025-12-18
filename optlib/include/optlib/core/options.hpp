#pragma once

namespace optlib
{
    struct SolverOptions
    {
        int max_iters = 1000;
        double grad_tol = 1e-8;

        bool verbose = false;
        int print_every = 0;

        bool store_history = true;
    };
}