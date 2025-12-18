#pragma once

#include <vector>
#include "optlib/core/types.hpp"

namespace optlib
{
    struct IterationRecord
    {
        int k;
        double f;
        double grad_norm;
        double step;
    };

    struct SolverResult
    {
        Vector x;

        int iters = 0;
        double f = 0.0;
        double grad_norm = 0.0;

        std::vector<IterationRecord> history;
    };
}
