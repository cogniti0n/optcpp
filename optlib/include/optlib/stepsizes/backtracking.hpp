#pragma once

#include "optlib/core/objectives.hpp"

using namespace optlib;

// TODO: fix parameters (maybe a universal stepsize solver class?)
struct BacktrackParams
{
    Scalar alpha = 0.5;
    Scalar beta = 0.5;
    Scalar stepsize_tol = 1e-8;
};

inline Scalar backtrack_search(
    const Objective &obj,
    const Vector &x,
    const Vector &dx,
    const BacktrackParams &btparams)
{
    Scalar t = 1.0;
    Scalar l = obj.gradient(x).dot(dx);
    while (t > btparams.stepsize_tol)
    {
        if (obj.value(x + t * dx) < obj.value(x) + btparams.alpha * t * l)
        {
            break;
        }
        t = btparams.beta * t;
    }
    return t;
}