#pragma once

#include "optlib/core/objectives.hpp"
#include "optlib/core/stepsizes.hpp"

using namespace optlib;

class BacktrackingStepsize : StepsizePolicy
{
public:
    BacktrackingStepsize(
        Scalar alpha_,
        Scalar beta_,
        Scalar stepsize_threshold_ = 1e-8) : alpha(alpha_), beta(beta_), stepsize_threshold(stepsize_threshold_) {}
    Scalar choose(
        const Objective &obj,
        const Vector &xk,
        const Vector &pk) override
    {
        Scalar t = 1.0;
        Scalar l = obj.gradient(xk).dot(pk);
        while (t > stepsize_threshold)
        {
            if (obj.value(xk + t * pk) < obj.value(xk) + alpha * t * l)
                break;
            t = beta * t;
        }
        return t;
    }
    void reset() override {}

private:
    Scalar alpha, beta, stepsize_threshold;
};