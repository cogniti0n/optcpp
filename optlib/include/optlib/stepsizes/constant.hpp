#pragma once

#include "optlib/core/objectives.hpp"
#include "optlib/core/stepsizes.hpp"

using namespace optlib;

class ConstantStepsize : public StepsizePolicy
{
public:
    ConstantStepsize(Scalar alpha_) : alpha(alpha_) {}
    Scalar choose(
        const Objective &,
        const Vector &,
        const Vector &) override
    {
        return alpha;
    }
    void reset() {}

private:
    Scalar alpha;
};