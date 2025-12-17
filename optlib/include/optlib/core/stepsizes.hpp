#pragma once

#include "objectives.hpp"
#include "types.hpp"

// objective abstract base class

using namespace optlib;

class StepsizePolicy
{
public:
    virtual ~StepsizePolicy() = default;

    virtual void reset() = 0;

    virtual Scalar choose(
        const Objective &obj,
        const Vector &xk,
        const Vector &pk) = 0;

private:
};
