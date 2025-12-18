#pragma once

#include "types.hpp"

// objective abstract base class

namespace optlib
{
    class Objective
    {
    public:
        virtual ~Objective() = default;
        virtual Scalar value(const Vector &x) const = 0;
        virtual Vector gradient(const Vector &x) const = 0;
        virtual Matrix hessian(const Vector &x) const
        {
            throw std::logic_error("Hessian is not implemented");
        }
        int getdim() { return dim; }

    private:
        int dim;
    };
}