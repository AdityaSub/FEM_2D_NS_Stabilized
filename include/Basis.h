#pragma once

#include"Node.h"
#include <array>

class Basis {
public:
    Basis(const Node &, const Node &, const Node &); // default constructor
    std::array<double, 3> calcBasis(const double &,
                                    const double &); // returns 3 x 1 vector of basis function values at query point (for triangle elements)
    std::array<std::array<double,2>, 3>
    calcBasisDer(); // returns 3 x 2 dNdX matrix of basis function derivative values at query point (for triangle elements)
    void calcDEDX(); // return dEdN/dXdY
    void calcDetJ(); // calculate determinant of Jacobian
    const double getDetJ(); // return determinant of Jacobian
    ~Basis(); // destructor

private:
    std::array<std::array<double,2>, 3> dNdE{};
    std::array<std::array<double,2>, 2> dEdX{};
    std::array<std::array<double, 2>, 3> coords;
    double detJ;
};
