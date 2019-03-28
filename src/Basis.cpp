#include<iostream>
#include<array>
#include "Basis.h"

using namespace std;

Basis::Basis(const Node &N1, const Node &N2, const Node &N3) {// : coords{ N1.getCoord(), N2.getCoord(), N3.getCoord() }
    coords[0] = N1.getCoord();
    coords[1] = N2.getCoord();
    coords[2] = N3.getCoord();
    //cout << "Basis initialized with x1 y1 x2 y2 x3 y3: " << coords[0][0] << " " << coords[0][1] << " " << coords[1][0] << " " << coords[1][1] << " " << coords[2][0] << " " << coords[2][1] << endl;
    dNdE[0][0] = -1.0;
    dNdE[0][1] = -1.0;
    dNdE[1][0] = 1.0;
    dNdE[1][1] = 0.0;
    dNdE[2][0] = 0.0;
    dNdE[2][1] = 1.0;
    calcDetJ();
    calcDEDX();
}

array<double, 3> Basis::calcBasis(const double &xi, const double &eta) {
    array<double, 3> basis_values{};
    basis_values[0] = 1 - xi - eta;
    basis_values[1] = xi;
    basis_values[2] = eta;
    return basis_values;
}

array<array<double,2>, 3> Basis::calcBasisDer() {
    array<array<double,2>, 3> basis_der_values{};

    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
            basis_der_values[i][j] = dNdE[i][0]*dEdX[0][j] + dNdE[i][1]*dEdX[1][j];
    }
    return basis_der_values;
}

void Basis::calcDEDX() {
    dEdX[0][0] = (coords[2][1] - coords[0][1]) / detJ;
    dEdX[0][1] = (coords[0][0] - coords[2][0]) / detJ;
    dEdX[1][0] = (coords[0][1] - coords[1][1]) / detJ;
    dEdX[1][1] = (coords[1][0] - coords[0][0]) / detJ;
}

void Basis::calcDetJ() {
    detJ = (coords[1][0] - coords[0][0]) * (coords[2][1] - coords[0][1]) -
           (coords[1][1] - coords[0][1]) * (coords[2][0] - coords[0][0]);
}

const double Basis::getDetJ() {
    return detJ;
}

Basis::~Basis() { ; }
