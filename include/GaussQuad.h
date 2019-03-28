//
// Created by adityak on 3/18/19.
//

#ifndef FEM_NS_GAUSSQUAD_H
#define FEM_NS_GAUSSQUAD_H

#include<array>

class GaussQuad {
public:
    GaussQuad() = default;

    inline const std::array<std::array<double, 2>, 3> &getQuadPts() { return gauss_pts; }

    inline const std::array<double, 3> &getQuadWts() { return gauss_pt_weights; }

    ~GaussQuad() = default;

private:
    const std::array<std::array<double, 2>, 3> gauss_pts = {{{{0.5, 0.0}}, {{0.0, 0.5}}, {{0.5, 0.5}}}};
    const std::array<double, 3> gauss_pt_weights = {{1 / 3.0, 1 / 3.0, 1 / 3.0}};
};

#endif //FEM_NS_GAUSSQUAD_H
