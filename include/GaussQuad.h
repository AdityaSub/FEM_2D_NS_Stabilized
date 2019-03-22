//
// Created by adityak on 3/18/19.
//

#ifndef FEM_NS_GAUSSQUAD_H
#define FEM_NS_GAUSSQUAD_H

#include<Eigen/Dense>
#include<array>

class GaussQuad {
public:
    GaussQuad() {};

    inline Eigen::Matrix<double, 3, 2> &getQuadPts() { return gauss_pts; }

    inline std::array<double, 3> &getQuadWts() { return gauss_pt_weights; }

    ~GaussQuad() {};

private:
    Eigen::Matrix<double, 3, 2> gauss_pts = (Eigen::Matrix<double, 3, 2>() << 0.5, 0.0, 0.0, 0.5, 0.5, 0.5).finished();
    std::array<double, 3> gauss_pt_weights = {{1 / 3.0, 1 / 3.0, 1 / 3.0}};
};

#endif //FEM_NS_GAUSSQUAD_H
