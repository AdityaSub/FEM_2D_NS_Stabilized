#pragma once

#include<vector>
#include "Element.h"
#include<petscsnes.h>

class Mesh {
public:
    explicit Mesh(std::string &, double &); // constructor
    void readMesh(); // read mesh-data from file
    void setInitialField(); // set initial guess
    std::vector<Element> &getGrid(); // return reference to grid
    void Assemble(); // assemble linear system
    void Solve(); // solve non-linear system for unknowns
    void writeField(const std::string &);

    ~Mesh(); // destructor

    // TODO: figure out a way to make the PETSc functions 'friend' to this class so the following variables can be made 'private'
    Vec x, r;
    Mat J,J_const;
    double x_min = 0.0, y_min = 0.0, x_max = 0.0, y_max = 0.0; // mesh bounds
    double Re = 1.0;
    PetscErrorCode ierr;
    PetscScalar one = 1.0, zero = 0.0;

private:
    std::string meshFileName; // input mesh-filename
    std::vector<std::vector<double>> nodes; // node-list for grid
    std::vector<std::vector<int>> elements; // grid connectivities
    std::vector<Element> mesh; // grid-vector containing list of elements
    SNES snes;
    KSP ksp;
    PC pc;
    PetscInt its;
};
