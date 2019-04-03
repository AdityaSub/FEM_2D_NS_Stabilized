// main.cpp
// Solve 2D Navier-Stokes equations (steady-state)

#include<iostream>
#include "Mesh.h"

using namespace std;

static char help[] = "Solves 2D steady-state Navier Stokes equations with SNES.\n\n";

int main(int argc, char *argv[]) {
    PetscInitialize(&argc, &argv, (char *) 0, help);

    string fileName = argv[1];
    double Re = atof(argv[2]);

    Mesh m(fileName, Re);
    //m.getGrid()[0].printElemStiffness();
    m.setInitialField();
    m.Assemble(); // for J_const
    m.writeField("initial");
    m.Solve();
    m.writeField("solution");
}

