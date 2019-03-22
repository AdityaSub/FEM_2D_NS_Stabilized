// main.cpp
// Solve 2D Navier-Stokes equations (lid-driven cavity problem)

#include<iostream>
#include "Mesh.h"

using namespace std;

static char help[] = "Solves 2D Poisson's equation with KSP.\n\n";

int main(int argc, char *argv[]) {
    PetscInitialize(&argc, &argv, (char *) 0, help);

    string fileName = argv[1];
    double Re = atof(argv[2]);
    Mesh m(fileName, Re);
    //m.getGrid()[0].printElemStiffness();
    m.setInitialField();
    m.writeField("initial");
    m.Solve();
    m.writeField("solution");
}
