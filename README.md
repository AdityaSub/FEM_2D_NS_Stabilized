# FEM_2D_NS_Stabilized

2D finite-element solver for steady-state Navier-Stokes equations

2D finite-element solver for steady-state Navier-Stokes equations; uses linear basis functions but the code is written to be easily modularizable; currently supports square domains, but this can be extended to arbitrary geometries by modifying Dirichlet conditions in 'Mesh.cpp'; the solver uses 'eigen' and 'PETSc' libraries (with very little tweaking, the 'eigen' part of it can be done away with); this code is meant to serve as an instructional example only
