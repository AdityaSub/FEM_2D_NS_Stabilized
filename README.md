# FEM_2D_NS_Stabilized

2D finite-element solver for steady-state stabilized Navier-Stokes equations

2D finite-element solver for steady-state stabilized Navier-Stokes equations (moderate Reynolds numbers); uses linear basis functions but the code is written to be easily modularizable; supports arbitrary geometries (by appropriately modifying Dirichlet conditions in 'Mesh.cpp'; the solver uses 'eigen' and 'PETSc' libraries (with very little tweaking, the 'eigen' part of it can be done away with); this code is meant to serve as an instructional example only; the code currently supports only triangular meshes; sample solution fields and plots included under 'sample_solutions') for lid-driven cavity, flow through a channel, flow past a confined cylinder, and flow through a channel with a backward step
