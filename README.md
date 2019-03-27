# FEM_2D_NS_Stabilized

2D finite-element solver for steady-state stabilized Navier-Stokes equations

2D finite-element solver for steady-state stabilized Navier-Stokes equations (inertial flow); uses linear basis functions but the code is written to be easily modularizable; supports arbitrary geometries (by appropriately modifying Dirichlet conditions in 'Mesh.cpp'; the solver uses 'eigen' and 'PETSc' libraries (with very little tweaking, the 'eigen' part of it can be done away with); this code is meant to serve as an instructional example only; the code currently supports only triangular meshes; sample solution fields and plots included under 'sample_solutions' for lid-driven cavity, flow through a channel, flow past a confined cylinder, and flow through a channel with a backward step

###To install###: run 'cmake' using the 'CMakeLists.txt' provided, followed by 'make' - you would have to point to the correct PETSc paths first

###To run###: use the examples provided under 'sample_solutions' for command-line options for the code (including the mesh file and PETSc options - nonlinear/linear solvers, preconditioners, convergence tolerances etc.); as mentioned above, boundary conditions are specified under 'Mesh.cpp' for the 4 simple geometries provided here, you would need to write your own for custom geometries

###Mesh format###: first line: <num. nodes> <num. dimensions>, next <num. nodes> lines: <node x-coord.> <node y-coord.>, next line: <num. elements> <num. nodes/element>, next <num. elements> lines: connectivities (WARNING: although <num. dimensions> and <num. nodes/element> have been specified in the mesh as input, these are NOT used by the code to generalize the number of dimensions/basis functions and element-type - the code currently only supports triangular elements, but extension to other elements is possible: please find contact below for assistance with the same)
