#include<iostream>
#include<iomanip>
#include<vector>
#include<fstream>
//#include<cmath>
#include "Mesh.h"
#include "GaussQuad.h"

#define PI 3.14159265358979323846

using namespace std;

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

// constructor
Mesh::Mesh(string &fileName, double &Re_in) : meshFileName(fileName), Re(Re_in) {
    readMesh();
    for (size_t i = 0; i < static_cast<size_t>(elements.size()); i++) {
        const Node n1(elements[i][0], nodes[elements[i][0]][0], nodes[elements[i][0]][1]);
        const Node n2(elements[i][1], nodes[elements[i][1]][0], nodes[elements[i][1]][1]);
        const Node n3(elements[i][2], nodes[elements[i][2]][0], nodes[elements[i][2]][1]);
        mesh.emplace_back(Element(n1, n2, n3, (const int) i));
    }
    cout << "Grid generated! Bounds: x_min = " << x_min << ", y_min = " << y_min << ", x_max = " << x_max
         << ", y_max = " << y_max << ", nodes = " << nodes.size() << ", elements = " << elements.size() << endl;

    // calculate element stiffnesses (to be used in Jacobian and residual calculations later)
    for (auto &it : mesh)
        it.calcElementStiffness();

    MatCreate(PETSC_COMM_WORLD, &J);
    MatCreate(PETSC_COMM_WORLD, &J_const);
    //MatCreateSeqAIJ(PETSC_COMM_WORLD, (PetscInt) (3 * nodes.size()), (PetscInt) (3 * nodes.size()), PETSC_DEFAULT, NULL, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, (PetscInt) (3 * nodes.size()), (PetscInt) (3 * nodes.size()));
    MatSetUp(J);
    MatZeroEntries(J);
    MatSetSizes(J_const, PETSC_DECIDE, PETSC_DECIDE, (PetscInt) (3 * nodes.size()), (PetscInt) (3 * nodes.size()));
    MatSetUp(J_const);
    MatZeroEntries(J_const);
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, (PetscInt) (3 * nodes.size()));
    VecSetFromOptions(x);
    VecZeroEntries(x);
    VecDuplicate(x, &r);
}

// read mesh-data from file
void Mesh::readMesh() {
    ifstream mesh_read;
    mesh_read.open(meshFileName);
    // read node coordinates
    unsigned long n_nodes, n_dim; // number of nodes, number of dimensions
    mesh_read >> n_nodes >> n_dim;
    nodes.resize(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        nodes[i].resize(n_dim);
    }
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < n_dim; j++) {
            mesh_read >> nodes[i][j];
        }
    }

    // read connectivities
    unsigned long n_elems, n_nodes_per_elem;
    mesh_read >> n_elems >> n_nodes_per_elem; // number of elements, number of nodes per element
    elements.resize(n_elems);
    for (int i = 0; i < n_elems; i++) {
        elements[i].resize(n_nodes_per_elem);
    }
    for (int i = 0; i < n_elems; i++) {
        for (int j = 0; j < n_nodes_per_elem; j++) {
            mesh_read >> elements[i][j];
            elements[i][j] -= 1; // '0' - indexing
        }
    }
    mesh_read.close();

    for (int i = 0; i < n_nodes; i++) {
        if (x_max < nodes[i][0])
            x_max = nodes[i][0];
        if (x_min > nodes[i][0])
            x_min = nodes[i][0];
        if (y_max < nodes[i][1])
            y_max = nodes[i][1];
        if (y_min > nodes[i][1])
            y_min = nodes[i][1];
    }
}

void Mesh::setInitialField() {
    array<int, 3> nodeIDs = {0};
    array<array<double, 2>, 3> elemCoords = {{0}};
    PetscInt i1, new_ind_i;
    PetscScalar guess_val;

    // initial guess/sets Dirichlet BCs
    for (auto &it : mesh) {
        nodeIDs[0] = it.getNode1().getID();
        nodeIDs[1] = it.getNode2().getID();
        nodeIDs[2] = it.getNode3().getID();
        elemCoords[0] = it.getNode1().getCoord();
        elemCoords[1] = it.getNode2().getCoord();
        elemCoords[2] = it.getNode3().getCoord();

        for (size_t i = 0; i < 3; i++) {
            new_ind_i = 3 * nodeIDs[i]; // global mapping
            /*lid-driven cavity*/
            if (fabs(elemCoords[i][1] - y_max) < 1e-5) { // top boundary: x-velocity = 1
                VecSetValues(x, 1, &new_ind_i, &one, INSERT_VALUES);
            } else if ((fabs(elemCoords[i][0] - x_max) < 1e-5) &&
                       (fabs(elemCoords[i][1] - y_min) < 1e-5)) { // bottom right corner: pressure = 1
                i1 = new_ind_i + 2;
                VecSetValues(x, 1, &i1, &one, INSERT_VALUES);
            }
            /*channel flow*/
            /*if (fabs(elemCoords[i][0] - x_min) < 1e-5) { // left boundary: x-velocity = 1
                VecSetValues(x, 1, &new_ind_i, &one, INSERT_VALUES);
            } else if (fabs(elemCoords[i][0] - x_max) < 1e-5) { // right boundary: pressure = 0
                i1 = new_ind_i + 2;
                VecSetValues(x, 1, &i1, &zero, INSERT_VALUES);
            } else { // for channel with obstruction - set initial velocity guess to uniform flow or zero
                VecSetValues(x, 1, &new_ind_i, &zero, INSERT_VALUES);
            }*/
            /*channel with backward step*/
            /*if (fabs(elemCoords[i][0] - x_min) < 1e-5) { // left boundary: x-velocity = parabolic (developed)
                guess_val = 16 * (1.0 - elemCoords[i][1]) * (-0.5 + elemCoords[i][1]);
                VecSetValues(x, 1, &new_ind_i, &guess_val, INSERT_VALUES);
                i1 = new_ind_i + 1;
                VecSetValues(x, 1, &i1, &zero, INSERT_VALUES);
            } else if ((fabs(elemCoords[i][0] - x_max) < 1e-5) &&
                       (fabs(elemCoords[i][1] - y_min) < 1e-5)) { // bottom right corner: pressure = 0
                i1 = new_ind_i + 2;
                VecSetValues(x, 1, &i1, &zero, INSERT_VALUES);
            } else if (((fabs(elemCoords[i][0] - 1.0) < 1e-5) && (elemCoords[i][1] >= y_min) &&
                        (elemCoords[i][1] <= 0.5)) ||
                       ((fabs(elemCoords[i][1] - 0.5) < 1e-5) && (elemCoords[i][0] >= x_min) &&
                        (elemCoords[i][0] <= 1.0))) { // the step - no-slip and no-penetration
                i1 = new_ind_i + 1;
                VecSetValues(x, 1, &new_ind_i, &zero, INSERT_VALUES);
                VecSetValues(x, 1, &i1, &zero, INSERT_VALUES);
            } else if ((elemCoords[i][1] == y_min) ||
                       (elemCoords[i][1] == y_max)) {
                i1 = new_ind_i + 1;
                VecSetValues(x, 1, &new_ind_i, &zero, INSERT_VALUES);
                VecSetValues(x, 1, &i1, &zero, INSERT_VALUES);
            } else { // for channel with obstruction - set initial velocity guess to zero
                VecSetValues(x, 1, &new_ind_i, &zero, INSERT_VALUES);
            }*/
        }
    }

    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    //VecView(x,PETSC_VIEWER_STDOUT_SELF);
}

// assemble global stiffness matrix
void Mesh::Assemble() {
    // loop through elements and add up contributions (except at Dirichlet boundary nodes)
    //MatZeroEntries(J);
    vector<int> nodeIDs;
    nodeIDs.resize(3);
    array<array<double, 2>, 3> elemCoords = {{0}};
    double norm_coeff;
    PetscInt new_ind_i, new_ind_j, new_nodeID_i, new_nodeID_j;
    for (auto &it : mesh) {
        //MatAssemblyBegin(J, MAT_FLUSH_ASSEMBLY);
        //MatAssemblyEnd(J, MAT_FLUSH_ASSEMBLY);
        MatAssemblyBegin(J_const, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(J_const, MAT_FLUSH_ASSEMBLY);
        nodeIDs[0] = it.getNode1().getID();
        nodeIDs[1] = it.getNode2().getID();
        nodeIDs[2] = it.getNode3().getID();
        elemCoords[0] = it.getNode1().getCoord();
        elemCoords[1] = it.getNode2().getCoord();
        elemCoords[2] = it.getNode3().getCoord();
        const array<array<double, 9>, 9> elemStiffness = it.getElemStiffness();
        for (size_t i = 0; i < 9; i++) {
            ierr = MatAssemblyBegin(J_const, MAT_FLUSH_ASSEMBLY);
            ierr = MatAssemblyEnd(J_const, MAT_FLUSH_ASSEMBLY);
            new_nodeID_i = (i < 3) ? (nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (nodeIDs[1]) : (
                    nodeIDs[2]));
            new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

            /*lid-driven cavity*/
            // left, right, bottom, top boundaries for velocities (and bottom right corner for pressure) - Dirichlet BCs
            if ((((elemCoords[i / 3][0] == x_min) || (elemCoords[i / 3][0] == x_max) ||
                  (elemCoords[i / 3][1] == y_min) ||
                  (elemCoords[i / 3][1] == y_max)) && ((i % 3) < 2)) || ((fabs(elemCoords[i / 3][0] - x_max) < 1e-5) &&
                                                                         (fabs(elemCoords[i / 3][1] - y_min) < 1e-5) &&
                                                                         ((i % 3) ==
                                                                          2))) {
                //ierr = MatSetValues(J, 1, &new_ind_i, 1, &new_ind_i, &one, INSERT_VALUES);
                ierr = MatSetValues(J_const, 1, &new_ind_i, 1, &new_ind_i, &one, INSERT_VALUES);
                it.setElemFlag(i, false);
            } else {
                ierr = MatAssemblyBegin(J, MAT_FLUSH_ASSEMBLY);
                ierr = MatAssemblyEnd(J, MAT_FLUSH_ASSEMBLY);
                for (size_t j = 0; j < 9; j++) {
                    //globalStiffness[nodeIDs[i]][nodeIDs[j]] += elemStiffness[i][j];
                    norm_coeff = elemStiffness[i][j] / Re;
                    new_nodeID_j = (j < 3) ? (nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (nodeIDs[1]) : (
                            nodeIDs[2]));
                    new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                    //ierr = MatSetValues(J, 1, &new_ind_i, 1, &new_ind_j, &norm_coeff,
                    //                    ADD_VALUES);
                    ierr = MatSetValues(J_const, 1, &new_ind_i, 1, &new_ind_j, &norm_coeff,
                                        ADD_VALUES);
                    //cout << "Ag for " << nodeIDs[i] << ", " << nodeIDs[j] << ": " << globalStiffness[nodeIDs[i]][nodeIDs[j]] << endl;
                }
            }

            /*channel flow*/
            // left, bottom, top boundaries for velocities, cavity boundary (if applicable) (and bottom right corner for pressure) - Dirichlet BCs
//            if ((((elemCoords[i / 3][0] == x_min) ||
//                  (elemCoords[i / 3][1] == y_min) ||
//                  (elemCoords[i / 3][1] == y_max)) && ((i % 3) < 2)) || ((fabs(elemCoords[i / 3][0] - x_max) < 1e-5) &&
//                                                                         ((i % 3) ==
//                                                                          2)) ||
//                ((fabs(sqrt(pow(elemCoords[i / 3][0] - 5, 2.0) + pow(elemCoords[i / 3][1] - 1, 2.0)) - 0.5) < 1e-5) &&
//                 ((i % 3) < 2))) {
//
//                /*channel with backward step*/
//                /*if ((((elemCoords[i / 3][0] == x_min) ||
//                      (elemCoords[i / 3][1] == y_min) ||
//                      (elemCoords[i / 3][1] == y_max)) && ((i % 3) < 2)) ||
//                    (((fabs(elemCoords[i / 3][0] - 1.0) < 1e-5) && (elemCoords[i / 3][1] >= y_min) &&
//                      (elemCoords[i / 3][1] <= 0.5)) && ((i % 3) < 2)) ||
//                    (((fabs(elemCoords[i / 3][1] - 0.5) < 1e-5) && (elemCoords[i / 3][0] >= x_min) &&
//                      (elemCoords[i / 3][0] <= 1.0)) && ((i % 3) < 2)) || ((fabs(elemCoords[i / 3][0] - x_max) < 1e-5) &&
//                                                                           (fabs(elemCoords[i / 3][1] - y_min) < 1e-5) &&
//                                                                           ((i % 3) ==
//                                                                            2))) {*/
//                //ierr = MatSetValues(J, 1, &new_ind_i, 1, &new_ind_i, &one, INSERT_VALUES);
//                ierr = MatSetValues(J_const, 1, &new_ind_i, 1, &new_ind_i, &one, INSERT_VALUES);
//                it.setElemFlag(i, false);
//            } else {
////                ierr = MatAssemblyBegin(J, MAT_FLUSH_ASSEMBLY);
////                ierr = MatAssemblyEnd(J, MAT_FLUSH_ASSEMBLY);
//                ierr = MatAssemblyBegin(J_const, MAT_FLUSH_ASSEMBLY);
//                ierr = MatAssemblyEnd(J_const, MAT_FLUSH_ASSEMBLY);
//                for (size_t j = 0; j < 9; j++) {
//                    //globalStiffness[nodeIDs[i]][nodeIDs[j]] += elemStiffness[i][j];
//                    norm_coeff = elemStiffness[i][j] / Re;
//                    new_nodeID_j = (j < 3) ? (nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (nodeIDs[1]) : (
//                            nodeIDs[2]));
//                    new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
//                    //ierr = MatSetValues(J, 1, &new_ind_i, 1, &new_ind_j, &norm_coeff,
//                    //                    ADD_VALUES);
//                    ierr = MatSetValues(J_const, 1, &new_ind_i, 1, &new_ind_j, &norm_coeff,
//                                        ADD_VALUES);
//                    //cout << "Ag for " << nodeIDs[i] << ", " << nodeIDs[j] << ": " << globalStiffness[nodeIDs[i]][nodeIDs[j]] << endl;
//                }
//            }
        }
    }
    //cout << "J assembled!" << endl;
//    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
//    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(J_const, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J_const, MAT_FINAL_ASSEMBLY);
    //MatView(J,PETSC_VIEWER_STDOUT_SELF);
}

void Mesh::Solve() {
    SNESCreate(PETSC_COMM_WORLD, &snes);
    SNESSetFunction(snes, this->r, FormFunction, this);
    SNESSetJacobian(snes, this->J, this->J, FormJacobian, this);

    SNESGetKSP(snes, &ksp);
    KSPGetPC(ksp, &pc);
    PCSetFromOptions(pc);
    //PCSetType(pc, PCLU);
    //KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);

    double atol = 1e-6, rtol = 1e-12, stol = 1e-12;
    int maxit = 500, maxf = 1000;
    SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf);
    SNESSetFromOptions(snes);
    SNESSolve(snes, PETSC_NULL, x);
    SNESGetIterationNumber(snes, &its);
    //PetscPrintf(PETSC_COMM_WORLD,"number of SNES iterations = %D\n",its);
    //VecView(x,PETSC_VIEWER_STDOUT_SELF);
}

// return reference to grid
vector<Element> &Mesh::getGrid() {
    return mesh;
}

void Mesh::writeField(const string &solName) {
    double temp_u = 0.0, temp_v = 0.0, temp_p = 0.0;
    PetscInt i1 = 0, i2 = 0;
    ofstream outFile;
    stringstream ss;
    ss << solName << ".plt";
    outFile.open(ss.str());
    outFile << "TITLE = \"SOLUTION_FIELD\"" << endl;
    outFile << R"(VARIABLES = "x" "y" "u" "v" "p")" << endl;
    outFile << "ZONE N=" << nodes.size() << ", E=" << elements.size() << ", F=FEPOINT, ET=TRIANGLE" << endl;
    for (int i = 0; i < 3 * nodes.size() - 2; i += 3) {
        i1 = i + 1;
        i2 = i + 2;
        VecGetValues(x, 1, &i, &temp_u);
        VecGetValues(x, 1, &i1, &temp_v);
        VecGetValues(x, 1, &i2, &temp_p);
        outFile << setprecision(6) << nodes[i / 3][0] << "\t" << setprecision(6) << nodes[i / 3][1] << "\t"
                << setprecision(6)
                << temp_u << "\t" << setprecision(6) << temp_v << "\t" << setprecision(6) << temp_p
                << endl;
    }

    for (auto &element : elements) {
        outFile << element[0] + 1 << "\t" << element[1] + 1 << "\t" << element[2] + 1 << endl;
    }
    outFile.close();
}

// destructor
Mesh::~Mesh() {
    MatDestroy(&J);
    VecDestroy(&x);
    SNESDestroy(&snes);
    PetscFinalize();
    cout << "Grid destroyed!" << endl;
}

// residual calculation
PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *meshPtr) {
    auto mPtr = (Mesh *) meshPtr;
    PetscScalar v, v1, v2, v3;
    array<array<double, 9>, 9> temp_vals = {{0}}; // temporary array for Jacobian values
    array<double, 3> basis_values{};
    array<array<double,2>, 3> der_values{};
    array<array<double, 2>, 3> elemCoords = {{0}};
    PetscInt new_ind_i, new_ind_j, i1, i2, i3, i4, i5, i6, i7, i8, i9, new_nodeID_i, new_nodeID_j, basis_ind;
    array<double, 9> curr_dof_vals = {
            0}; // previous iteration nodal values for (u,v,p) for current element being assembled
    array<double, 3> curr_GP_vals = {0}; // current gauss point values for DOFs for current element assembly
    array<array<double,2>, 3> curr_GP_val_ders{}; // current gauss point derivative values for DOFs for current element assembly
    array<int, 3> curr_nodeIDs = {0}; // node IDs of current element being assembled
    GaussQuad quadObj;
    array<double, 3> gauss_pt_weights = quadObj.getQuadWts();
    array<array<double,2>, 3> gauss_pts = quadObj.getQuadPts();
    double tau_supg, tau_pspg, h = 0, h_hash, z, Re_u, Re_U, u_mag, U_mag = mPtr->one, v_j, v_r;
    array<PetscInt, 9> i_array{};
    array<array<PetscScalar, 3>, 3> A_block{};
    array<PetscInt, 3> A_block_row_ind{};
    array<PetscInt, 3> A_block_col_ind{};
    array<PetscScalar, 3> res_block{};

    //mPtr->Assemble();
    MatZeroEntries(mPtr->J);
    MatCopy(mPtr->J_const, mPtr->J, DIFFERENT_NONZERO_PATTERN);
    VecZeroEntries(f);

    //VecView(x,PETSC_VIEWER_STDOUT_SELF);
    for (auto it = mPtr->getGrid().begin(); it != mPtr->getGrid().end(); it++) {
        curr_nodeIDs[0] = it->getNode1().getID();
        curr_nodeIDs[1] = it->getNode2().getID();
        curr_nodeIDs[2] = it->getNode3().getID();
        elemCoords[0] = it->getNode1().getCoord();
        elemCoords[1] = it->getNode2().getCoord();
        elemCoords[2] = it->getNode3().getCoord();

//        i1 = 3 * curr_nodeIDs[0];
//        i2 = 3 * curr_nodeIDs[0] + 1;
//        i3 = 3 * curr_nodeIDs[0] + 2;
//        i4 = 3 * curr_nodeIDs[1];
//        i5 = 3 * curr_nodeIDs[1] + 1;
//        i6 = 3 * curr_nodeIDs[1] + 2;
//        i7 = 3 * curr_nodeIDs[2];
//        i8 = 3 * curr_nodeIDs[2] + 1;
//        i9 = 3 * curr_nodeIDs[2] + 2;

        i_array[0] = 3 * curr_nodeIDs[0];
        i_array[1] = 3 * curr_nodeIDs[0] + 1;
        i_array[2] = 3 * curr_nodeIDs[0] + 2;
        i_array[3] = 3 * curr_nodeIDs[1];
        i_array[4] = 3 * curr_nodeIDs[1] + 1;
        i_array[5] = 3 * curr_nodeIDs[1] + 2;
        i_array[6] = 3 * curr_nodeIDs[2];
        i_array[7] = 3 * curr_nodeIDs[2] + 1;
        i_array[8] = 3 * curr_nodeIDs[2] + 2;

        // previous iteration DOF values
//        VecGetValues(x, 1, &i1, &curr_dof_vals[0]); // u1
//        VecGetValues(x, 1, &i2, &curr_dof_vals[1]); // v1
//        VecGetValues(x, 1, &i3, &curr_dof_vals[2]); // p1
//        VecGetValues(x, 1, &i4, &curr_dof_vals[3]); // u2
//        VecGetValues(x, 1, &i5, &curr_dof_vals[4]); // v2
//        VecGetValues(x, 1, &i6, &curr_dof_vals[5]); // p2
//        VecGetValues(x, 1, &i7, &curr_dof_vals[6]); // u3
//        VecGetValues(x, 1, &i8, &curr_dof_vals[7]); // v3
//        VecGetValues(x, 1, &i9, &curr_dof_vals[8]); // p3
        VecGetValues(x, 9, &i_array[0], &curr_dof_vals[0]);

        der_values = it->getBasis().calcBasisDer(); // TODO: should take in Gauss point for higher-order basis functions, but here it is outside loop because of linear basis - WARNING!!!!

        // (u_x,u_y) at GP // TODO: as noted above, the following derivatives at Gauss points should feature inside the loop, rather than outside as done here, which is only for constant derivatives (linear triangles)
        curr_GP_val_ders[0][0] = curr_dof_vals[0] * der_values[0][0] + curr_dof_vals[3] * der_values[1][0] +
                                 curr_dof_vals[6] * der_values[2][0];
        curr_GP_val_ders[0][1] = curr_dof_vals[0] * der_values[0][1] + curr_dof_vals[3] * der_values[1][1] +
                                 curr_dof_vals[6] * der_values[2][1];

        // (v_x,v_y) at GP
        curr_GP_val_ders[1][0] = curr_dof_vals[1] * der_values[0][0] + curr_dof_vals[4] * der_values[1][0] +
                                 curr_dof_vals[7] * der_values[2][0];
        curr_GP_val_ders[1][1] = curr_dof_vals[1] * der_values[0][1] + curr_dof_vals[4] * der_values[1][1] +
                                 curr_dof_vals[7] * der_values[2][1];

        // (p_x,p_y) at GP
        curr_GP_val_ders[2][0] = curr_dof_vals[2] * der_values[0][0] + curr_dof_vals[5] * der_values[1][0] +
                                 curr_dof_vals[8] * der_values[2][0];
        curr_GP_val_ders[2][1] = curr_dof_vals[2] * der_values[0][1] + curr_dof_vals[5] * der_values[1][1] +
                                 curr_dof_vals[8] * der_values[2][1];

        // ###### Jacobian matrix ######
//        for (size_t i = 0; i < 9; i++) { // loop over ndofs * nodes ('i'th equation/row for an element)
//            /*if ((elemCoords[i / 3][0] != mPtr->x_min) && (elemCoords[i / 3][0] != mPtr->x_max) &&
//                (elemCoords[i / 3][1] != mPtr->y_min) &&
//                (elemCoords[i / 3][1] != mPtr->y_max)) { // left, right, bottom, top boundaries*/
//
//            basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);
//
//            /*lid-driven cavity*/
//            /*if (!((((elemCoords[i / 3][0] == mPtr->x_min) || (elemCoords[i / 3][0] == mPtr->x_max) ||
//                    (elemCoords[i / 3][1] == mPtr->y_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_max)) && ((i % 3) < 2)) ||
//                  ((fabs(elemCoords[i / 3][0] - mPtr->x_max) < 1e-5) &&
//                   (fabs(elemCoords[i / 3][1] - mPtr->y_min) < 1e-5) &&
//                   ((i % 3) ==
//                    2)))) {*/
//            /*channel flow*/
//            if (!((((elemCoords[i / 3][0] == mPtr->x_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_max)) && ((i % 3) < 2)) ||
//                  ((fabs(elemCoords[i / 3][0] - mPtr->x_max) < 1e-5) &&
//                   ((i % 3) ==
//                    2)) ||
//                  ((fabs(sqrt(pow(elemCoords[i / 3][0] - 5, 2.0) + pow(elemCoords[i / 3][1] - 1, 2.0)) - 0.5) < 1e-5) &&
//                   ((i % 3) < 2)))) {
//                //cout << "elem " << it->getElemID() << " node " << i/3 << " and nodal dof " << i << " are in" << endl;
//                for (size_t j = 0; j < 9; j++) { // 'j'th column for 'i'th row
//                    v = 0.0;
//                    for (size_t k = 0; k < gauss_pts.rows(); k++) {
//                        // basis values at current GP
//                        basis_values = it->getBasis().calcBasis(gauss_pts(k, 0), gauss_pts(k, 1));
//
//                        // (u,v,p) at GP
//                        curr_GP_vals[0] = curr_dof_vals[0] * basis_values[0] + curr_dof_vals[3] * basis_values[1] +
//                                          curr_dof_vals[6] * basis_values[2]; // hard-coded for linear triangles
//                        curr_GP_vals[1] = curr_dof_vals[1] * basis_values[0] + curr_dof_vals[4] * basis_values[1] +
//                                          curr_dof_vals[7] * basis_values[2];
//                        curr_GP_vals[2] = curr_dof_vals[2] * basis_values[0] + curr_dof_vals[5] * basis_values[1] +
//                                          curr_dof_vals[8] * basis_values[2];
//
//                        // SUPG calculation
//                        u_mag = sqrt(pow(curr_GP_vals[0], 2.0) + pow(curr_GP_vals[1], 2.0));
//                        h = 0.0;
//                        for (size_t i_supg = 0; i_supg < 3; i_supg++)
//                            h += curr_GP_vals[0] * der_values(i_supg, 0) + curr_GP_vals[1] * der_values(i_supg, 1);
//
//                        h = (h < 1e-6) ? 0 : (2.0 * fabs(u_mag / h));
//                        Re_u = u_mag * h / (2 * nu);
//                        z = ((Re_u >= 0) && (Re_u <= 3)) ? Re_u / 3.0 : 1.0;
//                        tau_supg = (u_mag < 1e-6) ? 0 : (h * z / (2 * u_mag));
//
//                        // PSPG calculation
//                        h_hash = sqrt(it->getBasis().getDetJ() / (2 * PI));
//                        Re_U = U_mag * h_hash / (2 * nu);
//                        z = ((Re_U >= 0) && (Re_U <= 3)) ? Re_U / 3.0 : 1.0;
//                        tau_pspg = h_hash * z / (2 * U_mag);
//
//                        // x-momentum
//                        //if (((i >= 0) && (i <= 2)) && ((j == 0) || (j == 3) || (j == 6))) { // coeffs. of delta u_i
//                        if (((i == 0) || (i == 3) || (i == 6)) && ((j == 0) || (j == 3) || (j == 6))) {
//                            /*v += 0.5 * gauss_pt_weights[k] * (basis_values[i] *
//                                                              (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                               curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                               basis_values[j / 3] * curr_GP_val_ders(0, 0))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg * (der_values(i, 0) * basis_values[j / 3] *
//                                                                         (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                                                          curr_GP_vals[1] * curr_GP_val_ders(0, 1)) +
//                                                                         (curr_GP_vals[0] * der_values(i, 0) +
//                                                                          curr_GP_vals[1] * der_values(i, 1)) *
//                                                                         (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                                          basis_values[j / 3] *
//                                                                          curr_GP_val_ders(0, 0))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (der_values(i, 0) * basis_values[j / 3] * curr_GP_val_ders(2, 0)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] *
//                                                              (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                               curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                               basis_values[j / 3] * curr_GP_val_ders(0, 0))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (der_values(basis_ind, 0) * basis_values[j / 3] *
//                                  (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                   curr_GP_vals[1] * curr_GP_val_ders(0, 1)) +
//                                  (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                   curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                  (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                   curr_GP_vals[1] * der_values(j / 3, 1) +
//                                   basis_values[j / 3] *
//                                   curr_GP_val_ders(0, 0))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (der_values(basis_ind, 0) * basis_values[j / 3] * curr_GP_val_ders(2, 0)) *
//                                 it->getBasis().getDetJ();
//                        }
//                        //if (((i >= 0) && (i <= 2)) && ((j == 1) || (j == 4) || (j == 7))) { // coeffs. of delta v_i
//                        if (((i == 0) || (i == 3) || (i == 6)) && ((j == 1) || (j == 4) || (j == 7))) {
//                            /*v += 0.5 * gauss_pt_weights[k] *
//                                 (basis_values[i] * basis_values[j / 3] * curr_GP_val_ders(0, 1)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg * (basis_values[j / 3] * (der_values(i, 1) *
//                                                                                                (curr_GP_vals[0] *
//                                                                                                 curr_GP_val_ders(0,
//                                                                                                                  0) +
//                                                                                                 curr_GP_vals[1] *
//                                                                                                 curr_GP_val_ders(0,
//                                                                                                                  1)) +
//                                                                                                curr_GP_val_ders(0, 1) *
//                                                                                                (curr_GP_vals[0] *
//                                                                                                 der_values(i, 0) +
//                                                                                                 curr_GP_vals[1] *
//                                                                                                 der_values(i, 1)))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                 (der_values(i, 1) * basis_values[j / 3] * curr_GP_val_ders(2, 0)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] *
//                                 (basis_values[basis_ind] * basis_values[j / 3] * curr_GP_val_ders(0, 1)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (basis_values[j / 3] * (der_values(basis_ind, 1) *
//                                                         (curr_GP_vals[0] *
//                                                          curr_GP_val_ders(0,
//                                                                           0) +
//                                                          curr_GP_vals[1] *
//                                                          curr_GP_val_ders(0,
//                                                                           1)) +
//                                                         curr_GP_val_ders(0, 1) *
//                                                         (curr_GP_vals[0] *
//                                                          der_values(basis_ind, 0) +
//                                                          curr_GP_vals[1] *
//                                                          der_values(basis_ind, 1)))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                 (der_values(basis_ind, 1) * basis_values[j / 3] * curr_GP_val_ders(2, 0)) *
//                                 it->getBasis().getDetJ();
//                        }
//                        //if (((i >= 0) && (i <= 2)) && ((j == 2) || (j == 5) || (j == 8))) { // coeffs. of delta p_i
//                        if (((i == 0) || (i == 3) || (i == 6)) && ((j == 2) || (j == 5) || (j == 8))) {
//                            /*v -= 0.5 * gauss_pt_weights[k] * (der_values(i, 0) * basis_values[j / 3]) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (curr_GP_vals[0] * der_values(i, 0) + curr_GP_vals[1] * der_values(i, 1)) *
//                                 (der_values(j / 3, 0)) * it->getBasis().getDetJ();*/
//                            v -= 0.5 * gauss_pt_weights[k] * (der_values(basis_ind, 0) * basis_values[j / 3]) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                  curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                 (der_values(j / 3, 0)) * it->getBasis().getDetJ();
//                        }
//
//                        // y-momentum
//                        //if (((i >= 3) && (i <= 5)) && ((j == 0) || (j == 3) || (j == 6))) { // coeffs. of delta u_i
//                        if (((i == 1) || (i == 4) || (i == 7)) && ((j == 0) || (j == 3) || (j == 6))) {
//                            /*v += 0.5 * gauss_pt_weights[k] *
//                                 (basis_values[i - 3] * basis_values[j / 3] * curr_GP_val_ders(1, 0)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg * (basis_values[j / 3] * (der_values(i - 3, 0) *
//                                                                                                (curr_GP_vals[0] *
//                                                                                                 curr_GP_val_ders(1,
//                                                                                                                  0) +
//                                                                                                 curr_GP_vals[1] *
//                                                                                                 curr_GP_val_ders(1,
//                                                                                                                  1)) +
//                                                                                                curr_GP_val_ders(1, 0) *
//                                                                                                (curr_GP_vals[0] *
//                                                                                                 der_values(i - 3, 0) +
//                                                                                                 curr_GP_vals[1] *
//                                                                                                 der_values(i - 3,
//                                                                                                            1)))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                 (der_values(i - 3, 0) * basis_values[j / 3] * curr_GP_val_ders(2, 1)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] *
//                                 (basis_values[basis_ind] * basis_values[j / 3] * curr_GP_val_ders(1, 0)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (basis_values[j / 3] * (der_values(basis_ind, 0) *
//                                                         (curr_GP_vals[0] *
//                                                          curr_GP_val_ders(1,
//                                                                           0) +
//                                                          curr_GP_vals[1] *
//                                                          curr_GP_val_ders(1,
//                                                                           1)) +
//                                                         curr_GP_val_ders(1, 0) *
//                                                         (curr_GP_vals[0] *
//                                                          der_values(basis_ind, 0) +
//                                                          curr_GP_vals[1] *
//                                                          der_values(basis_ind,
//                                                                     1)))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                 (der_values(basis_ind, 0) * basis_values[j / 3] * curr_GP_val_ders(2, 1)) *
//                                 it->getBasis().getDetJ();
//                        }
//                        //if (((i >= 3) && (i <= 5)) && ((j == 1) || (j == 4) || (j == 7))) { // coeffs. of delta v_i
//                        if (((i == 1) || (i == 4) || (i == 7)) && ((j == 1) || (j == 4) || (j == 7))) {
//                            /*v += 0.5 * gauss_pt_weights[k] * (basis_values[i - 3] *
//                                                              (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                               curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                               basis_values[j / 3] * curr_GP_val_ders(1, 1))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg * (der_values(i - 3, 1) * basis_values[j / 3] *
//                                                                         (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                                          curr_GP_vals[1] * curr_GP_val_ders(1, 1)) +
//                                                                         (curr_GP_vals[0] * der_values(i - 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(i - 3, 1)) *
//                                                                         (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                                          basis_values[j / 3] *
//                                                                          curr_GP_val_ders(1, 1))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (der_values(i - 3, 1) * basis_values[j / 3] * curr_GP_val_ders(2, 1)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] *
//                                                              (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                               curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                               basis_values[j / 3] * curr_GP_val_ders(1, 1))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (der_values(basis_ind, 1) * basis_values[j / 3] *
//                                  (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                   curr_GP_vals[1] * curr_GP_val_ders(1, 1)) +
//                                  (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                   curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                  (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                   curr_GP_vals[1] * der_values(j / 3, 1) +
//                                   basis_values[j / 3] *
//                                   curr_GP_val_ders(1, 1))) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (der_values(basis_ind, 1) * basis_values[j / 3] * curr_GP_val_ders(2, 1)) *
//                                 it->getBasis().getDetJ();
//                        }
//                        //if (((i >= 3) && (i <= 5)) && ((j == 2) || (j == 5) || (j == 8))) { // coeffs. of delta p_i
//                        if (((i == 1) || (i == 4) || (i == 7)) && ((j == 2) || (j == 5) || (j == 8))) {
//                            /*v -= 0.5 * gauss_pt_weights[k] * (der_values(i - 3, 1) * basis_values[j / 3]) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (curr_GP_vals[0] * der_values(i - 3, 0) + curr_GP_vals[1] * der_values(i - 3, 1)) *
//                                 (der_values(j / 3, 1)) * it->getBasis().getDetJ();*/
//                            v -= 0.5 * gauss_pt_weights[k] * (der_values(basis_ind, 1) * basis_values[j / 3]) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                 (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                  curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                 (der_values(j / 3, 1)) * it->getBasis().getDetJ();
//                        }
//
//                        // continuity
//                        //if (((i >= 6) && (i <= 8)) && ((j == 0) || (j == 3) || (j == 6))) { // coeffs. of delta u_i
//                        if (((i == 2) || (i == 5) || (i == 8)) && ((j == 0) || (j == 3) || (j == 6))) {
//                            /*v += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values(i - 6, 0) *
//                                                                         (basis_values[j / 3] * curr_GP_val_ders(0, 0) +
//                                                                          curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(j / 3, 1)) +
//                                                                         der_values(i - 6, 1) * basis_values[j / 3] *
//                                                                         curr_GP_val_ders(1, 0)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * (basis_values[i - 6] * der_values(j / 3, 0)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values(basis_ind, 0) *
//                                                                         (basis_values[j / 3] * curr_GP_val_ders(0, 0) +
//                                                                          curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(j / 3, 1)) +
//                                                                         der_values(basis_ind, 1) *
//                                                                         basis_values[j / 3] *
//                                                                         curr_GP_val_ders(1, 0)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] * der_values(j / 3, 0)) *
//                                 it->getBasis().getDetJ();
//                        }
//                        //if (((i >= 6) && (i <= 8)) && ((j == 1) || (j == 4) || (j == 7))) { // coeffs. of delta v_i
//                        if (((i == 2) || (i == 5) || (i == 8)) && ((j == 1) || (j == 4) || (j == 7))) {
//                            /*v += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values(i - 6, 1) *
//                                                                         (basis_values[j / 3] * curr_GP_val_ders(1, 1) +
//                                                                          curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(j / 3, 1)) +
//                                                                         der_values(i - 6, 0) * basis_values[j / 3] *
//                                                                         curr_GP_val_ders(0, 1)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * (basis_values[i - 6] * der_values(j / 3, 1)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values(basis_ind, 1) *
//                                                                         (basis_values[j / 3] * curr_GP_val_ders(1, 1) +
//                                                                          curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                          curr_GP_vals[1] * der_values(j / 3, 1)) +
//                                                                         der_values(basis_ind, 0) *
//                                                                         basis_values[j / 3] *
//                                                                         curr_GP_val_ders(0, 1)) *
//                                 it->getBasis().getDetJ();
//                            v += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] * der_values(j / 3, 1)) *
//                                 it->getBasis().getDetJ();
//                        }
//                        //if (((i >= 6) && (i <= 8)) && ((j == 2) || (j == 5) || (j == 8))) { // coeffs. of delta p_i
//                        if (((i == 2) || (i == 5) || (i == 8)) && ((j == 2) || (j == 5) || (j == 8))) {
//                            /*v += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                 (der_values(i - 6, 0) * der_values(j / 3, 0) +
//                                  der_values(i - 6, 1) * der_values(j / 3, 1)) *
//                                 it->getBasis().getDetJ();*/
//                            v += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                 (der_values(basis_ind, 0) * der_values(j / 3, 0) +
//                                  der_values(basis_ind, 1) * der_values(j / 3, 1)) *
//                                 it->getBasis().getDetJ();
//                        }
//                    }
//                    new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
//                            curr_nodeIDs[2]));
//                    new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1]) : (
//                            curr_nodeIDs[2]));
//                    new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;
//                    new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
//
//                    MatSetValues(mPtr->J, 1, &new_ind_i, 1, &new_ind_j, &v, ADD_VALUES);
//
//                    if (std::isnan((double) v)) {
//                        cout << "v (Jacobian entry) is NaN for elem: " << it->getElemID() << endl;
//                    }
//                }
//            }
//        }
//
//        // ###### residual vector ###### (ver. 1)
//        for (size_t i = 0; i < 9; i++) {
//            /*lid-driven cavity*/
//            /*if (!((((elemCoords[i / 3][0] == mPtr->x_min) || (elemCoords[i / 3][0] == mPtr->x_max) ||
//                    (elemCoords[i / 3][1] == mPtr->y_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_max)) && ((i % 3) < 2)) ||
//                  ((fabs(elemCoords[i / 3][0] - mPtr->x_max) < 1e-5) &&
//                   (fabs(elemCoords[i / 3][1] - mPtr->y_min) < 1e-5) &&
//                   ((i % 3) == 2)))) {*/
//            /*channel flow*/
//            if (!((((elemCoords[i / 3][0] == mPtr->x_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_max)) && ((i % 3) < 2)) ||
//                  ((fabs(elemCoords[i / 3][0] - mPtr->x_max) < 1e-5) &&
//                   ((i % 3) == 2)) ||
//                  ((fabs(sqrt(pow(elemCoords[i / 3][0] - 5, 2.0) + pow(elemCoords[i / 3][1] - 1, 2.0)) - 0.5) < 1e-5) &&
//                   ((i % 3) < 2)))) {
//                v = 0.0;
//
//                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);
//
//                for (size_t k = 0; k < gauss_pts.rows(); k++) {
//                    // basis values at current GP
//                    basis_values = it->getBasis().calcBasis(gauss_pts(k, 0), gauss_pts(k, 1));
//
//                    // (u,v,p) at GP
//                    curr_GP_vals[0] = curr_dof_vals[0] * basis_values[0] + curr_dof_vals[3] * basis_values[1] +
//                                      curr_dof_vals[6] * basis_values[2]; // hard-coded for linear triangles
//                    curr_GP_vals[1] = curr_dof_vals[1] * basis_values[0] + curr_dof_vals[4] * basis_values[1] +
//                                      curr_dof_vals[7] * basis_values[2];
//                    curr_GP_vals[2] = curr_dof_vals[2] * basis_values[0] + curr_dof_vals[5] * basis_values[1] +
//                                      curr_dof_vals[8] * basis_values[2];
//
//                    // SUPG calculation
//                    u_mag = sqrt(pow(curr_GP_vals[0], 2.0) + pow(curr_GP_vals[1], 2.0));
//                    h = 0.0;
//                    for (size_t i_supg = 0; i_supg < 3; i_supg++)
//                        h += curr_GP_vals[0] * der_values(i_supg, 0) + curr_GP_vals[1] * der_values(i_supg, 1);
//
//                    h = (h < 1e-6) ? 0 : (2.0 * fabs(u_mag / h));
//                    Re_u = u_mag * h / (2 * nu);
//                    z = ((Re_u >= 0) && (Re_u <= 3)) ? Re_u / 3.0 : 1.0;
//                    tau_supg = (u_mag < 1e-6) ? 0 : (h * z / (2 * u_mag));
//
//                    // PSPG calculation
//                    h_hash = sqrt(it->getBasis().getDetJ() / (2 * PI));
//                    Re_U = U_mag * h_hash / (2 * nu);
//                    z = ((Re_U >= 0) && (Re_U <= 3)) ? Re_U / 3.0 : 1.0;
//                    tau_pspg = h_hash * z / (2 * U_mag);
//
//                    //if (i < 3) { // x-momentum
//                    if ((i == 0) || (i == 3) || (i == 6)) {
//                    /*v += 0.5 * gauss_pt_weights[k] *
//                         ((basis_values[i] * (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                              curr_GP_vals[1] *
//                                              curr_GP_val_ders(0, 1))) +
//                          (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                       curr_GP_vals[1] *
//                                       curr_GP_val_ders(0, 1)) *
//                           (curr_GP_vals[0] * der_values(i, 0) +
//                            curr_GP_vals[1] *
//                            der_values(i, 1))) - (curr_GP_vals[2] * der_values(i, 0)) +
//                          (tau_supg * (curr_GP_vals[0] * der_values(i, 0) +
//                                       curr_GP_vals[1] *
//                                       der_values(i, 1)) * curr_GP_val_ders(2, 0)) +
//                          ((1.0 / mPtr->Re) *
//                           (der_values(i, 0) * curr_GP_val_ders(0, 0) +
//                            der_values(i, 1) * curr_GP_val_ders(0, 1)))) *
//                         it->getBasis().getDetJ();*/
//                    v += 0.5 * gauss_pt_weights[k] *
//                          ((basis_values[basis_ind] * (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                                       curr_GP_vals[1] *
//                                                       curr_GP_val_ders(0, 1))) +
//                           (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                        curr_GP_vals[1] *
//                                        curr_GP_val_ders(0, 1)) *
//                            (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                             curr_GP_vals[1] *
//                             der_values(basis_ind, 1))) - (curr_GP_vals[2] * der_values(basis_ind, 0)) +
//                           (tau_supg * (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                        curr_GP_vals[1] *
//                                        der_values(basis_ind, 1)) * curr_GP_val_ders(2, 0)) +
//                           ((1.0 / mPtr->Re) *
//                            (der_values(basis_ind, 0) * curr_GP_val_ders(0, 0) +
//                             der_values(basis_ind, 1) * curr_GP_val_ders(0, 1)))) *
//                          it->getBasis().getDetJ();
//
//                    }
//
//                    // y-momentum
//                    //if ((i >= 3) && (i < 6)) {
//                    if ((i == 1) || (i == 4) || (i == 7)) {
//                    /*v += 0.5 * gauss_pt_weights[k] *
//                         ((basis_values[i % 3] * (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                  curr_GP_vals[1] *
//                                                  curr_GP_val_ders(1, 1))) +
//                          (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                       curr_GP_vals[1] *
//                                       curr_GP_val_ders(1, 1)) *
//                           (curr_GP_vals[0] * der_values(i % 3, 0) +
//                            curr_GP_vals[1] *
//                            der_values(i % 3, 1))) - (curr_GP_vals[2] * der_values(i % 3, 1)) +
//                          (tau_supg * (curr_GP_vals[0] * der_values(i % 3, 0) +
//                                       curr_GP_vals[1] *
//                                       der_values(i % 3, 1)) * curr_GP_val_ders(2, 1)) +
//                          ((1.0 / mPtr->Re) *
//                           (der_values(i % 3, 0) * curr_GP_val_ders(1, 0) +
//                            der_values(i % 3, 1) * curr_GP_val_ders(1, 1)))) *
//                         it->getBasis().getDetJ();*/
//                    v += 0.5 * gauss_pt_weights[k] *
//                          ((basis_values[basis_ind] * (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                       curr_GP_vals[1] *
//                                                       curr_GP_val_ders(1, 1))) +
//                           (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                        curr_GP_vals[1] *
//                                        curr_GP_val_ders(1, 1)) *
//                            (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                             curr_GP_vals[1] *
//                             der_values(basis_ind, 1))) - (curr_GP_vals[2] * der_values(basis_ind, 1)) +
//                           (tau_supg * (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                        curr_GP_vals[1] *
//                                        der_values(basis_ind, 1)) * curr_GP_val_ders(2, 1)) +
//                           ((1.0 / mPtr->Re) *
//                            (der_values(basis_ind, 0) * curr_GP_val_ders(1, 0) +
//                             der_values(basis_ind, 1) * curr_GP_val_ders(1, 1)))) *
//                          it->getBasis().getDetJ();
//                    }
//
//                    // continuity
//                    //if ((i >= 6) && (i < 9)) {
//                    if ((i == 2) || (i == 5) || (i == 8)) {
//                    /*v += 0.5 * gauss_pt_weights[k] * (tau_pspg *
//                                                      ((der_values(i % 3, 0) *
//                                                        (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                                         curr_GP_vals[1] *
//                                                         curr_GP_val_ders(0, 1) + curr_GP_val_ders(2, 0))) +
//                                                       (der_values(i % 3, 1) *
//                                                        (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                         curr_GP_vals[1] *
//                                                         curr_GP_val_ders(1, 1) + curr_GP_val_ders(2, 1)))) +
//                                                      basis_values[i % 3] *
//                                                      (curr_GP_val_ders(0, 0) + curr_GP_val_ders(1, 1))) *
//                         it->getBasis().getDetJ();*/
//                    v += 0.5 * gauss_pt_weights[k] * (tau_pspg *
//                                                       ((der_values(basis_ind, 0) *
//                                                         (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                                          curr_GP_vals[1] *
//                                                          curr_GP_val_ders(0, 1) + curr_GP_val_ders(2, 0))) +
//                                                        (der_values(basis_ind, 1) *
//                                                         (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                          curr_GP_vals[1] *
//                                                          curr_GP_val_ders(1, 1) + curr_GP_val_ders(2, 1)))) +
//                                                       basis_values[basis_ind] *
//                                                       (curr_GP_val_ders(0, 0) + curr_GP_val_ders(1, 1))) *
//                          it->getBasis().getDetJ();
//                    }
//                }
//                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
//                        curr_nodeIDs[2]));
//                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;
//
//                if (std::isnan((double) v)) {
//                    cout << "v (residual) is NaN for elem: " << it->getElemID() << endl;
//                }
//
//                mPtr->ierr = VecSetValues(f, 1, &new_ind_i, &v, ADD_VALUES);
//                CHKERRQ(mPtr->ierr);
//            }
//        }

//        //// Jacobian and residual ////
//        for (size_t i = 0; i < 9; i++) { // loop over ndofs * nodes ('i'th equation/row for an element)
//            /*if ((elemCoords[i / 3][0] != mPtr->x_min) && (elemCoords[i / 3][0] != mPtr->x_max) &&
//                (elemCoords[i / 3][1] != mPtr->y_min) &&
//                (elemCoords[i / 3][1] != mPtr->y_max)) { // left, right, bottom, top boundaries*/
//
//            basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);
//
//            new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
//                    curr_nodeIDs[2]));
//            new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;
//
//            /*lid-driven cavity*/
//            /*if (!((((elemCoords[i / 3][0] == mPtr->x_min) || (elemCoords[i / 3][0] == mPtr->x_max) ||
//                    (elemCoords[i / 3][1] == mPtr->y_min) ||
//                    (elemCoords[i / 3][1] == mPtr->y_max)) && ((i % 3) < 2)) ||
//                  ((fabs(elemCoords[i / 3][0] - mPtr->x_max) < 1e-5) &&
//                   (fabs(elemCoords[i / 3][1] - mPtr->y_min) < 1e-5) &&
//                   ((i % 3) ==
//                    2)))) {*/
//            /*channel flow*/
////            if (!((((elemCoords[i / 3][0] == mPtr->x_min) ||
////                    (elemCoords[i / 3][1] == mPtr->y_min) ||
////                    (elemCoords[i / 3][1] == mPtr->y_max)) && ((i % 3) < 2)) ||
////                  ((fabs(elemCoords[i / 3][0] - mPtr->x_max) < 1e-5) &&
////                   ((i % 3) ==
////                    2)) ||
////                  ((fabs(sqrt(pow(elemCoords[i / 3][0] - 5, 2.0) + pow(elemCoords[i / 3][1] - 1, 2.0)) - 0.5) < 1e-5) &&
////                   ((i % 3) < 2)))) {
//            if (it->getElemFlag(i)) {
//                v_r = 0.0;
//                for (size_t k = 0; k < gauss_pts.rows(); k++) {
//                    // basis values at current GP
//                    basis_values = it->getBasis().calcBasis(gauss_pts(k, 0), gauss_pts(k, 1));
//
//                    // (u,v,p) at GP
//                    curr_GP_vals[0] = curr_dof_vals[0] * basis_values[0] + curr_dof_vals[3] * basis_values[1] +
//                                      curr_dof_vals[6] * basis_values[2]; // hard-coded for linear triangles
//                    curr_GP_vals[1] = curr_dof_vals[1] * basis_values[0] + curr_dof_vals[4] * basis_values[1] +
//                                      curr_dof_vals[7] * basis_values[2];
//                    curr_GP_vals[2] = curr_dof_vals[2] * basis_values[0] + curr_dof_vals[5] * basis_values[1] +
//                                      curr_dof_vals[8] * basis_values[2];
//
//                    // SUPG calculation
//                    u_mag = sqrt(pow(curr_GP_vals[0], 2.0) + pow(curr_GP_vals[1], 2.0));
//                    h = 0.0;
//                    for (size_t i_supg = 0; i_supg < 3; i_supg++)
//                        h += curr_GP_vals[0] * der_values(i_supg, 0) + curr_GP_vals[1] * der_values(i_supg, 1);
//
//                    h = (h < 1e-6) ? 0 : (2.0 * fabs(u_mag / h));
//                    Re_u = u_mag * h / (2 * nu);
//                    z = ((Re_u >= 0) && (Re_u <= 3)) ? Re_u / 3.0 : 1.0;
//                    tau_supg = (u_mag < 1e-6) ? 0 : (h * z / (2 * u_mag));
//
//                    // PSPG calculation
//                    h_hash = sqrt(it->getBasis().getDetJ() / (2 * PI));
//                    Re_U = U_mag * h_hash / (2 * nu);
//                    z = ((Re_U >= 0) && (Re_U <= 3)) ? Re_U / 3.0 : 1.0;
//                    tau_pspg = h_hash * z / (2 * U_mag);
//
//                    for (size_t j = 0; j < 9; j++) { // 'j'th column for 'i'th row
//                        v_j = 0.0;
//                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1]) : (
//                                curr_nodeIDs[2]));
//                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
//
//                        // x-momentum
//                        if ((i == 0) || (i == 3) || (i == 6)) {
//                            if ((j == 0) || (j == 3) || (j == 6)) {
//                                v_j += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] *
//                                                                    (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                     curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                                     basis_values[j / 3] * curr_GP_val_ders(0, 0))) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (der_values(basis_ind, 0) * basis_values[j / 3] *
//                                        (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                         curr_GP_vals[1] * curr_GP_val_ders(0, 1)) +
//                                        (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                         curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                        (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                         curr_GP_vals[1] * der_values(j / 3, 1) +
//                                         basis_values[j / 3] *
//                                         curr_GP_val_ders(0, 0))) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (der_values(basis_ind, 0) * basis_values[j / 3] * curr_GP_val_ders(2, 0)) *
//                                       it->getBasis().getDetJ();
//                            } else if ((j == 1) || (j == 4) || (j == 7)) {
//                                v_j += 0.5 * gauss_pt_weights[k] *
//                                       (basis_values[basis_ind] * basis_values[j / 3] * curr_GP_val_ders(0, 1)) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (basis_values[j / 3] * (der_values(basis_ind, 1) *
//                                                               (curr_GP_vals[0] *
//                                                                curr_GP_val_ders(0,
//                                                                                 0) +
//                                                                curr_GP_vals[1] *
//                                                                curr_GP_val_ders(0,
//                                                                                 1)) +
//                                                               curr_GP_val_ders(0, 1) *
//                                                               (curr_GP_vals[0] *
//                                                                der_values(basis_ind, 0) +
//                                                                curr_GP_vals[1] *
//                                                                der_values(basis_ind, 1)))) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                       (der_values(basis_ind, 1) * basis_values[j / 3] * curr_GP_val_ders(2, 0)) *
//                                       it->getBasis().getDetJ();
//                            } else if ((j == 2) || (j == 5) || (j == 8)) {
//                                v_j -= 0.5 * gauss_pt_weights[k] * (der_values(basis_ind, 0) * basis_values[j / 3]) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                        curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                       (der_values(j / 3, 0)) * it->getBasis().getDetJ();
//                            }
//                        }
//
//                        // y-momentum
//                        if ((i == 1) || (i == 4) || (i == 7)) {
//                            if ((j == 0) || (j == 3) || (j == 6)) {
//                                v_j += 0.5 * gauss_pt_weights[k] *
//                                       (basis_values[basis_ind] * basis_values[j / 3] * curr_GP_val_ders(1, 0)) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (basis_values[j / 3] * (der_values(basis_ind, 0) *
//                                                               (curr_GP_vals[0] *
//                                                                curr_GP_val_ders(1,
//                                                                                 0) +
//                                                                curr_GP_vals[1] *
//                                                                curr_GP_val_ders(1,
//                                                                                 1)) +
//                                                               curr_GP_val_ders(1, 0) *
//                                                               (curr_GP_vals[0] *
//                                                                der_values(basis_ind, 0) +
//                                                                curr_GP_vals[1] *
//                                                                der_values(basis_ind,
//                                                                           1)))) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                       (der_values(basis_ind, 0) * basis_values[j / 3] * curr_GP_val_ders(2, 1)) *
//                                       it->getBasis().getDetJ();
//                            } else if ((j == 1) || (j == 4) || (j == 7)) {
//                                v_j += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] *
//                                                                    (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                     curr_GP_vals[1] * der_values(j / 3, 1) +
//                                                                     basis_values[j / 3] * curr_GP_val_ders(1, 1))) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (der_values(basis_ind, 1) * basis_values[j / 3] *
//                                        (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                         curr_GP_vals[1] * curr_GP_val_ders(1, 1)) +
//                                        (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                         curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                        (curr_GP_vals[0] * der_values(j / 3, 0) +
//                                         curr_GP_vals[1] * der_values(j / 3, 1) +
//                                         basis_values[j / 3] *
//                                         curr_GP_val_ders(1, 1))) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (der_values(basis_ind, 1) * basis_values[j / 3] * curr_GP_val_ders(2, 1)) *
//                                       it->getBasis().getDetJ();
//                            } else if ((j == 2) || (j == 5) || (j == 8)) {
//                                v_j -= 0.5 * gauss_pt_weights[k] * (der_values(basis_ind, 1) * basis_values[j / 3]) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_supg *
//                                       (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                        curr_GP_vals[1] * der_values(basis_ind, 1)) *
//                                       (der_values(j / 3, 1)) * it->getBasis().getDetJ();
//                            }
//                        }
//
//                        // continuity
//                        if ((i == 2) || (i == 5) || (i == 8)) {
//                            if ((j == 0) || (j == 3) || (j == 6)) {
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values(basis_ind, 0) *
//                                                                               (basis_values[j / 3] *
//                                                                                curr_GP_val_ders(0, 0) +
//                                                                                curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                                curr_GP_vals[1] *
//                                                                                der_values(j / 3, 1)) +
//                                                                               der_values(basis_ind, 1) *
//                                                                               basis_values[j / 3] *
//                                                                               curr_GP_val_ders(1, 0)) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] * der_values(j / 3, 0)) *
//                                       it->getBasis().getDetJ();
//                            } else if ((j == 1) || (j == 4) || (j == 7)) {
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values(basis_ind, 1) *
//                                                                               (basis_values[j / 3] *
//                                                                                curr_GP_val_ders(1, 1) +
//                                                                                curr_GP_vals[0] * der_values(j / 3, 0) +
//                                                                                curr_GP_vals[1] *
//                                                                                der_values(j / 3, 1)) +
//                                                                               der_values(basis_ind, 0) *
//                                                                               basis_values[j / 3] *
//                                                                               curr_GP_val_ders(0, 1)) *
//                                       it->getBasis().getDetJ();
//                                v_j += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] * der_values(j / 3, 1)) *
//                                       it->getBasis().getDetJ();
//                            } else if ((j == 2) || (j == 5) || (j == 8)) {
//                                v_j += 0.5 * gauss_pt_weights[k] * tau_pspg *
//                                       (der_values(basis_ind, 0) * der_values(j / 3, 0) +
//                                        der_values(basis_ind, 1) * der_values(j / 3, 1)) *
//                                       it->getBasis().getDetJ();
//                            }
//                        }
//
//                        MatSetValues(mPtr->J, 1, &new_ind_i, 1, &new_ind_j, &v_j, ADD_VALUES);
//
//                        if (std::isnan(v_j)) {
//                            cout << "v (Jacobian entry) is NaN for elem: " << it->getElemID() << endl;
//                        }
//                    }
//
//                    // x-momentum
//                    if ((i == 0) || (i == 3) || (i == 6)) {
//                        v_r += 0.5 * gauss_pt_weights[k] *
//                               ((basis_values[basis_ind] * (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                                            curr_GP_vals[1] *
//                                                            curr_GP_val_ders(0, 1))) +
//                                (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                             curr_GP_vals[1] *
//                                             curr_GP_val_ders(0, 1)) *
//                                 (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                  curr_GP_vals[1] *
//                                  der_values(basis_ind, 1))) - (curr_GP_vals[2] * der_values(basis_ind, 0)) +
//                                (tau_supg * (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                             curr_GP_vals[1] *
//                                             der_values(basis_ind, 1)) * curr_GP_val_ders(2, 0)) +
//                                ((1.0 / mPtr->Re) *
//                                 (der_values(basis_ind, 0) * curr_GP_val_ders(0, 0) +
//                                  der_values(basis_ind, 1) * curr_GP_val_ders(0, 1)))) *
//                               it->getBasis().getDetJ();
//                    }
//
//                    // y-momentum
//                    if ((i == 1) || (i == 4) || (i == 7)) {
//                        v_r += 0.5 * gauss_pt_weights[k] *
//                               ((basis_values[basis_ind] * (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                            curr_GP_vals[1] *
//                                                            curr_GP_val_ders(1, 1))) +
//                                (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                             curr_GP_vals[1] *
//                                             curr_GP_val_ders(1, 1)) *
//                                 (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                  curr_GP_vals[1] *
//                                  der_values(basis_ind, 1))) - (curr_GP_vals[2] * der_values(basis_ind, 1)) +
//                                (tau_supg * (curr_GP_vals[0] * der_values(basis_ind, 0) +
//                                             curr_GP_vals[1] *
//                                             der_values(basis_ind, 1)) * curr_GP_val_ders(2, 1)) +
//                                ((1.0 / mPtr->Re) *
//                                 (der_values(basis_ind, 0) * curr_GP_val_ders(1, 0) +
//                                  der_values(basis_ind, 1) * curr_GP_val_ders(1, 1)))) *
//                               it->getBasis().getDetJ();
//                    }
//
//                    // continuity
//                    if ((i == 2) || (i == 5) || (i == 8)) {
//                        v_r += 0.5 * gauss_pt_weights[k] * (tau_pspg *
//                                                            ((der_values(basis_ind, 0) *
//                                                              (curr_GP_vals[0] * curr_GP_val_ders(0, 0) +
//                                                               curr_GP_vals[1] *
//                                                               curr_GP_val_ders(0, 1) + curr_GP_val_ders(2, 0))) +
//                                                             (der_values(basis_ind, 1) *
//                                                              (curr_GP_vals[0] * curr_GP_val_ders(1, 0) +
//                                                               curr_GP_vals[1] *
//                                                               curr_GP_val_ders(1, 1) + curr_GP_val_ders(2, 1)))) +
//                                                            basis_values[basis_ind] *
//                                                            (curr_GP_val_ders(0, 0) + curr_GP_val_ders(1, 1))) *
//                               it->getBasis().getDetJ();
//                    }
//                }
//
//                mPtr->ierr = VecSetValues(f, 1, &new_ind_i, &v_r, ADD_VALUES);
//                CHKERRQ(mPtr->ierr);
//
//                if (std::isnan(v_r)) {
//                    cout << "v (residual) is NaN for elem: " << it->getElemID() << endl;
//                }
//            }
//        }

        // PSPG calculation
        h_hash = sqrt(2 * it->getBasis().getDetJ() / PI);
        Re_U = U_mag * h_hash * mPtr->Re / 2;
        z = ((Re_U >= 0) && (Re_U <= 3)) ? (Re_U / 3.0) : 1.0;
        tau_pspg = h_hash * z / (2 * U_mag);

        //// Jacobian and residual ////
        for (size_t k = 0; k < gauss_pts.size(); k++) {
            // basis values at current GP
            basis_values = it->getBasis().calcBasis(gauss_pts[k][0], gauss_pts[k][1]);

            // (u,v,p) at GP
            curr_GP_vals[0] = curr_dof_vals[0] * basis_values[0] + curr_dof_vals[3] * basis_values[1] +
                              curr_dof_vals[6] * basis_values[2]; // hard-coded for linear triangles
            curr_GP_vals[1] = curr_dof_vals[1] * basis_values[0] + curr_dof_vals[4] * basis_values[1] +
                              curr_dof_vals[7] * basis_values[2];
            curr_GP_vals[2] = curr_dof_vals[2] * basis_values[0] + curr_dof_vals[5] * basis_values[1] +
                              curr_dof_vals[8] * basis_values[2];

            // SUPG calculation
            u_mag = sqrt(pow(curr_GP_vals[0], 2.0) + pow(curr_GP_vals[1], 2.0));
            h = 0.0;
            for (size_t i_supg = 0; i_supg < 3; i_supg++)
                h += curr_GP_vals[0] * der_values[i_supg][0] + curr_GP_vals[1] * der_values[i_supg][1];

            h = (h < 1e-6) ? 0 : (2.0 * u_mag/ fabs(h));
            Re_u = u_mag * h * mPtr->Re / 2;
            z = ((Re_u >= 0) && (Re_u <= 3)) ? (Re_u / 3.0) : 1.0;
            tau_supg = (u_mag < 1e-6) ? 0 : (h * z / (2 * u_mag));

            // x-momentum
            A_block = {};
            for (size_t i = 0; i < 7; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 0; j < 7; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] *
                                                                              (curr_GP_vals[0] *
                                                                               der_values[j / 3][0] +
                                                                               curr_GP_vals[1] *
                                                                               der_values[j / 3][1] +
                                                                               basis_values[j / 3] *
                                                                               curr_GP_val_ders[0][0])) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (der_values[basis_ind][0] * basis_values[j / 3] *
                                                  (curr_GP_vals[0] * curr_GP_val_ders[0][0] +
                                                   curr_GP_vals[1] * curr_GP_val_ders[0][1]) +
                                                  (curr_GP_vals[0] * der_values[basis_ind][0] +
                                                   curr_GP_vals[1] * der_values[basis_ind][1]) *
                                                  (curr_GP_vals[0] * der_values[j / 3][0] +
                                                   curr_GP_vals[1] * der_values[j / 3][1] +
                                                   basis_values[j / 3] *
                                                   curr_GP_val_ders[0][0])) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (der_values[basis_ind][0] * basis_values[j / 3] *
                                                  curr_GP_val_ders[2][0]) *
                                                 it->getBasis().getDetJ();
                    }
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);

            A_block = {};
            for (size_t i = 0; i < 7; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 1; j < 8; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] *
                                                 (basis_values[basis_ind] * basis_values[j / 3] *
                                                  curr_GP_val_ders[0][1]) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (basis_values[j / 3] * (der_values[basis_ind][1] *
                                                                         (curr_GP_vals[0] *
                                                                          curr_GP_val_ders[0]
                                                                                           [0] +
                                                                          curr_GP_vals[1] *
                                                                          curr_GP_val_ders[0]
                                                                                           [1]) +
                                                                         curr_GP_val_ders[0][1] *
                                                                         (curr_GP_vals[0] *
                                                                          der_values[basis_ind][0] +
                                                                          curr_GP_vals[1] *
                                                                          der_values[basis_ind][1]))) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_pspg *
                                                 (der_values[basis_ind][1] * basis_values[j / 3] *
                                                  curr_GP_val_ders[2][0]) *
                                                 it->getBasis().getDetJ();
                    }
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);

            A_block = {};
            res_block = {};
            for (size_t i = 0; i < 7; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 2; j < 9; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] +=
                                0.5 * gauss_pt_weights[k] * (-der_values[basis_ind][0] * basis_values[j / 3]) *
                                it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (curr_GP_vals[0] * der_values[basis_ind][0] +
                                                  curr_GP_vals[1] * der_values[basis_ind][1]) *
                                                 (der_values[j / 3][0]) * it->getBasis().getDetJ();
                    }
                }

                if (it->getElemFlag(i)) {
                    res_block[i / 3] += 0.5 * gauss_pt_weights[k] *
                                        ((basis_values[basis_ind] * (curr_GP_vals[0] * curr_GP_val_ders[0][0] +
                                                                     curr_GP_vals[1] *
                                                                     curr_GP_val_ders[0][1])) +
                                         (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders[0][0] +
                                                      curr_GP_vals[1] *
                                                      curr_GP_val_ders[0][1]) *
                                          (curr_GP_vals[0] * der_values[basis_ind][0] +
                                           curr_GP_vals[1] *
                                           der_values[basis_ind][1])) - (curr_GP_vals[2] * der_values[basis_ind][0]) +
                                         (tau_supg * (curr_GP_vals[0] * der_values[basis_ind][0] +
                                                      curr_GP_vals[1] *
                                                      der_values[basis_ind][1]) * curr_GP_val_ders[2][0]) +
                                         ((1.0 / mPtr->Re) *
                                          (der_values[basis_ind][0] * curr_GP_val_ders[0][0] +
                                           der_values[basis_ind][1] * curr_GP_val_ders[0][1]))) *
                                        it->getBasis().getDetJ();
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);
            VecSetValues(f, 3, &A_block_row_ind[0], &res_block[0], ADD_VALUES);

            // y-momentum
            A_block = {};
            for (size_t i = 1; i < 8; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 0; j < 7; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] *
                                                 (basis_values[basis_ind] * basis_values[j / 3] *
                                                  curr_GP_val_ders[1][0]) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (basis_values[j / 3] * (der_values[basis_ind][0] *
                                                                         (curr_GP_vals[0] *
                                                                          curr_GP_val_ders[1]
                                                                                           [0] +
                                                                          curr_GP_vals[1] *
                                                                          curr_GP_val_ders[1]
                                                                                           [1]) +
                                                                         curr_GP_val_ders[1][0] *
                                                                         (curr_GP_vals[0] *
                                                                          der_values[basis_ind][0] +
                                                                          curr_GP_vals[1] *
                                                                          der_values[basis_ind]
                                                                                     [1]))) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_pspg *
                                                 (der_values[basis_ind][0] * basis_values[j / 3] *
                                                  curr_GP_val_ders[2][1]) *
                                                 it->getBasis().getDetJ();
                    }
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);

            A_block = {};
            for (size_t i = 1; i < 8; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 1; j < 8; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] *
                                                                              (curr_GP_vals[0] * der_values[j / 3][0] +
                                                                               curr_GP_vals[1] * der_values[j / 3][1] +
                                                                               basis_values[j / 3] *
                                                                               curr_GP_val_ders[1][1])) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (der_values[basis_ind][1] * basis_values[j / 3] *
                                                  (curr_GP_vals[0] * curr_GP_val_ders[1][0] +
                                                   curr_GP_vals[1] * curr_GP_val_ders[1][1]) +
                                                  (curr_GP_vals[0] * der_values[basis_ind][0] +
                                                   curr_GP_vals[1] * der_values[basis_ind][1]) *
                                                  (curr_GP_vals[0] * der_values[j / 3][0] +
                                                   curr_GP_vals[1] * der_values[j / 3][1] +
                                                   basis_values[j / 3] *
                                                   curr_GP_val_ders[1][1])) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (der_values[basis_ind][1] * basis_values[j / 3] *
                                                  curr_GP_val_ders[2][1]) *
                                                 it->getBasis().getDetJ();
                    }
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);

            A_block = {};
            res_block = {};
            for (size_t i = 1; i < 8; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 2; j < 9; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] +=
                                0.5 * gauss_pt_weights[k] * (-der_values[basis_ind][1] * basis_values[j / 3]) *
                                it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_supg *
                                                 (curr_GP_vals[0] * der_values[basis_ind][0] +
                                                  curr_GP_vals[1] * der_values[basis_ind][1]) *
                                                 (der_values[j / 3][1]) * it->getBasis().getDetJ();
                    }
                }

                if (it->getElemFlag(i)) {
                    res_block[i / 3] += 0.5 * gauss_pt_weights[k] *
                                        ((basis_values[basis_ind] * (curr_GP_vals[0] * curr_GP_val_ders[1][0] +
                                                                     curr_GP_vals[1] *
                                                                     curr_GP_val_ders[1][1])) +
                                         (tau_supg * (curr_GP_vals[0] * curr_GP_val_ders[1][0] +
                                                      curr_GP_vals[1] *
                                                      curr_GP_val_ders[1][1]) *
                                          (curr_GP_vals[0] * der_values[basis_ind][0] +
                                           curr_GP_vals[1] *
                                           der_values[basis_ind][1])) - (curr_GP_vals[2] * der_values[basis_ind][1]) +
                                         (tau_supg * (curr_GP_vals[0] * der_values[basis_ind][0] +
                                                      curr_GP_vals[1] *
                                                      der_values[basis_ind][1]) * curr_GP_val_ders[2][1]) +
                                         ((1.0 / mPtr->Re) *
                                          (der_values[basis_ind][0] * curr_GP_val_ders[1][0] +
                                           der_values[basis_ind][1] * curr_GP_val_ders[1][1]))) *
                                        it->getBasis().getDetJ();
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);
            VecSetValues(f, 3, &A_block_row_ind[0], &res_block[0], ADD_VALUES);

            // continuity
            A_block = {};
            for (size_t i = 2; i < 9; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 0; j < 7; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values[basis_ind][0] *
                                                                                         (basis_values[j / 3] *
                                                                                          curr_GP_val_ders[0][0] +
                                                                                          curr_GP_vals[0] *
                                                                                          der_values[j / 3][0] +
                                                                                          curr_GP_vals[1] *
                                                                                          der_values[j / 3][1]) +
                                                                                         der_values[basis_ind][1] *
                                                                                         basis_values[j / 3] *
                                                                                         curr_GP_val_ders[1][0]) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] +=
                                0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] * der_values[j / 3][0]) *
                                it->getBasis().getDetJ();
                    }
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);

            A_block = {};
            for (size_t i = 2; i < 9; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 1; j < 8; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_pspg * (der_values[basis_ind][1] *
                                                                                         (basis_values[j / 3] *
                                                                                          curr_GP_val_ders[1][1] +
                                                                                          curr_GP_vals[0] *
                                                                                          der_values[j / 3][0] +
                                                                                          curr_GP_vals[1] *
                                                                                          der_values[j / 3][1]) +
                                                                                         der_values[basis_ind][0] *
                                                                                         basis_values[j / 3] *
                                                                                         curr_GP_val_ders[0][1]) *
                                                 it->getBasis().getDetJ();
                        A_block[i / 3][j / 3] +=
                                0.5 * gauss_pt_weights[k] * (basis_values[basis_ind] * der_values[j / 3][1]) *
                                it->getBasis().getDetJ();
                    }
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);

            A_block = {};
            res_block = {};
            for (size_t i = 2; i < 9; i += 3) {

                basis_ind = (i < 3) ? 0 : (((i >= 3) && (i < 6)) ? 1 : 2);

                new_nodeID_i = (i < 3) ? (curr_nodeIDs[0]) : (((i >= 3) && (i < 6)) ? (curr_nodeIDs[1]) : (
                        curr_nodeIDs[2]));
                new_ind_i = 3 * new_nodeID_i + (PetscInt) i % 3;

                A_block_row_ind[i / 3] = new_ind_i;

                for (size_t j = 2; j < 9; j += 3) {
                    if (it->getElemFlag(i)) {
                        new_nodeID_j = (j < 3) ? (curr_nodeIDs[0]) : (((j >= 3) && (j < 6)) ? (curr_nodeIDs[1])
                                                                                            : (
                                                                              curr_nodeIDs[2]));
                        new_ind_j = 3 * new_nodeID_j + (PetscInt) j % 3;
                        A_block_col_ind[j / 3] = new_ind_j;

                        A_block[i / 3][j / 3] += 0.5 * gauss_pt_weights[k] * tau_pspg *
                                                 (der_values[basis_ind][0] * der_values[j / 3][0] +
                                                  der_values[basis_ind][1] * der_values[j / 3][1]) *
                                                 it->getBasis().getDetJ();
                    }
                }

                if (it->getElemFlag(i)) {
                    res_block[i / 3] += 0.5 * gauss_pt_weights[k] * (tau_pspg *
                                                                     ((der_values[basis_ind][0] *
                                                                       (curr_GP_vals[0] * curr_GP_val_ders[0][0] +
                                                                        curr_GP_vals[1] *
                                                                        curr_GP_val_ders[0][1] +
                                                                        curr_GP_val_ders[2][0])) +
                                                                      (der_values[basis_ind][1] *
                                                                       (curr_GP_vals[0] * curr_GP_val_ders[1][0] +
                                                                        curr_GP_vals[1] *
                                                                        curr_GP_val_ders[1][1] +
                                                                        curr_GP_val_ders[2][1]))) +
                                                                     basis_values[basis_ind] *
                                                                     (curr_GP_val_ders[0][0] +
                                                                      curr_GP_val_ders[1][1])) *
                                        it->getBasis().getDetJ();
                }
            }
            MatSetValues(mPtr->J, 3, &A_block_row_ind[0], 3, &A_block_col_ind[0], &A_block[0][0], ADD_VALUES);
            VecSetValues(f, 3, &A_block_row_ind[0], &res_block[0], ADD_VALUES);
        }
    }
    MatAssemblyBegin(mPtr->J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mPtr->J, MAT_FINAL_ASSEMBLY);
    //MatView(mPtr->J, PETSC_VIEWER_STDOUT_SELF);
    //VecView(f, PETSC_VIEWER_STDOUT_SELF);
}

// Jacobian
PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *meshPtr) {
    return 0;
}
