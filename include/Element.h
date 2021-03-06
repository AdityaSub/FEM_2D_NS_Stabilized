#pragma once

#include "Node.h"
#include"Basis.h"
#include<array>
#include<vector>

class Element {
public:
    Element(const Node &, const Node &, const Node &, const int &); // constructor
    const Node &getNode1() const; // return 'node1'
    const Node &getNode2() const; // return 'node2'
    const Node &getNode3() const; // return 'node3'
    const int &getElemID() const; // return element ID
    Basis &getBasis(); // return basis object for current element
    void calcElementStiffness(); // calculate 3 x 3 element stiffness matrix
    const std::array<std::array<double, 9>, 9> getElemStiffness() const; // return element stiffness matrix
    void printElemStiffness() const; // print element stiffness matrix
    bool getElemFlag(size_t); // get an element assemble flag
    void setElemFlag(size_t,bool); // set an element assemble flag
    ~Element(); // destructor

private:
    Node node1; // (0, 0)
    Node node2; // (1, 0)
    Node node3; // (0, 1)
    int elemID; // element ID of current element
    std::array<std::array<double, 9>, 9> elemStiffness; // element stiffness matrix
    Basis basis; // basis function calculations
    std::array<bool,9> assembleFlag; // flag set at runtime to decide assembly of the 'i'th element equation
};
