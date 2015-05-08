#ifndef SEQUENTIAL_BC_H
#define SEQUENTIAL_BC_H

#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <iostream>
#include "CycleTimer.h"
#include "graph.h"


std::vector<float> compute_bc(graph *g);
int forward_propagation(graph *g, int src_node, std::vector<int> &d, 
    std::vector<int> &sigma);
void print_solution(const float *bc, int num_nodes);
std::vector<float> compute_bc_openmp(graph *g);
void bc_cpu (int* xadj, int* adj, int nVtx, int nz, float* bc);
void bc_cpu_openmp (int* xadj, int* adj, int nVtx, int nz, float* bc);

#endif  /* SEQUENTIAL_BC_H */
