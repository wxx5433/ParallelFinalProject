#ifndef SEQUENTIAL_BC_H
#define SEQUENTIAL_BC_H

#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <iostream>
#include <omp.h>
#include "CycleTimer.h"
#include "graph.h"

//struct graph;

std::vector<double> compute_bc(graph *g);
//void compute_bc_openmp(graph *g);
int forward_propagation(graph *g, int src_node, std::vector<int> &d, 
    std::vector<long> &sigma);
void print_solution(const std::vector<double> &bc, int num_nodes);
std::vector<double> compute_bc_openmp(graph *g);

#endif  /* SEQUENTIAL_BC_H */
