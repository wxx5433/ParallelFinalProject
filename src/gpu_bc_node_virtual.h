#ifndef __GPU_BC_NODE_VIRTUAL_H__
#define __GPU_BC_NODE_VIRTUAL_H__

#include "graph_virtual.h"
#include "CycleTimer.h"

#define NOT_VISITED_MARKER -1

int gpu_bc_node_virtual (const graph_virtual *g, float *bc);
int bc_virtual (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, float *h_bc);

#endif
