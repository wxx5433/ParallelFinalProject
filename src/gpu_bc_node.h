#ifndef __GPU_BC_NODE_H
#define __GPU_BC_NODE_H

#include "graph.h"
#include "CycleTimer.h"

#define NOT_VISITED_MARKER -1

int gpu_bc_node (const graph *g, float *bc);

#endif
