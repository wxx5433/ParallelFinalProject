#ifndef __GPU_BC_NODE_VIRTUAL_H__
#define __GPU_BC_NODE_VIRTUAL_H__

#include <iostream>
#include <stdio.h>

#include "graph.h"
#include "CycleTimer.h"

#define THREAD_NUM 256

#define NOT_VISITED_MARKER -1

int gpu_bc_node_virtual (const graph *g, float *bc);
int bc_virtual (const graph *g, float *bc);
int bc_virtual_stride (const graph *g, float *bc);

#endif
