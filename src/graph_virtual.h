#ifndef __GRAPH_VIRTUAL_H__
#define __GRAPH_VIRTUAL_H__

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

#define GRAPH_HEADER_TOKEN 0xDEADBEEF
#define MAX_SPAN 4

struct graph_virtual
{
  int num_edges;
  int num_nodes;
  int num_virtual_nodes;

  int *vmap;
  int *offset;
  int *nvir;
  int *outgoing_starts;
  int *outgoing_edges;
};

void print_graph_virtual(const graph_virtual *graph);
void load_graph_virtual(const char *filename, graph_virtual *graph);

#endif
