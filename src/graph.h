#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

#define GRAPH_HEADER_TOKEN 0xDEADBEEF
#define MAX_SPAN 4

struct graph
{
  int num_edges;
  int num_nodes;
  int num_vnodes;

  int *vmap;
  int *offset;
  int *nvir;
  int *voutgoing_starts;
  int *outgoing_starts;
  int *outgoing_edges;
};

void print_graph(const graph *graph);
void load_graph(const char *filename, graph *graph);
void read_graph_file(std::ifstream& file, int* scratch);

#endif
