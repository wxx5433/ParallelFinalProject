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

  int *outgoing_starts;
  int *outgoing_edges;
};

struct graph_virtual {
  int num_edges;
  int num_nodes;
  int num_vnodes;

  int *outgoing_starts;
  int *outgoing_edges;
  int *vmap;
  int *offset;
  int *nvir;
  int *voutgoing_starts;
};

void print_graph(const graph *graph);
void load_graph(const char *filename, graph *graph);
void read_graph_file(std::ifstream& file, int* scratch);
void build_virtual_graph(const int *outgoing_starts, const int *outgoing_edges, int num_edges, int num_nodes, graph_virtual *g_v);
void print_graph_virtual(const graph_virtual *graph);

#endif
