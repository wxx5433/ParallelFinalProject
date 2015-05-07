#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

#include "graph.h"

#define GRAPH_HEADER_TOKEN 0xDEADBEEF
#define MAX_SPAN 4

void build_start(graph* graph, int* scratch)
{
  int num_nodes = graph->num_nodes;
  graph->outgoing_starts = (int*)malloc(sizeof(int) * (num_nodes + 1));
  for(int i = 0; i < num_nodes; i++)
  {
    graph->outgoing_starts[i] = scratch[i];
  }
  graph->outgoing_starts[num_nodes] = graph->num_edges;
}

void build_edges(graph* graph, int* scratch) {
  int num_nodes = graph->num_nodes;

  graph->outgoing_edges = (int*)malloc(sizeof(int) * graph->num_edges);
  for(int i = 0; i < graph->num_edges; i++)
  {
    graph->outgoing_edges[i] = scratch[num_nodes + i];
  }
}

void get_meta_data(std::ifstream& file, graph* graph) {
  // going back to the beginning of the file
  file.clear();
  file.seekg(0, std::ios::beg);
  std::string buffer;
  std::getline(file, buffer);
  if ((buffer.compare(std::string("AdjacencyGraph"))))
  {
    std::cout << "Invalid input file" << buffer << std::endl;
    exit(1);
  }
  buffer.clear();
  std::getline(file, buffer);
  graph->num_nodes = atoi(buffer.c_str());
  buffer.clear();
  std::getline(file, buffer);
  graph->num_edges = atoi(buffer.c_str());

}

void read_graph_file(std::ifstream& file, int* scratch)
{
  std::string buffer;
  int idx = 0;
  while(!file.eof())
  {
    buffer.clear();
    std::getline(file, buffer);
    std::stringstream parse(buffer);
    int v;
    parse >> v;
    if (parse.fail())
    {
      break;
    }
    scratch[idx] = v;
    idx++;
  }
}

void print_graph(const graph* graph) {

    printf("Graph pretty print:\n");
    printf("num_nodes=%d\n", graph->num_nodes);
    printf("num_edges=%d\n", graph->num_edges);

    for (int i=0; i<graph->num_nodes; i++) {

        int start_edge = graph->outgoing_starts[i];
        int end_edge = graph->outgoing_starts[i + 1];
        printf("node %02d: out=%d: ", i, end_edge - start_edge);
        for (int j=start_edge; j<end_edge; j++) {
            int target = graph->outgoing_edges[j];
            printf("%d ", target);
        }
        printf("\n");
    }
}

void load_graph(const char* filename, graph* graph)
{
  // open the file
  std::ifstream graph_file;
  graph_file.open(filename);
  get_meta_data(graph_file, graph);

  int* scratch = (int*) malloc(sizeof(int) * (graph->num_nodes + graph->num_edges));
  read_graph_file(graph_file, scratch);

  build_start(graph, scratch);
  build_edges(graph, scratch);
  free(scratch);

  //build_incoming_edges(graph);

  //print_graph(graph);
}

void load_graph_binary(const char* filename, graph* graph) {

    FILE* input = fopen(filename, "rb");

    if (!input) {
        fprintf(stderr, "Could not open: %s\n", filename);
        exit(1);
    }

    int header[3];

    if (fread(header, sizeof(int), 3, input) != 3) {
        fprintf(stderr, "Error reading header.\n");
        exit(1);
    }

    if (header[0] != GRAPH_HEADER_TOKEN) {
        fprintf(stderr, "Invalid graph file header. File may be corrupt.\n");
        exit(1);
    }

    graph->num_nodes = header[1];
    graph->num_edges = header[2];

    graph->outgoing_starts = (int*)malloc(sizeof(int) * graph->num_nodes);
    graph->outgoing_edges = (int*)malloc(sizeof(int) * graph->num_edges);

    if (fread(graph->outgoing_starts, sizeof(int), graph->num_nodes, input) != graph->num_nodes) {
        fprintf(stderr, "Error reading nodes.\n");
        exit(1);
    }

    if (fread(graph->outgoing_edges, sizeof(int), graph->num_edges, input) != graph->num_edges) {
        fprintf(stderr, "Error reading edges.\n");
        exit(1);
    }

    fclose(input);

    //build_incoming_edges(graph);
    //print_graph(graph);
}

void store_graph_binary(const char* filename, graph* graph) {

    FILE* output = fopen(filename, "wb");

    if (!output) {
        fprintf(stderr, "Could not open: %s\n", filename);
        exit(1);
    }

    int header[3];
    header[0] = GRAPH_HEADER_TOKEN;
    header[1] = graph->num_nodes;
    header[2] = graph->num_edges;

    if (fwrite(header, sizeof(int), 3, output) != 3) {
        fprintf(stderr, "Error writing header.\n");
        exit(1);
    }

    if (fwrite(graph->outgoing_starts, sizeof(int), graph->num_nodes, output) != graph->num_nodes) {
        fprintf(stderr, "Error writing nodes.\n");
        exit(1);
    }

    if (fwrite(graph->outgoing_edges, sizeof(int), graph->num_edges, output) != graph->num_edges) {
        fprintf(stderr, "Error writing edges.\n");
        exit(1);
    }

    fclose(output);
}
