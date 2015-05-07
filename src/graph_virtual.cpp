#include "graph_virtual.h"
#include "graph.h"

void build_start_virtual(graph_virtual *graph, int *scratch) {
  int num_nodes = graph->num_nodes;
  int num_edges = graph->num_edges;
  int num;
  std::vector<int> vmap;
  std::vector<int> offset;
  std::vector<int> nvir;
  std::vector<int> outgoing_starts;

  vmap.push_back(0);
  offset.push_back(0);
  outgoing_starts.push_back(scratch[0]);
  for (int i = 1; i < num_nodes; ++i) {
    num = (scratch[i] - scratch[i - 1] - 1) / MAX_SPAN;
    nvir.push_back(num + 1);
    for (int j = 1; j <= num; ++j) {
      vmap.push_back(i - 1);
      offset.push_back(j);
      outgoing_starts.push_back(scratch[i - 1] + j * MAX_SPAN);
    }
    offset.push_back(0);
    vmap.push_back(i);
    outgoing_starts.push_back(scratch[i]);
  }
  num = (num_edges - scratch[num_nodes - 1] - 1) / MAX_SPAN;
  nvir.push_back(num + 1);

  int num_virtual_nodes = vmap.size();
  graph->num_virtual_nodes = num_virtual_nodes;
  offset[num_virtual_nodes] = 0;
  vmap[num_virtual_nodes] = num_nodes;
  outgoing_starts[num_virtual_nodes] = num_nodes;

  graph->vmap = (int*)malloc(sizeof(int) * (num_virtual_nodes + 1));
  graph->offset = (int*)malloc(sizeof(int) * (num_virtual_nodes + 1));
  graph->nvir = (int*)malloc(sizeof(int) * (num_nodes + 1));
  graph->outgoing_starts = (int*)malloc(sizeof(int) * (num_virtual_nodes + 1));
  for (int i = 0; i < num_virtual_nodes + 1; ++i) {
    graph->vmap[i] = vmap[i];
    graph->offset[i] = offset[i];
    graph->nvir[i] = nvir[i];
    graph->outgoing_starts[i] = outgoing_starts[i];
  }
}

void build_edges_virtual(graph_virtual *graph, int *scratch) {
  int num_nodes = graph->num_nodes;

  graph->outgoing_edges = (int*)malloc(sizeof(int) * graph->num_edges);
  for(int i = 0; i < graph->num_edges; i++)
  {
    graph->outgoing_edges[i] = scratch[num_nodes + i];
  }
}

void get_meta_data_virtual(std::ifstream& file, graph_virtual* graph) {
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

void load_graph_virtual(const char *filename, graph_virtual *graph) {
  // open the file
  std::ifstream graph_file;
  graph_file.open(filename);
  get_meta_data_virtual(graph_file, graph);

  int* scratch = (int*) malloc(sizeof(int) * (graph->num_nodes + graph->num_edges));
  read_graph_file(graph_file, scratch);

  build_start_virtual(graph, scratch);
  build_edges_virtual(graph, scratch);

  free(scratch);

  //print_graph_virtual(graph);
}

void print_graph_virtual(const graph_virtual *graph) {

    printf("Graph pretty print:\n");
    printf("num_nodes=%d\n", graph->num_nodes);
    printf("num_edges=%d\n", graph->num_edges);
    printf("num_virtual_nodes=%d\n", graph->num_virtual_nodes);

    for (int i=0; i<graph->num_virtual_nodes; i++) {

        int start_edge = graph->outgoing_starts[i];
        int end_edge = (i == graph->num_virtual_nodes-1) ? graph->num_edges : graph->outgoing_starts[i+1];
        printf("virtual node: %d, node %d, offset %d, out=%d: ", i, graph->vmap[i], graph->offset[i], end_edge - start_edge);
        for (int j=start_edge; j<end_edge; j++) {
            int target = graph->outgoing_edges[j];
            printf("%d ", target);
        }
        printf("\n");
    }
}

