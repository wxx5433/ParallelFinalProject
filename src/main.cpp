#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include "edge_bc.h"
#include "CycleTimer.h"
#include "graph.h"
#include "cpu_bc.h"
#include "gpu_bc_node.h"
#include "gpu_bc_node_virtual.h"
#include "deg1_order.h"

#define USE_BINARY_GRAPH 0

void all_preprocess(const graph &g, int **starts, int **edges, 
    int *num_nodes, int *num_edges, float *pre_bc) {
  int n = g.num_nodes;
  int nz = g.num_edges;

  *starts = (int*)malloc(sizeof(int) * (n + 1));
  *edges = (int*)malloc(sizeof(int) * nz);
  int* degs = (int*)malloc(sizeof(int) * (n + 1));
  int* tadj = (int*)malloc(sizeof(int) * nz);

  memcpy(*starts, g.outgoing_starts, sizeof(int) * (n + 1));
  memcpy(*edges, g.outgoing_edges, sizeof(int) * nz);

  // construct tadj
  memcpy(degs, *starts, sizeof(int) * (n + 1));
  for(int i = 0; i < n; i++) {
    for(int ptr = (*starts)[i]; ptr < (*starts)[i+1]; ptr++) {
      int j = (*edges)[ptr];
      if(i < j) {
        tadj[ptr] = degs[j];
        tadj[degs[j]++] = ptr;
      }
    }
  }

  // prepare for ordering
  init();
  FILE* ofp;
  ofp = fopen("bc_out.txt", "w");   
  int* map_for_order = (int *) malloc(n * sizeof(int));
  int* reverse_map_for_order = (int *) malloc(n * sizeof(int));
  int* weight = (int *) malloc(sizeof(int) * n);
  for(int i = 0; i < n; i++) {
      weight[i] = 1;
      map_for_order[i] = -1;
      reverse_map_for_order[i] = -1;
  }

  // preprocess to remove deg1 vertex
  printf("prepro reaches here\n");
  preprocess (*starts, *edges, tadj, &n, pre_bc, weight, map_for_order, reverse_map_for_order, ofp);

  *num_nodes = n;
  *num_edges = (*starts)[n];
  
  // order graph
  printf("pre order reaches here\n"); 
  order_graph (*starts, *edges, weight, pre_bc, n, g.num_nodes, 1, map_for_order, reverse_map_for_order);
  
  free(degs);
  free(tadj);
  free(map_for_order);
  free(reverse_map_for_order);
  free(weight);
}

void test_cpu_seq(const graph &g) {
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), g.num_nodes);
  bc_cpu(g.outgoing_starts, g.outgoing_edges, g.num_nodes, g.num_edges, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for cpu_seq: " << total_time << std::endl;
  print_solution(bc, g.num_nodes);
  free(bc);
}

void test_cpu_openmp(const graph &g) {
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), g.num_nodes);
  bc_cpu_openmp(g.outgoing_starts, g.outgoing_edges, g.num_nodes, g.num_edges, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for cpu_openmp: " << total_time << std::endl;
  print_solution(bc, g.num_nodes);
  free(bc);
}

void test_gpu_node(const graph &g) {
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), g.num_nodes);
  gpu_bc_node(&g, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for node: " << total_time << std::endl;
  print_solution(bc, g.num_nodes);
  free(bc);
}

void test_gpu_edge(const graph &g) {
  // construct list for gpu_edge
  int* list = (int*)calloc(sizeof(int), g.num_edges);
  for(int i = 0; i < g.num_nodes; i++) {
    for(int j = g.outgoing_starts[i]; j < g.outgoing_starts[i+1]; j++) {
        list[j] = i;
    }
  }

  float *bc = (float*)calloc(sizeof(float), g.num_nodes);
  double start_time = CycleTimer::currentSeconds();
  bc_edge (list, g.outgoing_edges, g.num_nodes, g.num_edges, g.num_nodes, bc); 
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for edge: " << total_time << std::endl;
  print_solution(bc, g.num_nodes);
  free(bc);
  free(list);
}

void test_gpu_edge_deg1(int *starts, int *edges, int n, int nz, int num_nodes, float *pre_bc) {
  // construct list for gpu_edge
  int* list = (int*) malloc(sizeof(int) * nz);
  for(int i = 0; i < n; i++) {
      for(int j = starts[i]; j < starts[i+1]; j++) {
          list[j] = i;
      }
  }

  float *bc = (float*)calloc(sizeof(float), num_nodes);

  memcpy(bc, pre_bc, sizeof(float) * num_nodes);
  double start_time = CycleTimer::currentSeconds();
  bc_edge (list, edges, n, nz, n, bc); 
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for edge_deg1: " << total_time << std::endl;
  print_solution(bc, num_nodes);

  free(bc);
  free(list);
}

void test_virtual(const graph_virtual &g_v) {
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), g_v.num_nodes);
  bc_virtual(&g_v, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for virtual: " << total_time << std::endl;
  print_solution(bc, g_v.num_nodes);
  free(bc);
}

void test_virtual_deg1(const graph_virtual &g_v, const float *pre_bc, int num_nodes) {
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), num_nodes);
  memcpy(bc, pre_bc, sizeof(float) * num_nodes);
  bc_virtual(&g_v, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for virtual+deg1: " << total_time << std::endl;
  print_solution(bc, num_nodes);
  free(bc);
}

void test_virtual_stride(const graph_virtual &g_v) {
  // virual stride + deg1
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), g_v.num_nodes);
  bc_virtual_stride(&g_v, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for virtual+stride+deg1: " << total_time << std::endl;
  print_solution(bc, g_v.num_nodes);
  free(bc);
}

void test_virtual_stride_deg1(const graph_virtual &g_v, const float *pre_bc, int num_nodes) {
  // virual stride + deg1
  double start_time = CycleTimer::currentSeconds();
  float *bc = (float*)calloc(sizeof(float), num_nodes);
  memcpy(bc, pre_bc, sizeof(float) * num_nodes);
  bc_virtual_stride(&g_v, bc);
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for virtual+stride+deg1: " << total_time << std::endl;
  print_solution(bc, num_nodes);
  free(bc);
}

int main(int argc, char** argv) {

    std::string graph_filename;

    graph_filename = argv[1];
    graph g;

    printf("Loading graph...\n");
    load_graph(argv[1], &g);
    print_graph_stats(&g);
    //print_graph(&g);

    // build virtual graph WITHOUT deg1
    graph_virtual g_v;
    build_virtual_graph(g.outgoing_starts, g.outgoing_edges, g.num_nodes, g.num_edges, &g_v);
    print_graph_virtual_stats(&g_v);
    //print_graph_virtual(&g_v);

    // do preprocess
    float* pre_bc = (float*)malloc(sizeof(float) * g.num_nodes);
    int *starts = NULL, *edges = NULL;
    int n = g.num_nodes, nz = g.num_edges;
    all_preprocess(g, &starts, &edges, &n, &nz, pre_bc);

    // build deg1 virtual graph 
    graph_virtual g_v_deg1;
    build_virtual_graph(starts, edges, n, nz, &g_v_deg1);
    print_graph_virtual_stats(&g_v_deg1);

    //test_cpu_seq(g);

    test_cpu_openmp(g);

    //test_gpu_node(g);

    //test_gpu_edge(g);

    //test_virtual(g_v);

    //test_virtual_stride(g_v);

    test_virtual_deg1(g_v_deg1, pre_bc, g.num_nodes);

    //test_virtual_stride_deg1(g_v_deg1, pre_bc);



    /********************** clean up ***************************/
    //free(pre_bc);
    //free(starts);
    //free(edges);

    free(g.outgoing_starts);
    free(g.outgoing_edges);
    free(g_v.vmap);
    free(g_v.offset);
    free(g_v.nvir);
    free(g_v.voutgoing_starts);
    free(g_v.outgoing_starts);
    free(g_v.outgoing_edges);
    //free(g_v_deg1.vmap);
    //free(g_v_deg1.offset);
    //free(g_v_deg1.nvir);
    //free(g_v_deg1.voutgoing_starts);
    //free(g_v_deg1.outgoing_starts);
    //free(g_v_deg1.outgoing_edges);

    return 0;
}
