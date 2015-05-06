#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <iostream>
#include <omp.h>

#include "cpu_bc.h"
#include "CycleTimer.h"
#include "graph.h"

#define NOT_VISITED_MARKER -1
#define DEBUG

void malloc_solution(solution *sol, int num_nodes) {
  sol->d = (int*)malloc(sizeof(int) * num_nodes);
  sol->sigma = (int*)malloc(sizeof(int) * num_nodes);
  sol->delta = (double*)malloc(sizeof(double) * num_nodes);
  sol->bc_score = (double*)malloc(sizeof(double) * num_nodes);

  for (int i = 0; i < num_nodes; ++i) {
    sol->bc_score[i] = 0;
  }
}

void init_solution(solution *sol, int num_nodes) {
  for (int i = 0; i < num_nodes; ++i) {
    sol->d[i] = NOT_VISITED_MARKER;
    sol->sigma[i] = 0;
  }
}

void reset_delta(double *delta, int num_nodes) {
  for (int i = 0; i < num_nodes; ++i) {
    delta[i] = 0;
  }
}

void clear_solution(struct solution *sol) {
  free(sol->d);
  free(sol->sigma);
  free(sol->bc_score);
}

int forward_propagation(graph *g, solution *sol, int src_node) {
  std::queue<int> queue;
  int distance = 0;
  int cur_node;

  init_solution(sol, g->num_nodes);
  sol->d[src_node] = 0;
  sol->sigma[src_node] = 1;
  queue.push(src_node);

  while (!queue.empty()) {
    cur_node = queue.front();
    queue.pop();
    distance = sol->d[cur_node];

    int start_edge = g->outgoing_starts[cur_node];
    int end_edge = (cur_node == g->num_nodes - 1)? 
      g->num_edges: g->outgoing_starts[cur_node + 1];

    // loop through all neighbors
    for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
      int dst_node = g->outgoing_edges[neighbor];
      // not visited yet
      if (sol->d[dst_node] == NOT_VISITED_MARKER) {
        queue.push(dst_node);
        sol->d[dst_node] = distance + 1;
      }
      if (sol->d[dst_node] == distance + 1) {
        sol->sigma[dst_node] += 
            sol->sigma[src_node];
      }
    }
  }

  return distance;
}

void backward_propagation(graph *g, solution *sol, int distance, int src_node) {
  int num_nodes = g->num_nodes;
  
  reset_delta(sol->delta, num_nodes);
  while (distance > 1) {
    --distance;
    for (int cur_node = 0; cur_node < num_nodes; ++cur_node) {
      if (sol->d[cur_node] == distance) {
        int start_edge = g->outgoing_starts[cur_node];
        int end_edge = (cur_node == num_nodes - 1)? 
          g->num_edges: g->outgoing_starts[cur_node + 1];

        // loop through all neighbors
        for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
          int dst_node = g->outgoing_edges[neighbor];
          if (sol->d[dst_node] == distance + 1) {
            sol->delta[cur_node] += 
              ((double)sol->sigma[cur_node] / sol->sigma[dst_node]) *
              (sol->delta[dst_node] + 1);
          }
        }
        if (cur_node != src_node) {
          sol->bc_score[cur_node] += sol->delta[cur_node];
        }
      }
    }
  }
}

void compute_bc(graph *g, solution *sol) {
  // initialize solution structure
  malloc_solution(sol, g->num_nodes);
  int distance;

  // loop through all nodes to compute BC score
  for (int src_node = 0; src_node < g->num_nodes; ++src_node) {
#ifdef DEBUG
    double forward_start_time = CycleTimer::currentSeconds();
#endif
    distance = forward_propagation(g, sol, src_node);
#ifdef DEBUG
    double forward_time = CycleTimer::currentSeconds() - forward_start_time;
    std::cout << "src_node: " << src_node + 1 << std::endl;
    std::cout << "\tforward compute time: " << forward_time << std::endl;
    for (int i = 0; i < g->num_nodes; ++i) {
      std::cout << "\t\td" << i + 1 << ": " << sol->d[i] <<
      ", num: " << sol->sigma[i] << std::endl;
    }
    double backward_start_time = CycleTimer::currentSeconds();
#endif
    backward_propagation(g, sol, distance, src_node);
#ifdef DEBUG
    double backward_time = CycleTimer::currentSeconds() - backward_start_time;
    std::cout << "\tbackward compute time: " << backward_time << std::endl;
    for (int i = 0; i < g->num_nodes; ++i) {
      std::cout << "\t\tdelta: " << sol->delta[i] << std::endl;
    }
#endif
  }
}

void print_solution(solution *sol, int num_nodes) {
  for (int i = 0; i < num_nodes; ++i) {
    std::cout << "node id: " << i + 1 << ", bc_score: " 
      << sol->bc_score[i] << std::endl;
  }
}
