#include "cpu_bc.h"

#define NOT_VISITED_MARKER -1
#define DEBUG

int forward_propagation(graph *g, int src_node, std::vector<int> &d, 
    std::vector<int> &sigma) {
  std::queue<int> queue;
  int distance = 0;

  d[src_node] = 0;
  sigma[src_node] = 1;
  queue.push(src_node);

  while (!queue.empty()) {
    int cur_node = queue.front();
    queue.pop();
    distance = d[cur_node];

    int start_edge = g->outgoing_starts[cur_node];
    int end_edge = (cur_node == g->num_nodes - 1)? 
      g->num_edges: g->outgoing_starts[cur_node + 1];

    // loop through all neighbors
    for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
      int dst_node = g->outgoing_edges[neighbor];
      // not visited yet
      if (d[dst_node] == NOT_VISITED_MARKER) {
        queue.push(dst_node);
        d[dst_node] = distance + 1;
      }
      if (d[dst_node] == distance + 1) {
        sigma[dst_node] += sigma[src_node];
      }
    }
  }

  return distance;
}

void backward_propagation(graph *g, int src_node, int distance,
    const std::vector<int> &d, const std::vector<int> &sigma, 
    std::vector<float> &bc) {
  int num_nodes = g->num_nodes;
  std::vector<float> delta(g->num_nodes, 0);

  while (distance > 1) {
    --distance;
    for (int cur_node = 0; cur_node < num_nodes; ++cur_node) {
      if (d[cur_node] == distance) {
        int start_edge = g->outgoing_starts[cur_node];
        int end_edge = (cur_node == num_nodes - 1)? 
          g->num_edges: g->outgoing_starts[cur_node + 1];

        // loop through all neighbors
        for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
          int dst_node = g->outgoing_edges[neighbor];
          if (d[dst_node] == distance + 1) {
            delta[cur_node] += 
              ((float)sigma[cur_node] / sigma[dst_node]) * (delta[dst_node] + 1);
          }
        }
        if (cur_node != src_node) {
          bc[cur_node] += delta[cur_node];
        }
      }
    }
  }
}

std::vector<float> compute_bc(graph *g) {
  // initialize solution structure
  //malloc_solution(sol, g->num_nodes);
  int num_nodes = g->num_nodes;
  std::vector<float> bc(num_nodes, 0);

#ifdef DEBUG
  float start_time = CycleTimer::currentSeconds();
#endif
  // loop through all nodes to compute BC score
  for (int src_node = 0; src_node < g->num_nodes; ++src_node) {
//#ifdef DEBUG
    //if (src_node % 100 == 0) {
      //std::cout << "processed " << src_node + 1 << " nodes!" << std::endl;
    //}
//#endif
    std::vector<int> d(num_nodes, NOT_VISITED_MARKER);
    std::vector<int> sigma(num_nodes, 0);
    int distance = forward_propagation(g, src_node, d, sigma);
#ifdef DEBUG
    std::cout << "node_id: " << src_node << ", distance: " << distance << std::endl;
#endif
    backward_propagation(g, src_node, distance, d, sigma, bc);
  }
#ifdef DEBUG
  float total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for cpu_sequential: " << total_time << std::endl;
#endif

  return bc;
}

void backward_propagation_openmp(graph *g, int src_node, int distance,
    const std::vector<int> &d, const std::vector<int> &sigma, 
    std::vector<float> &bc) {
  int num_nodes = g->num_nodes;
  std::vector<float> delta(g->num_nodes, 0);

  while (distance > 1) {
    --distance;
    for (int cur_node = 0; cur_node < num_nodes; ++cur_node) {
      if (d[cur_node] == distance) {
        int start_edge = g->outgoing_starts[cur_node];
        int end_edge = (cur_node == num_nodes - 1)? 
          g->num_edges: g->outgoing_starts[cur_node + 1];

        // loop through all neighbors
        for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
          int dst_node = g->outgoing_edges[neighbor];
          if (d[dst_node] == distance + 1) {
            delta[cur_node] += 
              ((float)sigma[cur_node] / sigma[dst_node]) * (delta[dst_node] + 1);
          }
        }
        if (cur_node != src_node) {
          #pragma omp atomic
          bc[cur_node] += delta[cur_node];
        }
      }
    }
  }
}

std::vector<float> compute_bc_openmp(graph *g) {
  int num_nodes = g->num_nodes;
  std::vector<float> bc(num_nodes, 0);

#ifdef DEBUG
  float start_time = CycleTimer::currentSeconds();
#endif
  // loop through all nodes to compute BC score
  #pragma omp parallel for schedule(dynamic)
  for (int src_node = 0; src_node < g->num_nodes; ++src_node) {
    std::vector<int> d(num_nodes, NOT_VISITED_MARKER);
    std::vector<int> sigma(num_nodes, 0);
    int distance = forward_propagation(g, src_node, d, sigma);
    backward_propagation_openmp(g, src_node, distance, d, sigma, bc);
  }
#ifdef DEBUG
  float total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for cpu_openmp: " << total_time << std::endl;
#endif

  return bc;
}

void print_solution(const float *bc, int num_nodes) {
  for (int i = 0; i < num_nodes; ++i) {
    std::cout << "node id: " << i + 1 << ", bc_score: " << bc[i] << std::endl;
  }
}

