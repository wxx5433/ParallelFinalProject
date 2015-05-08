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
    int end_edge = g->outgoing_starts[cur_node + 1];

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
        int end_edge = g->outgoing_starts[cur_node + 1];

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
  int num_nodes = g->num_nodes;
  std::vector<float> bc(num_nodes, 0);

#ifdef DEBUG
  float start_time = CycleTimer::currentSeconds();
#endif
  // loop through all nodes to compute BC score
  for (int src_node = 0; src_node < g->num_nodes; ++src_node) {
    std::vector<int> d(num_nodes, NOT_VISITED_MARKER);
    std::vector<int> sigma(num_nodes, 0);
    int distance = forward_propagation(g, src_node, d, sigma);
    backward_propagation(g, src_node, distance, d, sigma, bc);
  }
#ifdef DEBUG
  float total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for cpu_sequential: " << total_time << std::endl;
#endif

  return bc;
}

void bc_cpu (int* xadj, int* adj, int nVtx, int nz, float* bc) {

  for (int i = 0; i < nVtx; i++)
    bc[i] = 0.;

  int* bfsorder = new int[nVtx];
  int* Pred = new int[xadj[nVtx]];
  int* endpred = new int[nVtx];
  int* level = new int[nVtx];
  int* sigma = new int[nVtx];
  float* delta = new float[nVtx];

  for (int source = 0; source < nVtx; source++) {
    int endofbfsorder = 1;
    bfsorder[0] = source;

    for (int i = 0; i < nVtx; i++)
      endpred[i] = xadj[i];

    for (int i = 0; i < nVtx; i++)
      level[i] = -2;
    level[source] = 0;

    for (int i = 0; i < nVtx; i++)
      sigma[i] = 0;
    sigma[source] = 1;

    //step 1: build shortest path graph
    int cur = 0;
    while (cur != endofbfsorder) {
      int v = bfsorder[cur];
      for (int j = xadj[v]; j < xadj[v+1]; j++) {
        int w = adj[j];
        if (level[w] < 0) {
          level[w] = level[v]+1;
          bfsorder[endofbfsorder++] = w;
        }
        if (level[w] == level[v]+1) {
          sigma[w] += sigma[v];
          //assert (sigma[w] > 0); //check for overflow
          //assert (isfinite(sigma[w]));
        }
        else if (level[w] == level[v] - 1) {
          Pred[endpred[v]++] = w;
        }
      }
      cur++;
    }

    for (int i = 0; i < nVtx; i++) {
      delta[i] = 0.;
    }

    //step 2: compute betweenness
    for (int i = endofbfsorder - 1; i > 0; i--) {
      int w = bfsorder[i];
      for (int j = xadj[w]; j < endpred[w]; j++) {
        int v = Pred[j];
        delta[v] += (sigma[v] * (1 + delta[w])) / sigma[w];
      }
      bc[w] += delta[w];
    }
  }

  delete[] bfsorder;
  delete[] Pred;
  delete[] level;
  delete[] sigma;
  delete[] delta;
  delete[] endpred;
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
        int end_edge = g->outgoing_starts[cur_node + 1];

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

