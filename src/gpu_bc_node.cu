#include "graph.h"
#include "gpu_bc_node.h"
#include "CycleTimer.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG

const int NOT_VISITED_MARKER = -1;

__constant__ graph device_graph;

__global__ void init_params(int *d, int *sigma, int node_id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= device_graph.num_nodes) {
    return;
  }

  if (i != node_id) {
    d[i] = NOT_VISITED_MARKER;
    sigma[i] = 0;
  } else {
    d[i] = 0;
    sigma[i] = 1;
  }
}

__global__ void forward_propagation_kernel(int *d, int *sigma, int *distance, bool *done) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;

  if (v >= device_graph.num_nodes) {
    return;
  }

  if (d[v] == *distance) {
#ifdef DEBUG
    printf("enter here\n");
#endif
    int start_edge = device_graph.outgoing_starts[v];
    int end_edge = (v == device_graph.num_nodes - 1)?
      device_graph.num_edges: device_graph.outgoing_starts[v + 1];
    for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
      int w = device_graph.outgoing_edges[neighbor];
      
      if (d[w] == NOT_VISITED_MARKER) {
        d[w] = *distance + 1;
        *done = false;
#ifdef DEBUG
        printf("set done here\n");
#endif
      }
      if (d[w] == *distance + 1) {
        atomicAdd(&sigma[w], sigma[v]);
      }
    }
  }
}

__global__ void backward_propagation_kernel(int *d, int *sigma, 
    float *delta, int *distance) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;

  if (v >= device_graph.num_nodes) {
    return;
  }

  if (d[v] == *distance) {
    int start_edge = device_graph.outgoing_starts[v];
    int end_edge = (v == device_graph.num_nodes - 1)? 
      device_graph.num_edges: device_graph.outgoing_starts[v + 1];
    float sum = 0;

    // loop through all neighbors
    for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
      int w = device_graph.outgoing_edges[neighbor];
      if (d[w] == *distance + 1) {
        sum += (float)sigma[v] / sigma[w] * (delta[w] + 1);
      }
    }
    delta[v] += sum;
  }
}

__global__ void compute_bc_kernel(int src_node, int *d, float *delta, float *bc) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= device_graph.num_nodes) {
    return;
  }

  if (src_node != v && d[v] != NOT_VISITED_MARKER) {
    bc[v] += delta[v];
  }
}

void setup(const graph *g, int **d, int **sigma, int **distance,
    float **delta, float **bc, bool **done) {
  cudaMemcpyToSymbol(device_graph, g, sizeof(graph));

  // TODO try using shared memory
  cudaMalloc((void**)d, sizeof(int) * g->num_nodes);
  cudaMalloc((void**)sigma, sizeof(int) * g->num_nodes);
  cudaMalloc((void **)distance, sizeof(int));
  cudaMalloc((void**)delta, sizeof(float) * g->num_nodes);
  cudaMalloc((void**)bc, sizeof(float) * g->num_nodes);
  cudaMalloc((void**)done, sizeof(bool));

  cudaMemset(bc, 0, sizeof(float) * g->num_nodes);
}

void clean(int **d, int **sigma, int **distance, float **delta,
    float **bc, bool **done) {
  cudaFree(d);
  cudaFree(sigma);
  cudaFree(distance);
  cudaFree(delta);
  cudaFree(bc);
  cudaFree(done);
}

int gpu_bc_node (const graph *g, float *bc) {
  int *device_d, *device_sigma, *device_distance;
  float *device_delta, *device_bc;
  bool *device_done;

#ifdef DEBUG
  double start_time = CycleTimer::currentSeconds();
#endif
  setup(g, &device_d, &device_sigma, &device_distance, 
      &device_delta, &device_bc, &device_done);

  dim3 blockDim(256);
  dim3 gridDim((g->num_nodes + blockDim.x - 1) / blockDim.x);

  for (int node_id = 0; node_id < g->num_nodes; ++node_id) {
    int distance = -1;
    bool done = false;

    // initialize parameters for d and sigma
    init_params<<<gridDim, blockDim>>>(device_d, device_sigma, node_id);
    
    // forward propagation
    while (!done) {
      done = true;
      ++distance;

#ifdef DEBUG
      std::cout << "forward, distance: " << distance << std::endl;
#endif

      cudaMemset(device_done, true, sizeof(bool));
      cudaMemcpy(device_distance, &distance, sizeof(int), cudaMemcpyHostToDevice);

      forward_propagation_kernel<<<gridDim, blockDim>>>(device_d, 
          device_sigma, device_distance, device_done);
      cudaMemcpy(&done, device_done, sizeof(bool), cudaMemcpyDeviceToHost);
    }
#ifdef DEBUG
    std::cout << "node_id: " << node_id << ", distance: " << distance << std::endl;
#endif

    // backward propagation
    cudaMemset(device_delta, 0, sizeof(float) * g->num_nodes);
    --distance;
    /*distance -= 2;*/
    cudaMemcpy(device_distance, &distance, sizeof(int), cudaMemcpyHostToDevice);
    while (distance > 1) {
      backward_propagation_kernel<<<gridDim, blockDim>>>(device_d, device_sigma, 
        device_delta, device_distance);
      --distance;
      cudaMemcpy(device_distance, &distance, sizeof(int), cudaMemcpyHostToDevice);
    }

    compute_bc_kernel<<<gridDim, blockDim>>>(node_id, device_d, device_delta, device_bc);
  }

  cudaMemcpy(bc, device_bc, sizeof(float) * g->num_nodes, cudaMemcpyDeviceToHost);

  clean(&device_d, &device_sigma, &device_distance, &device_delta, 
      &device_bc, &device_done);

#ifdef DEBUG
  double total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for gpu_bc_node: " << total_time << std::endl;
#endif

  return 0;
}
