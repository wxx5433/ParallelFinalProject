#include "gpu_bc_node.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_NUM 256
#define DEBUG

__global__ void forward_kernel (int *outgoing_starts, int *outgoing_edges, 
    int *d, int *sigma, bool *cont, int *dist, int num_nodes) {
	int v = blockIdx.x * blockDim.x + threadIdx.x;

  if(v >= num_nodes){
    return;
  }

  if(d[v] == *dist) {
    int start = outgoing_starts[v];
    int end = outgoing_starts[v + 1];
    for(int p = start; p < end; p++) {
      int w = outgoing_edges[p];
      if(d[w] == NOT_VISITED_MARKER) {
        d[w] = *dist + 1;
        *cont = true;
      }
      if(d[w] == *dist + 1) {
        atomicAdd(&sigma[w], sigma[v]);
      }
    }
  }
}

__global__ void backward_kernel (int *outgoing_starts, 
    int* outgoing_edges, int *d, int *sigma, float *delta, float* bc, 
    int *dist, int num_nodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;

  if(v >= num_nodes) {
    return;
  }

  if(d[v] == *dist - 1) {
    int start = outgoing_starts[v];
    int end = outgoing_starts[v + 1];
    float sum = 0;
    for(int p = start; p < end; p++) {
      int w = outgoing_edges[p];
      if(d[w] == *dist) {
        sum += (float)sigma[v] / sigma[w] * (delta[w] + 1);
      }
    }
    delta[v] += sum;
  }
}

__global__ void compute_bc_kernel (int node_id, int *d, float *delta, 
    float *bc, int num_nodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;

  if(v < num_nodes && v != node_id && d[v] != NOT_VISITED_MARKER) {
    bc[v] += delta[v];
  }
}

/*
__global__ void compute_bc_kernel_deg1 (int s, int *d, float *delta, float *bc, int num_nodes, int* d_weight) {
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v < num_nodes && v != s && d[v] != -1) {
		bc[v] += delta[v] * d_weight[s];
	}
}
*/

__global__ void init_params_kernel (int s, int *d, int *sigma, 
    int num_nodes, int* dist){
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i >= num_nodes) {
    return;
  }

  if(s == i) {
    d[i] = 0;
    sigma[i] = 1;
    *dist = 0;
  } else {
    d[i] = -1;
    sigma[i] = 0;
  }
}

/*
__global__ void set_int_vertex (int* dest, int val){
	*dest = val;
}
*/

/*
__global__ void init_delta (int *d_weight, float* delta, int num_nodes) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < num_nodes) {
		delta[i] = d_weight[i]-1;
	}
}
*/

void setup(const graph *g, int **outgoing_starts, int **outgoing_edges, 
    int **d, int **sigma, float **delta, int **dist, float **bc, 
    bool **d_cont) {
  int num_nodes = g->num_nodes;
  int num_edges = g->num_edges;

  cudaMalloc((void **)outgoing_starts, sizeof(int) * (num_nodes + 1));
  cudaMalloc((void **)outgoing_edges, sizeof(int) * num_edges);

  cudaMemcpy(*outgoing_starts, g->outgoing_starts, 
      sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice); 
  cudaMemcpy(*outgoing_edges, g->outgoing_edges, 
      sizeof(int) * num_edges, cudaMemcpyHostToDevice); 

  cudaMalloc((void **)d, sizeof(int) * num_nodes);

  cudaMalloc((void **)sigma, sizeof(int) * num_nodes);
  cudaMalloc((void **)delta, sizeof(float) * num_nodes);
  cudaMalloc((void **)dist, sizeof(int));

  cudaMalloc((void **)bc, sizeof(float) * num_nodes);
  cudaMemset(*bc, 0, sizeof(float) * num_nodes);

  cudaMalloc((void **)d_cont, sizeof(bool));
}

void clean(int *outgoing_starts, int *outgoing_edges, int *d, 
    int *sigma, float *delta, int *dist, float *bc, bool *d_cont) {
  cudaFree(outgoing_starts);
  cudaFree(outgoing_edges);
  cudaFree(d);
  cudaFree(sigma);
  cudaFree(delta);
  cudaFree(dist);
  cudaFree(bc);
  cudaFree(d_cont);
}

int gpu_bc_node (const graph *g, float *bc) {
  int *device_outgoing_starts, *device_outgoing_edges;
  int *device_d, *device_sigma, *device_dist, distance;
  float *device_delta, *device_bc;
  bool cont, *device_cont;
  int num_nodes = g->num_nodes;

  setup(g, &device_outgoing_starts, &device_outgoing_edges, &device_d, 
      &device_sigma, &device_delta, &device_dist, &device_bc, &device_cont);

  dim3 blockDim(THREAD_NUM);
  dim3 gridDim((g->num_nodes + blockDim.x - 1) / blockDim.x);

  for(int node_id = 0; node_id < num_nodes; node_id++) {
    distance = 0;
    init_params_kernel<<<gridDim,blockDim>>>(node_id, device_d, device_sigma, 
      num_nodes, device_dist);

    // BFS
    do {
      cudaMemset(device_cont, false, sizeof(bool));
      forward_kernel<<<gridDim, blockDim>>>(device_outgoing_starts, 
        device_outgoing_edges, device_d, device_sigma, device_cont, 
        device_dist, num_nodes);
      cudaDeviceSynchronize();
      cudaMemcpy(device_dist, &(++distance), sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(&cont, device_cont, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (cont);

    //Back propagation
    cudaMemset(device_delta, 0, sizeof(int) * num_nodes);
    cudaMemcpy(device_dist, &(--distance), sizeof(int), cudaMemcpyHostToDevice);
    while (distance > 1) {
      backward_kernel<<<gridDim, blockDim>>>(device_outgoing_starts, 
        device_outgoing_edges, device_d, device_sigma, device_delta, 
        device_bc, device_dist, num_nodes);
      cudaDeviceSynchronize();
      cudaMemcpy(device_dist, &(--distance), sizeof(int), cudaMemcpyHostToDevice);
    }
    compute_bc_kernel<<<gridDim, blockDim>>>(node_id, device_d, 
      device_delta, device_bc, num_nodes);
  }

  cudaMemcpy(bc, device_bc, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);

  clean(device_outgoing_starts, device_outgoing_edges, device_d, 
      device_sigma, device_delta, device_dist, device_bc, device_cont);

  return 0;
}
