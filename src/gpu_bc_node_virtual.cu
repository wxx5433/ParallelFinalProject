#include "gpu_bc_node_virtual.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG

using namespace std;

int *device_vmap, *device_outgoing_starts, *device_outgoing_edges;
int *device_d, *device_sigma, *device_dist;
int *device_offset, *device_nvir;
float *device_delta, *device_bc;
bool *device_cont;
dim3 blockDim_virtual;
dim3 gridDim_virtual;

dim3 blockDim1;
dim3 gridDim1;

__global__ void forward_virtual_kernel (int* device_vmap, 
    int* device_voutgoing_starts, int* device_outgoing_edges, 
    int *device_d, int *device_sigma, bool *device_cont, 
    int *device_dist, int num_vnodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_vnodes) {
    return;
  }

  int v = device_vmap[i];
  if(device_d[v] == *device_dist) {
    int start = device_voutgoing_starts[i];
    int end = device_voutgoing_starts[i + 1];
    for(int p = start; p < end; p++) {
      int w = device_outgoing_edges[p];
      if(device_d[w] == NOT_VISITED_MARKER) {
        device_d[w] = *device_dist + 1;
        *device_cont = 1;
      }
      if(device_d[w] == *device_dist + 1) {
        atomicAdd(&device_sigma[w], device_sigma[v]);
      }
    }
  }
}

__global__ void forward_virtual_stride_kernel (int* device_vmap, 
    int* device_outgoing_starts, int* device_outgoing_edges, 
    int *device_d, int *device_sigma, bool *device_cont, int *device_dist, 
    int num_vnodes, int *device_nvir, int *device_offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_vnodes) {
    return;
  }

  int v = device_vmap[i];
  if(device_d[v] == *device_dist) {
    int start = device_offset[i];
    int end = device_outgoing_starts[v + 1];
    int stride = device_nvir[v];  
    for(int p = start; p < end; p += stride) {
      int w = device_outgoing_edges[p];
      if(device_d[w] == -1) {
        device_d[w] = *device_dist + 1;
        *device_cont = 1;
      }
      if(device_d[w] == *device_dist + 1) {
        atomicAdd(&device_sigma[w], device_sigma[v]);
      }
    }
  }
}

__global__ void backward_virtual_stride_kernel (int* device_vmap, 
    int* device_outgoing_starts, int* device_outgoing_edges, 
    int *device_d, float *device_delta, int *device_dist, 
    int num_vnodes, int *device_nvir, int *device_offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_vnodes) {
    return;
  }

  int v = device_vmap[i];
  if(device_d[v] == *device_dist - 1) {
    int start = device_offset[i];
    int end = device_outgoing_starts[v + 1];
    int stride = device_nvir[v];
    float sum = 0;
    for(int p = start; p < end; p += stride) {
      int w = device_outgoing_edges[p];
      if(device_d[w] == *device_dist ) {
        sum += device_delta[w];
      }
    }
    atomicAdd(&device_delta[v], sum);
  }
}

__global__ void backward_virtual_kernel (int* device_vmap, 
    int* device_voutgoing_starts, int* device_outgoing_edges, 
    int *device_d, float *device_delta, int *device_dist, int num_vnodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_vnodes) {
    return;
  }

  int v = device_vmap[i];
  if(device_d[v] == *device_dist - 1) {
    int start = device_voutgoing_starts[i];
    int end = device_voutgoing_starts[i + 1];
    float sum = 0;
    for(int p = start; p < end; p++) {
      int w = device_outgoing_edges[p];
      if(device_d[w] == *device_dist ) {
        sum += device_delta[w];
      }
    }
    atomicAdd(&device_delta[v], sum);
  }
}

__global__ void intermediate_virtual_kernel (int *device_d, 
    int *device_sigma, float *device_delta, int *device_dist, int num_nodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_nodes) {
    device_delta[i] = 1.0f / device_sigma[i];
  }
}

/*__global__ void intermediate_virtual_kernel_deg1 (int *device_d, int *device_sigma, float *device_delta, int *device_dist, int num_nodes, int* device_weight) {*/
  /*int u = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;*/
  /*if(u < num_nodes) {*/
    /*device_delta[u] = device_weight[u] / (float)device_sigma[u];*/
  /*}*/
/*}*/

/*__global__ void compute_bc_virtual_kernel_deg1 (int s, int *device_d, float *device_delta, int *device_sigma, float *device_bc, int num_nodes, int* device_weight){*/
  /*int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;*/
  /*if(tid < num_nodes && tid != s && device_d[tid] != -1) {*/
    /*device_bc[tid] += (device_delta[tid] * device_sigma[tid] - 1) * device_weight[s];*/
  /*}*/
/*}*/

__global__ void compute_bc_virtual_kernel (int node_id, int *device_d, 
    float *device_delta, int *device_sigma, float *device_bc, int num_nodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < num_nodes && i != node_id && device_d[i] != NOT_VISITED_MARKER) {
    device_bc[i] += device_delta[i] * device_sigma[i] - 1;
  }
}

__global__ void init_virtual_kernel (int s, int *device_d, 
    int *device_sigma, int num_nodes, int* device_dist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_nodes) {
    return;
  }

  if(s == i) {
    device_d[i] = 0;
    device_sigma[i] = 1;
    *device_dist = 0;
  } else {
    device_d[i] = NOT_VISITED_MARKER;
    device_sigma[i] = 0;
  }
}


void bc_virtual_setup(const graph_virtual *g, int **device_vmap, 
    int **device_voutgoing_starts, int **device_outgoing_edges, 
    int **device_d, int **device_sigma, float **device_delta, 
    int **device_dist, float **device_bc, bool **device_cont,
    float *pre_bc) {
  int num_vnodes = g->num_vnodes;
  int num_nodes = g->num_nodes;
  int num_edges = g->num_edges;

  cudaMalloc((void **)device_vmap, sizeof(int) *  num_vnodes);
  cudaMalloc((void **)device_voutgoing_starts, sizeof(int) * (num_vnodes + 1));
  cudaMalloc((void **)device_outgoing_edges, sizeof(int) * num_edges);

  cudaMemcpy(*device_vmap, g->vmap, sizeof(int) * num_vnodes, 
      cudaMemcpyHostToDevice);
  cudaMemcpy(*device_voutgoing_starts, g->voutgoing_starts, 
      sizeof(int) * (num_vnodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(*device_outgoing_edges, g->outgoing_edges, 
      sizeof(int) * num_edges, cudaMemcpyHostToDevice);

  cudaMalloc((void **)device_d, sizeof(int) * num_nodes);

  cudaMalloc((void **)device_sigma, sizeof(int) * num_nodes);
  cudaMalloc((void **)device_delta, sizeof(float) * num_nodes);
  cudaMalloc((void **)device_dist, sizeof(int));

  cudaMalloc((void **)device_bc, sizeof(float) * num_nodes);
  cudaMemcpy(*device_bc, pre_bc, sizeof(float) * num_nodes, cudaMemcpyHostToDevice);

  cudaMalloc((void **)device_cont, sizeof(bool));
}

void bc_virtual_clean(int *device_vmap, 
    int *device_voutgoing_starts, int *device_outgoing_edges, 
    int *device_d, int *device_sigma, float *device_delta, 
    int *device_dist, float *device_bc, bool *device_cont) {
  cudaFree(device_vmap);
  cudaFree(device_voutgoing_starts);
  cudaFree(device_outgoing_edges);
  cudaFree(device_d);
  cudaFree(device_sigma);
  cudaFree(device_delta);
  cudaFree(device_dist);
  cudaFree(device_bc);
  cudaFree(device_cont);
}

int bc_virtual (const graph_virtual *g, float *bc) {
  int *device_vmap, *device_voutgoing_starts, *device_outgoing_edges;
  int *device_d, *device_sigma, *device_dist, dist;
  float *device_delta, *device_bc;
  bool cont, *device_cont;
  int num_nodes = g->num_nodes;
  int num_vnodes = g->num_vnodes;

  bc_virtual_setup(g, &device_vmap, &device_voutgoing_starts, 
      &device_outgoing_edges, &device_d, &device_sigma, 
      &device_delta, &device_dist, &device_bc, &device_cont, bc);

  dim3 blockDim_virtual(THREAD_NUM);
  dim3 gridDim_virtual((num_vnodes + blockDim_virtual.x - 1) / blockDim_virtual.x);

  dim3 blockDim(THREAD_NUM);
  dim3 gridDim((num_nodes + blockDim.x - 1) / blockDim.x);

  for(int i = 0; i < num_nodes; i++){
    dist = 0;
    init_virtual_kernel<<<gridDim,blockDim>>>(i, device_d, 
        device_sigma, num_nodes, device_dist);
    cudaDeviceSynchronize();

    do{
      cudaMemset(device_cont, false, sizeof(bool));
      forward_virtual_kernel<<<gridDim_virtual,blockDim_virtual>>>(
          device_vmap, device_voutgoing_starts, device_outgoing_edges, 
          device_d, device_sigma, device_cont, device_dist, num_vnodes);
      cudaDeviceSynchronize();
      cudaMemcpy(device_dist, &(++dist), sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(&cont, device_cont, sizeof(bool), cudaMemcpyDeviceToHost);

    } while(cont);

    cudaMemcpy(device_dist, &(--dist), sizeof(int), cudaMemcpyHostToDevice);
    intermediate_virtual_kernel<<<gridDim, blockDim>>>(device_d, device_sigma, 
        device_delta, device_dist, num_nodes);
    cudaDeviceSynchronize();
    while(dist > 1) {
      backward_virtual_kernel<<<gridDim_virtual, blockDim_virtual>>>(
          device_vmap, device_voutgoing_starts, device_outgoing_edges, 
          device_d, device_delta, device_dist, num_vnodes);
      cudaDeviceSynchronize();
      cudaMemcpy(device_dist, &(--dist), sizeof(int), cudaMemcpyHostToDevice);
    }
    compute_bc_virtual_kernel<<<gridDim, blockDim>>>(i, device_d,  
        device_delta, device_sigma, device_bc, num_nodes);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(bc, device_bc, sizeof(float) * num_nodes, cudaMemcpyDeviceToHost);
  bc_virtual_clean(device_vmap, device_voutgoing_starts, device_outgoing_edges, 
      device_d, device_sigma, device_delta, device_dist, device_bc, device_cont);

  return 0;
}

void bc_virtual_stride_setup(const graph_virtual *g, float *pre_bc) {
  int num_vnodes = g->num_vnodes;
  int num_nodes = g->num_nodes;
  int num_edges = g->num_edges;

  cudaMalloc((void **)&device_vmap, sizeof(int) *  num_vnodes);
  cudaMemcpy(device_vmap, g->vmap, 
      sizeof(int) * num_vnodes, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_outgoing_edges, sizeof(int) * num_edges);
  cudaMemcpy(device_outgoing_edges, g->outgoing_edges, 
      sizeof(int) * num_edges, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_outgoing_starts, sizeof(int) * (num_nodes + 1));
  cudaMemcpy(device_outgoing_starts, g->outgoing_starts, 
      sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_offset, sizeof(int) * (num_vnodes));
  cudaMemcpy(device_offset, g->offset, 
      sizeof(int) * (num_vnodes), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_nvir, sizeof(int) * (num_nodes));
  cudaMemcpy(device_nvir, g->nvir, 
      sizeof(int) * (num_nodes), cudaMemcpyHostToDevice);


  cudaMalloc((void **)&device_d, sizeof(int)*num_nodes);

  cudaMalloc((void **)&device_sigma, sizeof(int)*num_nodes);
  cudaMalloc((void **)&device_delta, sizeof(float)*num_nodes);
  cudaMalloc((void **)&device_dist, sizeof(int));

  cudaMalloc((void **)&device_bc, sizeof(float) * num_nodes);
  cudaMemcpy(device_bc, pre_bc, sizeof(float) * num_nodes, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_cont, sizeof(bool));
}

void bc_virtual_stride_clean() {
  cudaFree(device_vmap);
  cudaFree(device_outgoing_edges);
  cudaFree(device_outgoing_starts);
  cudaFree(device_offset);
  cudaFree(device_nvir);
  cudaFree(device_d);
  cudaFree(device_sigma);
  cudaFree(device_delta);
  cudaFree(device_dist);
  cudaFree(device_bc);
  cudaFree(device_cont);
}

/*
 * compute bc score starting from one source node
 */
int bc_virtual_stride_helper(const graph_virtual *g, int src_node) {
  bool cont, *device_cont;
  int dist = 0;
  int num_nodes = g->num_nodes;
  int num_vnodes = g->num_vnodes;

  init_virtual_kernel<<<gridDim1, blockDim1>>>(src_node, device_d, 
      device_sigma, num_nodes, device_dist);
  cudaDeviceSynchronize();
  
  do{
    cudaMemset(device_cont, false, sizeof(bool));
    forward_virtual_stride_kernel<<<gridDim_virtual, blockDim_virtual>>>(
        device_vmap, device_outgoing_starts, device_outgoing_edges, 
        device_d, device_sigma, device_cont, device_dist, num_vnodes, 
        device_nvir, device_offset);
    cudaDeviceSynchronize();
    cudaMemcpy(device_dist, &(++dist), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&cont, device_cont, sizeof(bool), cudaMemcpyDeviceToHost);

  } while(cont);

  cudaMemcpy(device_dist, &(--dist), sizeof(int), cudaMemcpyHostToDevice);
  intermediate_virtual_kernel<<<gridDim1, blockDim1>>>(device_d, 
      device_sigma, device_delta, device_dist, num_nodes);
  cudaDeviceSynchronize();
  while(dist > 1) {
    backward_virtual_stride_kernel<<<gridDim_virtual, blockDim_virtual>>>(
        device_vmap, device_outgoing_starts, device_outgoing_edges, device_d, 
        device_delta, device_dist, num_vnodes, device_nvir, device_offset);
    cudaDeviceSynchronize();
    cudaMemcpy(device_dist, &(--dist), sizeof(int), cudaMemcpyHostToDevice);
  }
  compute_bc_virtual_kernel<<<gridDim1, blockDim1>>>(src_node, device_d, 
      device_delta, device_sigma, device_bc, num_nodes);
  cudaDeviceSynchronize();

  return 0;
}


int bc_virtual_stride (const graph_virtual *g, float *bc) {
  int num_nodes = g->num_nodes;
  int num_vnodes = g->num_vnodes;

  blockDim_virtual = THREAD_NUM;
  gridDim_virtual = (num_vnodes + blockDim_virtual.x - 1) / blockDim_virtual.x;

  blockDim1 = THREAD_NUM;
  gridDim1 = (num_nodes + blockDim1.x - 1) / blockDim1.x;

  bc_virtual_stride_setup(g, bc);
      
  for(int node_id = 0; node_id < num_nodes; node_id++){
    bc_virtual_stride_helper(g, node_id);
  }

  cudaMemcpy(bc, device_bc, sizeof(float) * num_nodes, cudaMemcpyDeviceToHost);

  bc_virtual_stride_clean();
  
  return 0;
}
