#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <list>
#include "CycleTimer.h"
#include <cuda.h>
#include "utils.h"
#define THRESHOLD 256

using namespace std;

__global__ void forward_edge (int *d_v, int *d_e, int  *d_d, int *d_sigma, bool *done, int *d_dist, int num_edges) {

    int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
    if(tid < num_edges) {
        /* for each edge (u, w) */
        int u = d_v[tid];
        if(d_d[u] == *d_dist) {
            int w = d_e[tid];
            if(d_d[w] == -1) {
                d_d[w] = *d_dist + 1;
		*done = false;
	    }
            if(d_d[w] == *d_dist + 1) {
                atomicAdd(&d_sigma[w], d_sigma[u]);
            }
        }
    }
}

__global__ void backward_edge (int *d_v, int *d_e, int *d_d, int *d_sigma, float *d_delta, int *d_dist, int num_edges) {

    int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
    if(tid < num_edges) {
        int u = d_v[tid];
        if(d_d[u] == *d_dist - 1) {
            int w = d_e[tid];
            if(d_d[w] == *d_dist) {
                atomicAdd(&d_delta[u], 1.0f*d_sigma[u]/d_sigma[w]*(1.0f+d_delta[w]));
            }
        }
    }
}

__global__ void backsum_edge (int s, int *d_d, float *d_delta, float *d_bc, int num_nodes) {

    int tid =  blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
    if(tid < num_nodes && tid != s && d_d[tid] != -1) {
        d_bc[tid] += d_delta[tid];
    }
}

__global__ void init_edge (int s, int *d_d, int *d_sigma, int num_nodes, int* d_dist) {

    int i =  blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
    if(i < num_nodes) {
        d_d[i] = -1;
        d_sigma[i] = 0;
        if(s == i) {
            d_d[i] = 0;
            d_sigma[i] = 1;
            *d_dist = 0;
        }
    }
}

__global__ void set_edge (int* dest, int val) {
    *dest = val;
}

int bc_edge (int* v, int* e, int num_nodes, int num_edges, int nb, float* bc) {
    int *d_v, *d_e, *d_d, *d_sigma, *d_dist, h_dist;
    float *d_delta, *d_bc;
    bool h_done, *done;
    
    checkCudaErrors(cudaMalloc((void**)&d_v, sizeof(int) * num_edges));
    checkCudaErrors(cudaMalloc((void**)&d_e, sizeof(int) * num_edges));
    checkCudaErrors(cudaMemcpy(d_v, v, sizeof(int) * num_edges, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_e, e, sizeof(int) * num_edges, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**)&d_d, sizeof(int) * num_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_sigma, sizeof(int) * num_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_delta, sizeof(float) * num_nodes));
    checkCudaErrors(cudaMalloc((void**)&d_dist, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_bc, sizeof(float) * num_nodes));
    checkCudaErrors(cudaMemset(d_bc, 0,  sizeof(int) * num_nodes));
    checkCudaErrors(cudaMalloc((void **)&done, sizeof(bool)));
    
    int threads_per_block = num_edges;
    int blocks = 1;
    if (num_edges > THRESHOLD) {
        blocks = (int)ceil(num_edges / (float) THRESHOLD);
        blocks = (int)ceil(sqrt((float)blocks));
        threads_per_block = THRESHOLD;
    }

    dim3 num_blocks;
    num_blocks.x = blocks;
    num_blocks.y = blocks;
    dim3 threadsPerBlock(threads_per_block);
    int threads_per_block2 = num_nodes;
    int blocks2 = 1;
    if (num_nodes > THRESHOLD) {
        blocks2 = (int)ceil(num_nodes / (double)THRESHOLD);
        blocks2 = (int)ceil(sqrt((float)blocks2));
        threads_per_block2 = THRESHOLD;
    }
    dim3 num_blocks2;
    num_blocks2.x = blocks2;
    num_blocks2.y = blocks2;
    dim3 threadsPerBlock2(threads_per_block2);

    for (int i = 0; i < min(nb, num_nodes); ++i) {
        h_dist = 0;
        init_edge<<<num_blocks, threadsPerBlock>>>(i, d_d, d_sigma, num_nodes, d_dist);
        // forward propagation
        do {
            checkCudaErrors(cudaMemset(done, 1, sizeof(bool)));
            forward_edge <<<num_blocks, threadsPerBlock>>>(d_v, d_e, d_d, d_sigma, done, d_dist, num_edges);
            set_edge<<<1, 1>>>(d_dist, ++h_dist);
            checkCudaErrors(cudaMemcpy(&h_done, done, sizeof(bool), cudaMemcpyDeviceToHost));
        } while (!done);
        // backward propagation
        checkCudaErrors(cudaMemset(d_delta, 0, sizeof(int) * num_nodes));
        set_edge<<<1, 1>>>(d_dist, --h_dist);
        while (h_dist > 1) {
            backward_edge <<<num_blocks, threadsPerBlock>>>(d_v, d_e, d_d, d_sigma, d_delta, d_dist, num_edges);
            cudaThreadSynchronize();
            set_edge<<<1, 1>>>(d_dist, --h_dist);
        }
        backsum_edge <<<num_blocks2, threadsPerBlock2>>>(i, d_d,  d_delta, d_bc, num_nodes);
        cudaThreadSynchronize();
    }
    checkCudaErrors(cudaMemcpy(bc, d_bc, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost));
    cudaFree(d_v); 
    cudaFree(d_e);
    cudaFree(d_d);
    cudaFree(d_sigma);
    cudaFree(d_delta);
    cudaFree(d_dist);
    cudaFree(d_bc);
    cudaFree(done);
    return 0;
}

