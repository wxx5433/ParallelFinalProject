#include "gpu_bc_node_virtual.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG

__global__ void forward_propagation_virtual_kernel (int *outgoing_starts, 
    int *outgoing_edges, int *vmap, int *d, int *sigma, bool *done, 
    int *dist, int num_virtual_nodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_virtual_nodes) {
    return;
  }

  int v = vmap[i];
  if(d[v] == *dist) {
    int start = outgoing_starts[v];
    int end = outgoing_starts[v + 1];
    for(int p = start; p < end; p++) {
      int w = outgoing_edges[p];
      if(d[w] == NOT_VISITED_MARKER) {
        d[w] = *dist + 1;
        *done = false;
      }
      if(d[w] == *dist + 1) {
        atomicAdd(&sigma[w], sigma[v]);
      }
    }
  }
}

__global__ void backward_propagation_virtual_kernel (int *outgoing_starts, 
    int *outgoing_edges, int *vmap, int *d, int *sigma, float *delta, 
    float* bc, int *dist, int num_virtual_nodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= num_virtual_nodes) {
    return;
  }

  int v = vmap[i];
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
    atomicAdd(&delta[v], sum);
    /*delta[v] += sum;*/
  }
}

__global__ void compute_bc_virtual_kernel (int node_id, int *d, float *delta, 
    float *bc, int num_nodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;

  if(v < num_nodes && v != node_id && d[v] != NOT_VISITED_MARKER) {
    bc[v] += delta[v];
  }
}

/*
__global__ void compute_bc_virtual_kernel_deg1 (int s, int *d, float *delta, float *bc, int num_nodes, int* d_weight) {
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v < num_nodes && v != s && d[v] != -1) {
		bc[v] += delta[v] * d_weight[s];
	}
}
*/

__global__ void init_params_virtual_kernel (int s, int *d, int *sigma, 
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

void setup(const graph_virtual *g, int **outgoing_starts, int **outgoing_edges, 
    int **vmap, int **d, int **sigma, float **delta, int **dist, float **bc, 
    bool **d_continue) {
  int num_nodes = g->num_nodes;
  int num_edges = g->num_edges;
  int num_virtual_nodes = g->num_virtual_nodes;

  cudaMalloc((void **)outgoing_starts, sizeof(int) * (num_virtual_nodes + 1));
  cudaMalloc((void **)outgoing_edges, sizeof(int) * num_edges);
  cudaMalloc((void **)vmap, sizeof(int) * (num_virtual_nodes + 1));

  cudaMemcpy(*outgoing_starts, g->outgoing_starts, 
      sizeof(int) * (num_virtual_nodes + 1), cudaMemcpyHostToDevice); 
  cudaMemcpy(*vmap, g->vmap, sizeof(int) * (num_virtual_nodes + 1), 
      cudaMemcpyHostToDevice); 
  cudaMemcpy(*outgoing_edges, g->outgoing_edges, 
      sizeof(int) * num_edges, cudaMemcpyHostToDevice); 

  cudaMalloc((void **)d, sizeof(int) * num_nodes);

  cudaMalloc((void **)sigma, sizeof(int) * num_nodes);
  cudaMalloc((void **)delta, sizeof(float) * num_nodes);
  cudaMalloc((void **)dist, sizeof(int));

  cudaMalloc((void **)bc, sizeof(float) * num_nodes);
  cudaMemset(bc, 0, sizeof(float) * num_nodes);

  cudaMalloc((void **)d_continue, sizeof(bool));
}

int gpu_bc_node_virtual (const graph_virtual *g, float *bc) {
  int *device_outgoing_starts, *device_outgoing_edges, *device_vmap;
  int *device_d, *device_sigma, *device_dist, distance;
  float *device_delta, *device_bc;
  bool done, *device_done;
  int num_nodes = g->num_nodes;
  int num_virtual_nodes = g->num_virtual_nodes;

#ifdef DEBUG
  float start_time = CycleTimer::currentSeconds();
#endif
  setup(g, &device_outgoing_starts, &device_outgoing_edges, 
      &device_vmap, &device_d, &device_sigma, &device_delta, 
      &device_dist, &device_bc, &device_done);

  dim3 blockDim(256);
  dim3 gridDim((num_nodes + blockDim.x - 1) / blockDim.x);

  dim3 blockDim_virtual(256);
  dim3 gridDim_virtual((num_virtual_nodes + blockDim_virtual.x - 1) / blockDim_virtual.x);

  for(int node_id = 0; node_id < num_nodes; node_id++) {
    distance = 0;
    init_params_virtual_kernel<<<gridDim,blockDim>>>(node_id, device_d, device_sigma, 
        num_nodes, device_dist);

    // BFS
    do {
      cudaMemset(device_done, true, sizeof(bool));
      forward_propagation_virtual_kernel<<<gridDim_virtual, blockDim_virtual>>>(device_outgoing_starts, 
          device_outgoing_edges, device_vmap, device_d, device_sigma, device_done, 
          device_dist, num_virtual_nodes);
      cudaDeviceSynchronize();
      cudaMemcpy(device_dist, &(++distance), sizeof(int), cudaMemcpyHostToDevice);
      /*set_int_vertex<<<1,1>>>(device_dist, ++distance);*/
      cudaMemcpy(&done, device_done, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (!done);


    //Back propagation
    cudaMemset(device_delta, 0, sizeof(int) * num_nodes);
    cudaMemcpy(device_dist, &(--distance), sizeof(int), cudaMemcpyHostToDevice);
    /*set_int_vertex<<<1,1>>>(device_dist, --distance);*/
    while (distance > 1) {
      backward_propagation_virtual_kernel<<<gridDim_virtual, blockDim_virtual>>>(device_outgoing_starts, 
          device_outgoing_edges, device_vmap, device_d, device_sigma, device_delta, 
          device_bc, device_dist, num_virtual_nodes);
      cudaDeviceSynchronize();
      cudaMemcpy(device_dist, &(--distance), sizeof(int), cudaMemcpyHostToDevice);
      /*set_int_vertex<<<1,1>>>(device_dist, --distance);*/
    }
    compute_bc_virtual_kernel<<<gridDim, blockDim>>>(node_id, device_d, 
        device_delta, device_bc, num_nodes);
  }

  cudaMemcpy(bc, device_bc, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);

  cudaFree(device_outgoing_starts);
  cudaFree(device_outgoing_edges);
  cudaFree(device_d);
  cudaFree(device_sigma);
  cudaFree(device_delta);
  cudaFree(device_dist);
  cudaFree(device_bc);
  cudaFree(device_done);
#ifdef DEBUG
  float total_time = CycleTimer::currentSeconds() - start_time;
  std::cout << "\ttotal time for gpu_bc_node: " << total_time << std::endl;
#endif

  return 0;
}
/*
int bc_vertex_deg1 (int *h_ptrs, int* h_js, int num_nodes, int num_edges, int nb, float *bc, int* h_weight) {

	int *device_outgoing_starts, *device_outgoing_edges, *device_d, *device_sigma, *device_dist, distance, *d_weight;
	float *device_delta, *device_bc;
	bool h_continue, *device_continue;

	cudaMalloc((void **)&device_outgoing_starts, sizeof(int) * (num_nodes + 1));
	cudaMalloc((void **)&device_outgoing_edges, sizeof(int) * num_edges);

	cudaMemcpy(device_outgoing_starts, h_ptrs, sizeof(int) * (num_nodes+1), cudaMemcpyHostToDevice); // xadj array
	cudaMemcpy(device_outgoing_edges, h_js, sizeof(int) * num_edges, cudaMemcpyHostToDevice); // adj array

	cudaMalloc((void **)&device_d, sizeof(int) * num_nodes);

	cudaMalloc((void **)&device_sigma, sizeof(int) * num_nodes);
	cudaMalloc((void **)&device_delta, sizeof(float) * num_nodes);
	cudaMalloc((void **)&d_weight, sizeof(int) * num_nodes);
	cudaMemcpy(d_weight, h_weight, sizeof(int) * num_nodes, cudaMemcpyHostToDevice); // weight array
	cudaMalloc((void **)&device_dist, sizeof(int));

	cudaMalloc((void **)&device_bc, sizeof(float) * num_nodes);
	cudaMemcpy(device_bc, bc, sizeof(int) * num_nodes, cudaMemcpyHostToDevice); // bc array

	cudaMalloc((void **)&device_continue, sizeof(bool));

	int threads_per_block = num_nodes;
	int blocks = 1;
	if(num_nodes > MTS){
		blocks = (int)ceil(num_nodes/(double)MTS);
		threads_per_block = MTS;
	}

	dim3 grid(blocks);
	dim3 threads(threads_per_block);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	for(int i = 0; i < min (nb, num_nodes); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		distance = 0;
		init_params_virtual_kernel<<<grid,threads>>>(i, device_d, device_sigma, num_nodes, device_dist);

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif

		// BFS
		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(device_continue, 0, sizeof(bool));
			forward_propagation_virtual_kernel<<<grid,threads>>>(device_outgoing_starts, device_outgoing_edges, device_d, device_sigma, device_continue, device_dist, num_nodes);
			cudaThreadSynchronize();
			set_int_vertex<<<1,1>>>(device_dist, ++distance);
			cudaMemcpy(&h_continue, device_continue, sizeof(bool), cudaMemcpyDeviceToHost);

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " << distance << " takes " << time << " secs\n";
#endif

		} while(h_continue);

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		//Back propagation

		init_delta<<<grid, threads>>>(d_weight, device_delta, num_nodes); // deltas are initialized
		set_int_vertex<<<1,1>>>(device_dist, --distance);
		while(distance > 1) {
			backward_propagation_virtual_kernel<<<grid, threads>>>(device_outgoing_starts, device_outgoing_edges, device_d, device_sigma, device_delta, device_bc, device_dist, num_nodes);
			cudaThreadSynchronize();
			set_int_vertex<<<1,1>>>(device_dist, --distance);
		}


		compute_bc_virtual_kernel_deg1<<<grid, threads>>>(i, device_d, device_delta, device_bc, num_nodes, d_weight);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif

	}

	cudaMemcpy(bc, device_bc, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);
	cudaFree(device_outgoing_starts);
	cudaFree(device_outgoing_edges);
	cudaFree(device_d);
	cudaFree(device_sigma);
	cudaFree(device_delta);
	cudaFree(device_dist);
	cudaFree(device_bc);
	cudaFree(device_continue);


	return 0;
}
*/
