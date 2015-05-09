#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <list>
#include <stdio.h>
#include "utils.h"

using namespace std;

__global__ void orderEdges_kernel(int* d_xadj, int* d_adj, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int wp, end, j, p;
        wp = d_xadj[tid];
        end = d_xadj[tid+1];
        for (p = wp; p < end; ++p) {
            j = d_adj[p];
            if (j != -1) {
                d_adj[p] = -1;
                d_adj[wp++] = j;
            }
        }

    }
}

__global__ void set_degree_kernel(int* v, int* d_degrees, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_degrees[tid] = v[tid + 1] - v[tid];
        //printf("degree of node %d is %d\n", tid, d_degrees[tid]);
    }
}

__global__ void degree1_kernel (int* v, int* e, int* d_tadj, int n, float* d_bc, int* d_weight, bool *d_continue, int* d_degree) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (d_degree[tid] == 1) {
            int i, vertex, end, remwght;
            *d_continue = true;
            d_degree[tid] = 0;
            end = v[tid + 1];
            //for all neighbors of node tid
            for (i = v[tid]; i < end; ++i) {
                vertex = e[i];
                if (vertex != -1) {
                    e[i] = -1;
                    e[d_tadj[i]] = -1;
                    remwght = n - d_weight[tid];
                    d_bc[tid] += (d_weight[tid] - 1) * remwght;

                    atomicAdd(d_bc + vertex, d_weight[tid] * (remwght - 1));
                    atomicAdd(d_weight + vertex, d_weight[tid]);
                    atomicAdd(d_degree + vertex, -1);
                    break;
                }
            }
        }
    }
}



int preprocess(int *xadj, int* adj, int* tadj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp) {
    int n = *np;      // number of vertices
    int nz = xadj[n]; // number of edges    
    //printf("n is %d and nz is %d\n", n, nz);
    //fflush(0);
    int *d_xadj, *d_adj, *d_weight, *d_tadj;
    int *d_degrees;
    float *d_bc;
    bool h_continue, *d_continue;

    //h_degrees = (int*)malloc(sizeof(int) * n);
    checkCudaErrors(cudaMalloc((void **)&d_xadj, sizeof(int)*(n+1)));
    checkCudaErrors(cudaMalloc((void **)&d_adj, sizeof(int)* nz));
    checkCudaErrors(cudaMalloc((void **)&d_tadj, sizeof(int)* nz));
    checkCudaErrors(cudaMalloc((void **)&d_weight, sizeof(int)* n));
    checkCudaErrors(cudaMalloc((void **)&d_bc, sizeof(float)* n));
    checkCudaErrors(cudaMalloc((void **)&d_degrees, sizeof(int)* n));
    checkCudaErrors(cudaMalloc((void **)&d_continue, sizeof(bool)));

    checkCudaErrors(cudaMemcpy(d_xadj, xadj, sizeof(int) * (n+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_adj, adj, sizeof(int) * nz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tadj, tadj, sizeof(int) * nz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_bc, 0, sizeof(float) * n));
    checkCudaErrors(cudaMemcpy(d_weight, weight, sizeof(int) * n, cudaMemcpyHostToDevice));    


    //printf("prep reaches here\n");
    int threads_per_block = n;
    int blocks = 1;
    if(n > 256){
        blocks = (int)ceil(n / (float)256);
        threads_per_block = 256;
    }
    dim3 grid(blocks);
    dim3 threads(threads_per_block);
    
    set_degree_kernel<<<grid, threads>>>(d_xadj, d_degrees, n);
    //cudaMemcpy(&h_degrees, d_degrees, sizeof(int) * n, cudaMemcpyDeviceToHost);
    //for (int i = 0; i < n; ++i) {
        //printf("%d\n", h_degrees[i]);
    //}
    do {
        h_continue = false;
        cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice);
        degree1_kernel<<<grid, threads>>>(d_xadj, d_adj, d_tadj, n, d_bc, d_weight, d_continue, d_degrees);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA Error: %s\n", cudaGetErrorString(error));
        }
        cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_continue);
    
    //reconstruct the graph
    orderEdges_kernel<<<grid,threads>>>(d_xadj, d_adj, n);
    
    checkCudaErrors(cudaMemcpy(bc, d_bc, sizeof(float) * n, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(weight, d_weight, sizeof(int) * n, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(adj, d_adj, sizeof(int) * nz, cudaMemcpyDeviceToHost));

    int i, j;
   
    cudaFree(d_xadj);
    cudaFree(d_adj);
    cudaFree(d_tadj);
    cudaFree(d_bc);
    cudaFree(d_weight);
    cudaFree(d_continue);

    //printf("prep is here\n");
    int ptr = 0, idx = 0;
    for (i = 0; i < n; ++i) {
        int flag = 0;
        for (j = xadj[i]; j < xadj[i+ 1]; ++j) {
            if (adj[j] != -1) {
                adj[ptr++] = adj[j];
            }
            else {
                flag = 1;
                xadj[idx++] = ptr;
                break;
            }
        }
        if (!flag)
            xadj[idx++] = ptr;
    }
    //printf("prep is here\n");
    for (i = idx; i > 0; i--) {
        xadj[i] = xadj[i-1];
    }
    xadj[0] = 0;
    //debug this section
    int vcount = 0;
    for (int i = 0; i < n; i++) {
        if(xadj[i+1] != xadj[i]) {
            bc[vcount] = bc[i];
            weight[vcount] = weight[i];
            map_for_order[i] = vcount;
            reverse_map_for_order[vcount] = i;
            vcount++;
            xadj[vcount] = xadj[i+1];
	} else
            fprintf(ofp, "bc[%d]: %lf\n", i, bc[i]);
    }

    //printf("i guess here\n");
    for (int i = 0; i < xadj[vcount]; i++) {
        adj[i] = map_for_order[adj[i]];
    }
    *np = vcount;

    return 0;
}


void init () {
    int* tmp;
    cudaMalloc((void **)&tmp, sizeof(int));
    cudaFree(tmp);
}


void order_graph (int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order) {

        int *new_xadj, *new_adj;

        new_xadj = (int*) calloc((n + 1), sizeof(int));
        new_adj = (int*) malloc(sizeof(int) * xadj[n]);

	int* my_map_for_order = (int *) malloc(n * sizeof(int));
	int* my_reverse_map_for_order = (int *) malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		my_map_for_order[i] = my_reverse_map_for_order[i] = -1;
	}

	int* mark = (int*) calloc((n + 1), sizeof(int));
	int* bfsorder = (int*) malloc((n + 1) * sizeof(int));
	int endofbfsorder = 0;
	int cur = 0;
	int ptr = 0;

	for (int i = 0; i < n; i++) {
		if (xadj[i+1] > xadj[i]) {
			bfsorder[endofbfsorder++] = i;
			mark[i] = 1;
			break;
		}
	}
        //printf("order reaches here\n");
	while (cur != endofbfsorder) {
		int v = bfsorder[cur];
		my_reverse_map_for_order[cur] = v;
		my_map_for_order[v] = cur;
		for (int j = xadj[v]; j < xadj[v+1]; j++) {
			int w = adj[j];
			if (mark[w] == 0) {
				mark[w] = 1;
				bfsorder[endofbfsorder++] = w;
			}
		}
		cur++;
	}
        //printf("order reaches here\n");
	for (int i = 0; i < n; i++) {
		if (mark[i] == 0) {
			my_reverse_map_for_order[cur] = i;
			my_map_for_order[i] = cur;
			cur++;
		}
	}

//printf("hahaha\n");
	ptr = 0;
	for (int i = 0; i < n; i++) {
		new_xadj[i+1] = new_xadj[i];
		int u = my_reverse_map_for_order[i];
		for (int j = xadj[u]; j < xadj[u+1]; j++) {
			int val = adj[j];
			if (!(ptr < xadj[n])) {
				printf("ptr is not less than xadj[n]\n");
				exit(1);
			}		
			if (!(val < n)) {
				printf("val %d is not less than n %d\n", val, n);
				//exit(1);
                                //continue;
			}
			new_adj[ptr++] = my_map_for_order[val];
			new_xadj[i+1]++;
		}
	}
printf("debug point1\n");
	//free(mark);
	free(bfsorder);

	int* new_weight = (int*) malloc (sizeof(int) * n);
	float* new_bc = (float*) malloc (sizeof(float) * n);
	for (int i = 0; i < n; i++) {
		new_bc[my_map_for_order[i]] = bc[i];
		new_weight[my_map_for_order[i]] = weight[i];
	}


	int* temp_map_for_order = (int *) malloc(vcount * sizeof(int));
	int* temp_reverse_map_for_order = (int *) malloc(vcount * sizeof(int));

	if (deg1) {
		for (int i = 0; i < vcount; i++) {
			if (map_for_order[i] != -1) {
				int u = my_map_for_order[map_for_order[i]];
				temp_map_for_order[i] = u;
				temp_reverse_map_for_order[u] = i;
			}
		}
	}
	else {
		for (int i = 0; i < vcount; i++) {
			int u = my_map_for_order[i];
			temp_map_for_order[i] = u;
			temp_reverse_map_for_order[u] = i;
		}
        }
	memcpy(map_for_order, temp_map_for_order, sizeof(int) * vcount);
	memcpy(reverse_map_for_order, temp_reverse_map_for_order, sizeof(int) * vcount);

	//free (my_map_for_order);
	//free (my_reverse_map_for_order);
	//free (temp_map_for_order);
	//free (temp_reverse_map_for_order);

	memcpy(xadj, new_xadj, sizeof(int) * (n+1));
	memcpy(adj, new_adj, sizeof(int) * xadj[n]);
	//free (new_adj);
	//free (new_xadj);

	memcpy(bc, new_bc, sizeof(int)*n);
	memcpy(weight, new_weight, sizeof(int)*n);
	free(new_bc);
	free(new_weight);     
        

        printf("Order graph done\n");
}
