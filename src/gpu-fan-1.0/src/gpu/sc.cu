/*
** Copyright (C) 2010 Zhiao Shi <zhiao.shi@accre.vanderbilt.edu>
**  
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 2 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software 
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include<iostream>
#include<unistd.h>
#include<cstdlib>
#include<string>
#include<cmath>
#include<algorithm>
#include<list>
#include"centrality.h"
#include"common.h"
#include"bitarray.h"

using namespace std;
extern int MAX_THREADS_PER_BLOCK;

__global__ void sc_bfs_kernel(int *d_v, int *d_e, int  *d_d, int *d_sigma,
  unsigned int *d_p, bool *d_continue, int *d_dist, int n_count, int e_count){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<e_count){
    /* for each edge (u, w) */
    int u=d_v[tid];
    int w=d_e[tid];
    if(d_d[u]==*d_dist){
      if(d_d[w]==-1){
        *d_continue=true;
        d_d[w]=*d_dist+1;
      }
      if(d_d[w]==*d_dist+1){
        unsigned long long bit=(unsigned long long)w*n_count+u;
        atomicOr(d_p+BIT_INT(bit), (unsigned int) BIT_IN_INT(bit));
//        atomicAdd(&d_sigma[w],d_sigma[u]);
      }
    }
  }
}

__global__ void sc_bfs_set_init_sigma(int *d_dist, int *d_d, int *d_sigma, int n_count){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<n_count){
    if(d_d[tid]==(*d_dist-1))
      d_sigma[tid]=1;
  }
}

__global__ void sc_bfs_back_prop_kernel(int *d_v, int *d_e, int *d_dist, int *d_d, int *d_sigma, unsigned int *d_p, 
     int n_count, int e_count){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<e_count){
    /* for each edge (u, w) */
    int u=d_v[tid];
    int w=d_e[tid];
    if(d_d[u]==(*d_dist-1)){
      unsigned long long bit=(unsigned long long)u*n_count+w;
      if((d_p[BIT_INT(bit)] & BIT_IN_INT(bit)) != 0){
        atomicAdd(&d_sigma[w], d_sigma[u]);
      }
    }
  }                                    
}

__global__ void sc_bfs_back_sum_kernel(int s, int *d_dist, int *d_d, int *d_sc, int *d_sigma, int n_count){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<n_count){
    if(tid!=s && d_d[tid]==(*d_dist-1))
      d_sc[tid]+=d_sigma[tid];
  }
}

/* stress centrality for unweighted graph 
 * intput: 
 *   n_count: the number of nodes
 *   e_count: the size of h_v and h_e
 *        for undirected graph, it is twice the number of edges
 *   h_v:  adjacency array 
 *   h_e:  adjacency array
 * output:
 *  h_sc: the stress centrality for each node
 *
 */
int sc_bfs(int n_count, int e_count, int *h_v, int *h_e, int *h_sc){
  int *d_v, *d_e;
  cudaCheckError(cudaMalloc((void **)&d_v, sizeof(int)*e_count));
  cudaCheckError(cudaMalloc((void **)&d_e, sizeof(int)*e_count));

  cudaCheckError(cudaMemcpy(d_v, h_v, sizeof(int)*e_count, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_e, h_e, sizeof(int)*e_count, cudaMemcpyHostToDevice));

  int *d_d, *d_sigma;
  int *d_dist;
  int *d_sc;
  unsigned int *d_p; /* two dimensional bit array (nxn)*/

  cudaCheckError(cudaMalloc((void **)&d_d, sizeof(int)*n_count));
  cudaCheckError(cudaMalloc((void **)&d_sigma, sizeof(int)*n_count)); 
  cudaCheckError(cudaMalloc((void **)&d_dist, sizeof(int)));
  cudaCheckError(cudaMalloc((void **)&d_sc, sizeof(int)*n_count));
  unsigned long long total_bits=(unsigned long long)n_count*n_count;
  unsigned int num_of_ints=BITS_TO_INTS(total_bits);  
  cudaCheckError(cudaMalloc((void **)&d_p, sizeof(unsigned int)*num_of_ints));
  cudaCheckError(cudaMalloc((void **)&d_sc, sizeof(int)*n_count));
  cudaCheckError(cudaMemset(d_sc, 0, sizeof(int)*n_count));

  int *h_d;
  h_d=(int *)malloc(sizeof(int)*n_count);
  for(int i=0; i<n_count; i++){
    for(int j=0; j<n_count; j++)
      h_d[j]=-1;
    h_d[i]=0;
    cudaCheckError(cudaMemcpy(d_d, h_d, sizeof(int)*n_count, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(d_sigma, 0, sizeof(int)*n_count));
    cudaCheckError(cudaMemset(d_p, 0, sizeof(unsigned int)*num_of_ints));     
    int threads_per_block=e_count;
    int blocks=1;
    if(e_count>MAX_THREADS_PER_BLOCK){
      blocks = (int)ceil(e_count/(float)MAX_THREADS_PER_BLOCK); 
      threads_per_block = MAX_THREADS_PER_BLOCK; 
    }
    dim3 grid(blocks);
    dim3 threads(threads_per_block);
    int threads_per_block2=n_count;
    int blocks2=1;
    if(n_count>MAX_THREADS_PER_BLOCK){
      blocks2 = (int)ceil(n_count/(double)MAX_THREADS_PER_BLOCK);
      threads_per_block2 = MAX_THREADS_PER_BLOCK; 
    }
    dim3 grid2(blocks2);
    dim3 threads2(threads_per_block2);
    bool h_continue;
    bool *d_continue;
    cudaCheckError(cudaMalloc((void **)&d_continue, sizeof(bool)));
    int h_dist=0;
    cudaCheckError(cudaMemset(d_dist, 0, sizeof(int)));
    // BFS  
    do{
      h_continue=false;
      cudaCheckError(cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice));
      sc_bfs_kernel<<<grid,threads>>>(d_v, d_e, d_d, d_sigma, d_p, d_continue, d_dist, n_count, e_count);
      checkCUDAError("Kernel sc_bfs_kernel invocation");
      cudaThreadSynchronize();
      h_dist++; 
      cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeof(int), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));
    }while(h_continue);   
    cudaCheckError(cudaMemcpy(&h_dist, d_dist, sizeof(int), cudaMemcpyDeviceToHost));
    sc_bfs_set_init_sigma<<<grid2,threads2>>>(d_dist, d_d, d_sigma, n_count);
    do{
      sc_bfs_back_prop_kernel<<<grid,threads>>>(d_v, d_e, d_dist, d_d, d_sigma, d_p, n_count, e_count);
      checkCUDAError("Kernel sc_bfs_back_prop_kernel invocation");
      cudaThreadSynchronize();
      h_dist--;
      cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeof(int), cudaMemcpyHostToDevice));
      sc_bfs_back_sum_kernel<<<grid2,threads2>>>(i, d_dist, d_d, d_sc, d_sigma, n_count);
      checkCUDAError("Kernel sc_bfs_back_sum_kernel invocation");
      cudaThreadSynchronize();
    }while(h_dist>1);
  }
  cudaCheckError(cudaMemcpy(h_sc, d_sc, sizeof(int)*n_count, cudaMemcpyDeviceToHost));
  free(h_d);
  cudaFree(d_v);
  cudaFree(d_e);
  cudaFree(d_d);
  cudaFree(d_sigma);
  cudaFree(d_dist);
  cudaFree(d_sc);
  return 0;
}

/* 
 * input: 
 *  h_v, h_e: the adjacency arrays
 *  output:
 *	 h_sc: the stress betweenness for each node
 */
int sc(bool directed, int n_count, int e_count, int * h_v, int *h_e,  int *h_sc){
  checkCUDADevice();
  sc_bfs(n_count, e_count, h_v, h_e, h_sc);
  /*
  if(!directed){
    for(int i=0; i<n_count; i++)
      h_sc[i]/=2.0;
  }
  */

  return 0;
}
