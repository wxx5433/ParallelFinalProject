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
#include"centrality.h"
#include"common.h"

using namespace std;

__global__ void cc_bfs_kernel(int *d_v, int *d_e, int  *d_d,
    bool *d_continue, int *d_dist, int e_count){
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
    }   
  }
}

/* 
 * input: 
 *  v, e: the adjacency arrays
 *   source:  the source id 
 * output:
 *    res: the cost from the "source" to each node
 */
int cc_bfs(int n_count, int e_count, int *h_v, int *h_e, float *h_cc, bool ec){
  int *d_v, *d_e;


  cudaCheckError(cudaMalloc((void **)&d_v, sizeof(int)*e_count));
  cudaCheckError(cudaMalloc((void **)&d_e, sizeof(int)*e_count)); 

  cudaCheckError(cudaMemcpy(d_v, h_v, sizeof(int)*e_count, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_e, h_e, sizeof(int)*e_count, cudaMemcpyHostToDevice));

  int *d_d, *d_dist; 

  cudaCheckError(cudaMalloc((void **)&d_d, sizeof(int)*n_count));
  cudaCheckError(cudaMalloc((void **)&d_dist, sizeof(int)));

  int *h_d;
  h_d=(int *)malloc(sizeof(int)*n_count);
  bool *d_continue;
  cudaCheckError(cudaMalloc((void**)&d_continue, sizeof(bool)));
  
  for(int s=0; s<n_count; s++){
    for(int i=0; i<n_count; i++)
      h_d[i]=-1;
    h_d[s]=0;
    cudaCheckError(cudaMemcpy(d_d, h_d, sizeof(int)*n_count, cudaMemcpyHostToDevice));
    int threads_per_block=e_count;
    int blocks=1;
    if(e_count>MAX_THREADS_PER_BLOCK){
      blocks = (int)ceil(e_count/(float)MAX_THREADS_PER_BLOCK); 
      threads_per_block = MAX_THREADS_PER_BLOCK; 
    }
    dim3 grid(blocks);
    dim3 threads(threads_per_block);
    bool h_continue;
    int h_dist=0;
    cudaCheckError(cudaMemset(d_dist, 0, sizeof(int)));
    do{
      h_continue=false;
      cudaCheckError(cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice));
      cc_bfs_kernel<<<grid, threads>>>(d_v, d_e, d_d, d_continue, d_dist, e_count);
      checkCUDAError("Kernel invocation");
      cudaThreadSynchronize();
      h_dist++;
      cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeof(int), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));
    }while(h_continue);
    cudaCheckError(cudaMemcpy(h_d, d_d, sizeof(int)*n_count, cudaMemcpyDeviceToHost));
    if(!ec){
      int sum=0;
      int connected=n_count;
      for(int i=0; i<n_count; i++){
        if(h_d[i]==-1) 
          connected--; 
        else
          sum+=h_d[i];
      }
      if(connected==0) //isolated
        h_cc[s]=0;
      else
        h_cc[s]=1.0*(connected-1)*(connected-1)/(n_count-1)/sum;
    }
    else{ //eccentricity
      int max=-1;
      int connected=n_count;
      for(int i=0; i<n_count; i++){
        if(h_d[i]==-1) 
          connected--; 
        else{
          if(h_d[i]>max)
            max=h_d[i];
        }
      }
      if(connected==0)
        h_cc[s]=0;
      else
        h_cc[s]=1.0*(connected-1)*(connected-1)/(n_count-1)/max;
    }
  }
  free(h_d);
  cudaFree(d_d);
  cudaFree(d_continue);
  cudaFree(d_v);
  cudaFree(d_e);
  cudaFree(d_dist);
  return 0;
}

/* if ec==true, compute eccentricity centrality */
int cc(int n_count, int e_count, int * h_v, int *h_e,  float *h_c, bool ec){
  checkCUDADevice();
  cc_bfs(n_count, e_count, h_v, h_e, h_c, ec);
  return 0;
}                                 
