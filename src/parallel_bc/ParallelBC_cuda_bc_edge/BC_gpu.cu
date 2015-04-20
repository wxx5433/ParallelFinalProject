#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include "cuBCStruct.h"
#include "device_functions.h"
#include "sm_12_atomic_functions.h"
//#include "sm_20_atomic_functions.h"

#define BLOCK_SIZE 768

void initGPUGraph(const cuGraph * pCPUGraph, cuGraph *& pGPUGraph)
{
   if(pGPUGraph)
      freeGPUGraph(pGPUGraph);
   
   pGPUGraph = (cuGraph*)calloc(1, sizeof(cuGraph));
   pGPUGraph->nnode = pCPUGraph->nnode;
   pGPUGraph->nedge = pCPUGraph->nedge;

   cudaMalloc ((void **) &(pGPUGraph->edge_node1), sizeof(int)*pGPUGraph->nedge*2);
   cudaMalloc ((void **) &(pGPUGraph->edge_node2), sizeof(int)*pGPUGraph->nedge*2);
   cudaMalloc ((void **) &(pGPUGraph->index_list), sizeof(int)*(pGPUGraph->nnode+1));

   cudaMemcpy(pGPUGraph->edge_node1, pCPUGraph->edge_node1, sizeof(int)*pGPUGraph->nedge*2, cudaMemcpyHostToDevice);
   cudaMemcpy(pGPUGraph->edge_node2, pCPUGraph->edge_node2, sizeof(int)*pGPUGraph->nedge*2, cudaMemcpyHostToDevice);
   cudaMemcpy(pGPUGraph->index_list, pCPUGraph->index_list, sizeof(int)*(pGPUGraph->nnode+1), cudaMemcpyHostToDevice);

#ifdef COMPUTE_EDGE_BC
   cudaMalloc ((void **) &(pGPUGraph->edge_id), sizeof(int)*pGPUGraph->nedge*2);
   cudaMemcpy(pGPUGraph->edge_id, pCPUGraph->edge_id, sizeof(int)*pGPUGraph->nedge*2, cudaMemcpyHostToDevice);
#endif
}

void freeGPUGraph(cuGraph *& pGraph)
{
   if(pGraph)
   {
      cudaFree(pGraph->edge_node1);
      cudaFree(pGraph->edge_node2);
      cudaFree(pGraph->index_list);
#ifdef COMPUTE_EDGE_BC
      cudaFree(pGraph->edge_id);
#endif
      free(pGraph);
      pGraph = NULL;
   }
}

void initGPUBC(const cuBC * pCPUBCData, cuBC *& pGPUBCData)
{
   if(pGPUBCData)
      freeGPUBC(pGPUBCData);

   pGPUBCData = (cuBC*)calloc(1, sizeof(cuBC));
   pGPUBCData->nnode = pCPUBCData->nnode;
   pGPUBCData->nedge = pCPUBCData->nedge;

   cudaMalloc ((void **) &(pGPUBCData->numSPs), sizeof(int)*pGPUBCData->nnode);
   cudaMalloc ((void **) &(pGPUBCData->dependency), sizeof(float)*pGPUBCData->nnode);
   cudaMalloc ((void **) &(pGPUBCData->distance), sizeof(int)*pGPUBCData->nnode);
   cudaMalloc ((void **) &(pGPUBCData->nodeBC), sizeof(float)*pGPUBCData->nnode);
   cudaMemset ((void *) pGPUBCData->nodeBC, 0, sizeof(float)*pGPUBCData->nnode);
   cudaMalloc ((void **) &(pGPUBCData->successor), sizeof(bool)*pGPUBCData->nedge*2);
#ifdef COMPUTE_EDGE_BC
   cudaMalloc ((void **) &(pGPUBCData->edgeBC), sizeof(float)*pGPUBCData->nedge);
   cudaMemset ((void *) pGPUBCData->edgeBC, 0, sizeof(float)*pGPUBCData->nedge);
#endif
}

void freeGPUBC(cuBC *& pBCData)
{
   if(pBCData)
   {
      cudaFree(pBCData->successor);
      cudaFree(pBCData->numSPs);
      cudaFree(pBCData->distance);
      cudaFree(pBCData->dependency);
      cudaFree(pBCData->nodeBC);
#ifdef COMPUTE_EDGE_BC
      cudaFree(pBCData->edgeBC);
#endif
      free(pBCData);
      pBCData = NULL;
   }
}

void clearGPUBC(cuBC * pBCData)
{
   if(pBCData)
   {
      cudaMemset ((void *) pBCData->numSPs, 0, sizeof(int)*pBCData->nnode);
      cudaMemset ((void *) pBCData->dependency, 0, sizeof(float)*pBCData->nnode);
      cudaMemset ((void *) pBCData->distance, 0xff, sizeof(int)*pBCData->nnode);
      cudaMemset ((void *) pBCData->successor, 0, sizeof(bool)*pBCData->nedge*2);
   }
}

void copyBackGPUBC(const cuBC * pGPUBCData, const cuBC * pCPUBCData)
{
   if(pCPUBCData && pGPUBCData)
   {
      cudaMemcpy(pCPUBCData->nodeBC, pGPUBCData->nodeBC, sizeof(float)*pGPUBCData->nnode, cudaMemcpyDeviceToHost);
#ifdef COMPUTE_EDGE_BC
      cudaMemcpy(pCPUBCData->edgeBC, pGPUBCData->edgeBC, sizeof(float)*pGPUBCData->nedge, cudaMemcpyDeviceToHost);
#endif
   }
}


void copyBCData2GPU(const cuBC * pGPUBCData, const cuBC * pCPUBCData)
{
   cudaMemcpy(pGPUBCData->successor, pCPUBCData->successor, sizeof(bool)*pGPUBCData->nedge*2, cudaMemcpyHostToDevice);
   cudaMemcpy(pGPUBCData->numSPs, pCPUBCData->numSPs, sizeof(int)*pGPUBCData->nnode, cudaMemcpyHostToDevice);
   cudaMemcpy(pGPUBCData->distance, pCPUBCData->distance, sizeof(int)*pGPUBCData->nnode, cudaMemcpyHostToDevice);
   cudaMemcpy(pGPUBCData->dependency, pCPUBCData->dependency, sizeof(float)*pGPUBCData->nnode, cudaMemcpyHostToDevice);
   cudaMemcpy(pGPUBCData->nodeBC, pCPUBCData->nodeBC, sizeof(float)*pGPUBCData->nnode, cudaMemcpyHostToDevice);
}

void copyBCData2CPU(const cuBC * pCPUBCData, const cuBC * pGPUBCData)
{
   cudaMemcpy(pCPUBCData->successor, pGPUBCData->successor, sizeof(bool)*pGPUBCData->nedge*2, cudaMemcpyDeviceToHost);
   cudaMemcpy(pCPUBCData->numSPs, pGPUBCData->numSPs, sizeof(int)*pGPUBCData->nnode, cudaMemcpyDeviceToHost);
   cudaMemcpy(pCPUBCData->distance, pGPUBCData->distance, sizeof(int)*pGPUBCData->nnode, cudaMemcpyDeviceToHost);
   cudaMemcpy(pCPUBCData->dependency, pGPUBCData->dependency, sizeof(float)*pGPUBCData->nnode, cudaMemcpyDeviceToHost);
   cudaMemcpy(pCPUBCData->nodeBC, pGPUBCData->nodeBC, sizeof(float)*pGPUBCData->nnode, cudaMemcpyDeviceToHost);
}

struct node_list
{
   int * nodes;
   int size;
};

__global__ void cuda_computeBC_block(const cuGraph graph, 
                                     const node_list srcs)
{
   __shared__ cuBC  bcData;
   __shared__ float toprocess;
   __shared__ int   edge2;
   if(threadIdx.x==0)
   {    
      bcData = const_BCDatas[blockIdx.x];
      edge2  = (bcData.nedge<<1);
   }
   int   * pNumSPs     = const_BCDatas[blockIdx.x].numSPs;
   float * pDependency = const_BCDatas[blockIdx.x].dependency;
   __syncthreads();

   for(int src_idx = blockIdx.x; src_idx < srcs.size; src_idx += NUM_BLOCKS)
   //int src_idx = blockIdx.x;
   {
      // clear data
      for(int node_idx = threadIdx.x; node_idx < bcData.nnode; node_idx += blockDim.x)
      {
         pNumSPs[node_idx] = 0;
         pDependency[node_idx] = 0.f;
         bcData.distance[node_idx] = -1;
      }
      for(int node_idx = threadIdx.x; node_idx < edge2; node_idx += blockDim.x)
      {
         bcData.successor[node_idx] = false;
      }

      __syncthreads();

      // initial BFS
      if(threadIdx.x==0)
      {
         int node = srcs.nodes[src_idx];
         // bcData.numSPs[node] = 1;
         bcData.distance[node] = 0;
         toprocess = 1;
         int nb_cur = graph.index_list[node];
         int nb_end = graph.index_list[node+1];
         for(; nb_cur<nb_end; nb_cur++)
         {
            node = graph.edge_node2[nb_cur];
            bcData.distance[node] = 1;
            bcData.numSPs[node] = 1;
         }
      }
      
      int distance = 0;
      __syncthreads();
            
      // BFS
      while(toprocess>0)
      {         
         __syncthreads();
         toprocess = 0;
         distance ++;
         __syncthreads();

         for(int edge_idx = threadIdx.x; edge_idx < edge2; edge_idx += blockDim.x)
         {
            int from = graph.edge_node1[edge_idx];
            if(bcData.distance[from] == distance)
            {
               int to          = graph.edge_node2[edge_idx];
               int to_distance = bcData.distance[to];
               if(to_distance<0)
               {
                  bcData.distance[to] = to_distance = distance+1;
                  toprocess =1;
               }
               //else if(to_distance<distance)
               //{
               //   bcData.successor[edge_idx] = true; // predecessor
               //}
               if(to_distance>distance)
               {
                  bcData.successor[edge_idx] = true; // successor
                  atomicAdd(&pNumSPs[to], pNumSPs[from]);
               }
            }
         }
         
         __syncthreads();
      }


      // compute BC
      while(distance >1)
      {
         distance--;
         for(int node_idx = threadIdx.x; node_idx < bcData.nnode; node_idx += blockDim.x)
         {
	         if (bcData.distance[node_idx] == distance)	// n_id is in frontier
	         {
		         // get neighbor indices (starting index and number of indices)
		         int nb_cur = graph.index_list[node_idx];
		         int nb_end = graph.index_list[node_idx+1];		    

		         // for each nieghboring index...
		         float numSPs = bcData.numSPs[node_idx];
		         float dependency = 0.0f;		   
		         for (; nb_cur < nb_end; nb_cur++)
		         {			
			         if (bcData.successor[nb_cur])	// neighbor is a successor, using successor32bits
			         {
				         int nb_id = graph.edge_node2[nb_cur];
				         dependency += (1.0f + bcData.dependency[nb_id]) * numSPs / bcData.numSPs[nb_id];
			         }
		         }
		         bcData.dependency[node_idx] = dependency;
		         bcData.nodeBC[node_idx] += dependency;
	         }
         }

         __syncthreads();
      }

   }
}



__global__ void cuda_sumBC_block()
{
   __shared__ cuBC  bcData[NUM_BLOCKS];
   if(threadIdx.x<NUM_BLOCKS)
   {    
      bcData[threadIdx.x] = const_BCDatas[threadIdx.x];
   }

   __syncthreads();

   int node_idx = threadIdx.x + blockIdx.x * blockDim.x;
   if(node_idx<bcData[0].nnode)
   {
      float sum = bcData[0].nodeBC[node_idx];
	   sum += bcData[1].nodeBC[node_idx];
      sum += bcData[2].nodeBC[node_idx];
      sum += bcData[3].nodeBC[node_idx];
      sum += bcData[4].nodeBC[node_idx];
      sum += bcData[5].nodeBC[node_idx];
      sum += bcData[6].nodeBC[node_idx];
      sum += bcData[7].nodeBC[node_idx];
      sum += bcData[8].nodeBC[node_idx];
      sum += bcData[9].nodeBC[node_idx];
      sum += bcData[10].nodeBC[node_idx];
      sum += bcData[11].nodeBC[node_idx];
      sum += bcData[12].nodeBC[node_idx];
      sum += bcData[13].nodeBC[node_idx];
      sum += bcData[14].nodeBC[node_idx];
      sum += bcData[15].nodeBC[node_idx];
      sum += bcData[16].nodeBC[node_idx];
      sum += bcData[17].nodeBC[node_idx];
      sum += bcData[18].nodeBC[node_idx];
      sum += bcData[19].nodeBC[node_idx];
      sum += bcData[20].nodeBC[node_idx];
      sum += bcData[21].nodeBC[node_idx];
      sum += bcData[22].nodeBC[node_idx];
      sum += bcData[23].nodeBC[node_idx];
      sum += bcData[24].nodeBC[node_idx];
      sum += bcData[25].nodeBC[node_idx];
      sum += bcData[26].nodeBC[node_idx];
      sum += bcData[27].nodeBC[node_idx];
      sum += bcData[28].nodeBC[node_idx];
      sum += bcData[29].nodeBC[node_idx];
      bcData[0].nodeBC[node_idx] = sum * 0.5f;
   }
}

void gpuComputeBCOpt(const cuGraph * pGraph, cuBC * pBCData)
{
   // init bc data for each block
   cuBC pCPUBCDatas[NUM_BLOCKS];
   cuBC * pTmp[NUM_BLOCKS]; 
   pCPUBCDatas[0] = *pBCData;
   for(int i=1; i<NUM_BLOCKS; i++)
   {
      pTmp[i] = NULL;
      initGPUBC(pBCData, pTmp[i]);
      pCPUBCDatas[i] = *pTmp[i];
   }
   cudaMemcpyToSymbol(const_BCDatas, pCPUBCDatas, sizeof(cuBC)*NUM_BLOCKS);   

   // prepare source node list
   node_list srcs;
   srcs.size = pGraph->nnode;
   cudaMalloc ((void **) &(srcs.nodes), sizeof(int)*pGraph->nnode);
   int * tmp = (int*)calloc(pGraph->nnode, sizeof(int));
   for(int i=0; i<pGraph->nnode; i++) tmp[i] = i;
   cudaMemcpy(srcs.nodes, tmp, sizeof(int)*pGraph->nnode, cudaMemcpyHostToDevice);
   free(tmp);

   /*unsigned int kernel_timer;*/
	/*startTimer(kernel_timer);*/

   // call kernels
   cuda_computeBC_block<<<NUM_BLOCKS, BLOCK_SIZE>>>(*pGraph, srcs);

   int num_blocks = (pBCData->nnode + BLOCK_SIZE-1)/BLOCK_SIZE;
   cuda_sumBC_block<<<num_blocks, BLOCK_SIZE>>>();
   
   /*printf("Kernel time: %f (ms)\n", endTimer(kernel_timer));*/
  
   cudaFree(srcs.nodes);
   for(int i=1; i<NUM_BLOCKS; i++)
      freeGPUBC(pTmp[i]);
}


void gpuComputeBCApprox(const cuGraph * pGraph, cuBC * pBCData)
{
   int nsize = pGraph->nnode;
   int * tmp = (int*)calloc(nsize, sizeof(int));
   for(int i=0; i<pGraph->nnode; i++) tmp[i] = i;
    
   int nkeep = 4096;//std::max<int>(50, (int)(log(nsize*1.0)/log(2.0)*10));
   nkeep = std::min(nsize, nkeep);
   //int nRemove = nsize - nkeep;
   printf("Keeping %d/%d nodes...\n", nkeep, nsize);	  

   for(int i=0; i<nkeep; i++)
   {
	   int idx = int(abs(rand()*1.0/RAND_MAX)*(nsize-1-i))+i;
      int t = tmp[i]; tmp[i] = tmp[idx]; tmp[idx] = t;
   }

   node_list srcs;
   srcs.size = nkeep;
   cudaMalloc ((void **) &(srcs.nodes), sizeof(int)*nkeep);
   cudaMemcpy(srcs.nodes, tmp, sizeof(int)*nkeep, cudaMemcpyHostToDevice);
   free(tmp);

   cuBC pCPUBCDatas[NUM_BLOCKS];
   cuBC * pTmp[NUM_BLOCKS]; 
   pCPUBCDatas[0] = *pBCData;
   for(int i=1; i<NUM_BLOCKS; i++)
   {
      pTmp[i] = NULL;
      initGPUBC(pBCData, pTmp[i]);
      pCPUBCDatas[i] = *pTmp[i];
   }
   cudaMemcpyToSymbol(const_BCDatas, pCPUBCDatas, sizeof(cuBC)*NUM_BLOCKS);   

   /*unsigned int kernel_timer;*/
	/*startTimer(kernel_timer);*/

   cuda_computeBC_block<<<NUM_BLOCKS, BLOCK_SIZE>>>(*pGraph, srcs);

   int num_blocks = (pBCData->nnode + BLOCK_SIZE-1)/BLOCK_SIZE;
   cuda_sumBC_block<<<num_blocks, BLOCK_SIZE>>>();

   /*printf("Kernel time: %f (ms)\n", endTimer(kernel_timer));*/

   cudaFree(srcs.nodes);
   for(int i=1; i<NUM_BLOCKS; i++)
      freeGPUBC(pTmp[i]);
}
