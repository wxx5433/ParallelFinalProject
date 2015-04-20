#ifndef __COMMON_H__
#define __COMMON_H__

#include"graph.h"
#include<stdio.h>
#include"driver_types.h"

extern int MAX_THREADS_PER_BLOCK;
#define cudaCheckError( err ) (__cudaCheckError( err, __FILE__, __LINE__ )) 

void checkCUDADevice();
void checkCUDAError(const char *msg);
void __cudaCheckError(cudaError_t err, const char *file, int line );
int create_adjacency_arrays(Graph &g, int *v, int *e);
void print_arrays(int len_v, int *v, int len_e, int *e, weight_t *w);

#endif
