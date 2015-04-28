#include"graph.h"
#include<stdio.h>
#include"common.h"

void checkCUDADevice(){
  int device_count;
  cudaGetDeviceCount(&device_count);
  if(device_count<1){
    cout<<"No CUDA device."<<endl;
    exit(1);
  }
}

void checkCUDAError(const char *msg){
  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
	 fprintf(stderr, "Cuda error: %s : %s.\n", msg, cudaGetErrorString(err));
     exit(1);
  }    
}

void __cudaCheckError(cudaError_t err, const char *file, int line ) { 
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
    exit( EXIT_FAILURE );
  }   
}
