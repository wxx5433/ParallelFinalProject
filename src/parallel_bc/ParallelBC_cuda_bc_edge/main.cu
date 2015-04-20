#pragma warning(disable : 4996)
#include <stdio.h>
#include <string.h>
#include <string>
#include "cuBCStruct.h"
#include "constant.h"
#include "graph_indexed.h"

#include "BC_gpu.cu"

#define CPU  2
#define CPUA 3    //cpu approximate
#define GPU  4
#define GPUA 5
#define ERROR 16

// parse the input arguments
void parse(int argc, char * argv[], 
           int & mode,
           char * filename[] );

int main(int argc, char * argv[])
{
   int mode = CPU;
   char * filename[3] = {NULL, NULL, NULL};
   parse(argc, argv, mode, filename);

   GraphIndexed* pGraph = new GraphIndexed();
   if(!pGraph->Load(filename[0]))
   {
      return -1;
   }
   
   printf("Initial graph and bc data on CPU\n");
   cuGraph* pCUGraph = NULL;
   cuBC*    pBCData  = NULL;
   initGraph(pGraph, pCUGraph);
   initBC(pCUGraph, pBCData);
   
   cuGraph* pGPUCUGraph = NULL;
   cuBC*    pGPUBCData  = NULL;
   if(mode&GPU)
   {
      printf("Initial graph and bc data on GPU\n");
      initGPUGraph(pCUGraph, pGPUCUGraph);
      initGPUBC(pBCData, pGPUBCData);
   }

   std::string bcfile(filename[0]);
   bcfile = bcfile.substr(0, bcfile.length()-5);

   printf("Start computing BC\n");
   // Start timing
   /*unsigned int total_timer;*/
	/*startTimer(total_timer);*/

   switch(mode)
   {
   case ERROR:
      {
         cuBC*    pBCDataApprox  = NULL;
         initBC(pCUGraph, pBCDataApprox);
         cpuLoadBC(pBCData, filename[1]);
         cpuLoadBC(pBCDataApprox, filename[2]);
         float error = measureBCApproxError(pBCData, pBCDataApprox);
         printf("BC approximation error : %f\n", error);
      }
      break;
   case GPU:
      gpuComputeBCOpt(pGPUCUGraph, pGPUBCData);
      bcfile.append(".gpu_bc");
      break;
   case GPUA:
      gpuComputeBCApprox(pGPUCUGraph, pGPUBCData);
      bcfile.append(".gpua_bc");
      break;
   case CPUA:
      cpuComputeBCOptApprox(pCUGraph, pBCData);
      bcfile.append(".cpua_bc");
      break;
   case CPU:
   default:
      cpuComputeBCOpt(pCUGraph, pBCData);
      bcfile.append(".cpu_bc");
      break;
   }
	
   /*printf("Total time: %f (ms)\n", endTimer(total_timer));*/
   
   if(mode&GPU)
   {
      copyBackGPUBC(pGPUBCData, pBCData);
   }
   cpuSaveBC(pGraph, pBCData, bcfile.c_str());

   if(mode&GPU)
   {
      freeGPUGraph(pGPUCUGraph);
      freeGPUBC(pGPUBCData);
   }
   freeGraph(pCUGraph);
   freeBC(pBCData);
   delete pGraph;   

   return 0;
}


void parse(int argc, char * argv[],
            int & mode,
            char * filename[])
{
    for(int i=0; i<argc; i++)
    {
        if(strcmp(argv[i], "-gpu")==0 ||
           strcmp(argv[i], "-g")==0)
        {
            mode = GPU;
        }
        else if(strcmp(argv[i], "-cpu")==0 ||
           strcmp(argv[i], "-c")==0)
        {
            mode = CPU;
        }
        else if(strcmp(argv[i], "-cpua")==0 ||
           strcmp(argv[i], "-ca")==0)
        {
            mode = CPUA;
        }
        else if(strcmp(argv[i], "-gpua")==0 ||
           strcmp(argv[i], "-ga")==0)
        {
            mode = GPUA;
        }
        else if(strcmp(argv[i], "-error")==0 ||
           strcmp(argv[i], "-e")==0)
        {
            if(i+2<argc)
            {
               mode = ERROR;
               i++;
               filename[1] = argv[i];
               i++;
               filename[2] = argv[i];
            }
        }
        else if(strcmp(argv[i], "-file")==0 ||
           strcmp(argv[i], "-f")==0)
        {
            i++;
            if(i<argc)
                filename[0] = argv[i];
        }
        else if(strcmp(argv[i], "-help")==0 ||
           strcmp(argv[i], "--help")==0)
        {
            printf("cudaRaytracer [options] -f graph_file\n"
                    "options:\n"
                    "   -gpu,  -g  : running program on GPU\n"
                    "   -cpu,  -c  : running program on CPU\n"
                    "   -gpua, -ga : running approximate program on GPU\n"
                    "   -cpua, -ca : running approximate program on CPU\n"
                    "   -e bc_accurate_file bc_approx_file: compute error of bc approximations\n"
                );
            exit(0);
        }
    }
   
   if(!filename[0])
   {
       printf("cudaRaytracer [options] -f graph_file\n"
              "options:\n"
              "   -gpu,  -g  : running program on GPU\n"
              "   -cpu,  -c  : running program on CPU\n"
              "   -gpua, -ga : running approximate program on GPU\n"
              "   -cpua, -ca : running approximate program on CPU\n"
              "   -e bc_accurate_file bc_approx_file: compute error of bc approximations\n"
            );
       exit(0);
   }
}


