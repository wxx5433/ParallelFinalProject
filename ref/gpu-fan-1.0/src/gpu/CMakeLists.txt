ADD_DEFINITIONS(-D_PROFILE)
CUDA_ADD_EXECUTABLE(centrality-gpu centrality-gpu.cc bc.cu cc-ec.cu sc.cu common.cu graph.cc
  OPTIONS -arch sm_20
  )
INSTALL(PROGRAMS centrality-gpu DESTINATION bin)
