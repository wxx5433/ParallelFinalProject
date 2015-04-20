ADD_DEFINITIONS(-D_PROFILE)
ADD_EXECUTABLE(centrality-cpu centrality-cpu.cc graph.cc fheap.cc)
INSTALL(PROGRAMS centrality-cpu DESTINATION bin)
