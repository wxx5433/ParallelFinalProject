EXECUTABLE := betweenness_centrality

CU_FILES   := gpu_bc_node.cu gpu_bc_node_virtual.cu deg1_order.cu edge_bc.cu

CC_FILES   := main.cpp graph.cpp cpu_bc.cpp 

LOGS	   := logs

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -fopenmp
#CXXFLAGS=-O0 -Wall -g -fopenmp

LIBS       :=
FRAMEWORKS := 

ifeq ($(ARCH), Darwin)
# Building on mac
NVCCFLAGS=-O3 -m64 -arch compute_10
FRAMEWORKS += OpenGL GLUT
LDFLAGS=-L/usr/local/cuda-7.0/lib/ -lcudart
else
# Building on Linux
NVCCFLAGS=-O3 -m64 -arch compute_35
LIBS += cudart
LDFLAGS=-L/usr/local/cuda-7.0/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/graph.o $(OBJDIR)/cpu_bc.o \
     $(OBJDIR)/gpu_bc_node.o $(OBJDIR)/gpu_bc_node_virtual.o \
     $(OBJDIR)/deg1_order.o $(OBJDIR)/edge_bc.o 


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
