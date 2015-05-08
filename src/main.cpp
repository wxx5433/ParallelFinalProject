#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include "edge_bc.h"
#include "CycleTimer.h"
#include "graph.h"
#include "cpu_bc.h"
#include "gpu_bc_node.h"
#include "gpu_bc_node_virtual.h"
#include "deg1_order.h"

#define USE_BINARY_GRAPH 0

int main(int argc, char** argv) {

    //int  num_threads = -1;
    std::string graph_filename;

    if (argc < 2)
    {
        std::cerr << "Usage: <path/to/graph/file> [manual_set_thread_count]\n";
        std::cerr << "To get results across all thread counts: <path/to/graph/file>\n";
        std::cerr << "Run with certain threads count (no correctness run): <path/to/graph/file> <thread_count>\n";
        exit(1);
    }

    int thread_count = -1;
    if (argc == 3)
    {
        thread_count = atoi(argv[2]);
    }

    graph_filename = argv[1];
    graph g;

    printf("----------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    if (thread_count > 0)
    {
        thread_count = std::min(thread_count, omp_get_max_threads());
        printf("Running with %d threads\n", thread_count);
    }
    printf("----------------------------------------------------------\n");

    printf("Loading graph...\n");
    //if (USE_BINARY_GRAPH) {
        //load_graph_binary(graph_filename.c_str(), &g);
    //} else {
        load_graph(argv[1], &g);
        //printf("storing binary form of graph!\n");
        //store_graph_binary(graph_filename.append(".bin").c_str(), &g);
        //exit(1);
    //}
    //print_graph(&g);
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", g.num_edges);
    printf("  Nodes: %d\n", g.num_nodes);

    int n = g.num_nodes;
    int nz = g.num_edges;

    int* starts = (int*)malloc(sizeof(int) * (n + 1));
    int* edges = (int*)malloc(sizeof(int) * nz);
    int* degs = (int*)malloc(sizeof(int) * (n + 1));
    int* tadj = (int*)malloc(sizeof(int) * nz);

    memcpy(starts, g.outgoing_starts, sizeof(int) * (n + 1));
    memcpy(edges, g.outgoing_edges, sizeof(int) * nz);

    // construct list for gpu_edge
    int* list = (int*) malloc(sizeof(int) * nz);
    for(int i = 0; i < n; i++) {
        for(int j = starts[i]; j < starts[i+1]; j++) {
            list[j] = i;
        }
    }

    // construct tadj
    memcpy(degs, starts, sizeof(int) * (n + 1));
    for(int i = 0; i < n; i++) {
      for(int ptr = starts[i]; ptr < starts[i+1]; ptr++) {
        int j = edges[ptr];
        if(i < j) {
          tadj[ptr] = degs[j];
          tadj[degs[j]++] = ptr;
        }
      }
    }
    //printf("im here\n");    

    // prepare for ordering
    init();
    FILE* ofp;
    ofp = fopen("bc_out.txt", "w");   
    int* map_for_order = (int *) malloc(n * sizeof(int));
    int* reverse_map_for_order = (int *) malloc(n * sizeof(int));
    int* weight = (int *) malloc(sizeof(int) * n);
    float* bc = (float*)malloc(sizeof(float) * g.num_nodes);
    for(int i = 0; i < n; i++) {
        weight[i] = 1;
        map_for_order[i] = -1;
        reverse_map_for_order[i] = -1;
    }

    // preprocess to remove deg1 vertex
    printf("prepro reaches here\n");
    preprocess (starts, edges, tadj, &n, bc, weight, map_for_order, reverse_map_for_order, ofp);
    nz = starts[n];
    
    // order graph
    printf("pre order reaches here\n"); 
    order_graph (starts, edges, weight, bc, n, g.num_nodes, 1, map_for_order, reverse_map_for_order);

    ////// construct list for gpu_edge
    //int* list = (int*) malloc(sizeof(int) * nz);
    //for(int i = 0; i < n; i++) {
        //for(int j = starts[i]; j < starts[i+1]; j++) {
            //list[j] = i;
        //}
    //}

    // build new graph
    graph_virtual g_v;
    build_virtual_graph(starts, edges, n, nz, &g_v);
    //build_virtual_graph(g.outgoing_starts, g.outgoing_edges, g.num_nodes, g.num_edges, &g_v);
    //print_graph_virtual(&g_v);
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", g_v.num_edges);
    printf("  Nodes: %d\n", g_v.num_nodes);
    printf("  VNodes: %d\n", g_v.num_vnodes);

    // seq
    float start_time = CycleTimer::currentSeconds();
    float *bc_1 = (float*)malloc(sizeof(float) * g.num_nodes);
    bc_cpu(g.outgoing_starts, g.outgoing_edges, g.num_nodes, g.num_edges, bc_1);
    float total_time = CycleTimer::currentSeconds() - start_time;
    std::cout << "\ttotal time for cpu_seq: " << total_time << std::endl;
    print_solution(bc_1, g.num_nodes);
    free(bc_1);

    // openmp
    start_time = CycleTimer::currentSeconds();
    bc_1 = (float*)malloc(sizeof(float) * g.num_nodes);
    bc_cpu_openmp(g.outgoing_starts, g.outgoing_edges, g.num_nodes, g.num_edges, bc_1);
    total_time = CycleTimer::currentSeconds() - start_time;
    std::cout << "\ttotal time for cpu_openmp: " << total_time << std::endl;
    print_solution(bc_1, g.num_nodes);
    free(bc_1);

    // edge
    bc_1 = (float*)malloc(sizeof(float) * g.num_nodes);
    double begin = CycleTimer::currentSeconds();
    bc_edge (list, g.outgoing_edges, g.num_nodes, g.num_edges, g.num_nodes, bc_1); 
    double end = CycleTimer::currentSeconds();
    std::cout << "\ttotal time for edge: " << end - begin<< std::endl;
    print_solution(bc_1, g.num_nodes);

    // node
    start_time = CycleTimer::currentSeconds();
    bc_1 = (float*)malloc(sizeof(float) * g.num_nodes);
    gpu_bc_node(&g, bc_1);
    total_time = CycleTimer::currentSeconds() - start_time;
    std::cout << "\ttotal time for node: " << total_time << std::endl;
    print_solution(bc_1, g.num_nodes);
    free(bc_1);

    // virtual + deg1
    start_time = CycleTimer::currentSeconds();
    float *bc_2 = (float*)malloc(sizeof(float) * g.num_nodes);
    memcpy(bc_2, bc, sizeof(float) * g.num_nodes);
    bc_virtual(&g_v, bc_2);
    total_time = CycleTimer::currentSeconds() - start_time;
    std::cout << "\ttotal time for virtual+deg1: " << total_time << std::endl;
    print_solution(bc_2, g.num_nodes);
    free(bc_2);

    // virual stride + deg1
    start_time = CycleTimer::currentSeconds();
    float *bc_3 = (float*)malloc(sizeof(float) * g.num_nodes);
    memcpy(bc_3, bc, sizeof(float) * g.num_nodes);
    bc_virtual_stride(&g_v, bc_3);
    total_time = CycleTimer::currentSeconds() - start_time;
    std::cout << "\ttotal time for virtual+stride+deg1: " << total_time << std::endl;
    print_solution(bc_3, g.num_nodes);
    free(bc_3);

    //printf("bc_virtual_stride\n");
    //float *bc_3 = (float*)malloc(sizeof(float) * g.num_nodes);
    //bc_virtual_stride(&g_v, bc_3);
    //print_solution(bc_3, g.num_nodes);
    //free(bc_3);

    free(bc);
    free(degs);
    free(tadj);
    free(map_for_order);
    free(reverse_map_for_order);
    free(weight);
    free(g.outgoing_starts);
    free(g.outgoing_edges);
    free(g_v.vmap);
    free(g_v.offset);
    free(g_v.nvir);
    free(g_v.voutgoing_starts);
    free(g_v.outgoing_starts);
    free(g_v.outgoing_edges);

    /*
    //Run the code with only one thread count and only report speedup
    bool tds_check = true, bus_check = true, hs_check = true;
    solution sol1;
    sol1.distances = (int*)malloc(sizeof(int) * g.num_nodes);
    solution sol2;
    sol2.distances = (int*)malloc(sizeof(int) * g.num_nodes);
    solution sol3;
    sol3.distances = (int*)malloc(sizeof(int) * g.num_nodes);
    //Solution sphere
    solution sol4;
    sol4.distances = (int*)malloc(sizeof(int) * g.num_nodes);
    double hybrid_time, top_time, bottom_time;
    double ref_hybrid_time, ref_top_time, ref_bottom_time;
    double start;
    std::stringstream timing;
    std::stringstream ref_timing;
#ifdef USE_HYBRID_FUNCTION
    timing << "Threads  Top Down    Bottom Up   Hybrid\n";
    ref_timing << "Threads  Top Down    Bottom Up   Hybrid\n";
#else
    timing << "Threads  Top Down    Bottom Up\n";
    ref_timing << "Threads  Top Down    Bottom Up\n";
#endif
    //Loop through assignment values;
    std::cout << "Running with " << thread_count << " threads" << std::endl;
    //Set thread count
    omp_set_num_threads(thread_count);
    //Run implementations
    start = CycleTimer::currentSeconds();
    bfs_top_down(&g, &sol1);
    top_time = CycleTimer::currentSeconds() - start;
    //Run reference implementation
    start = CycleTimer::currentSeconds();
    reference_bfs_top_down(&g, &sol4);
    ref_top_time = CycleTimer::currentSeconds() - start;
    std::cout << "Testing Correctness of Top Down\n";
    for (int j=0; j<g.num_nodes; j++) {
        if (sol1.distances[j] != sol4.distances[j]) {
            fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, sol1.distances[j], sol4.distances[j]);
            tds_check = false;
            break;
        }
    }
    //Run implementations
    start = CycleTimer::currentSeconds();
    bfs_bottom_up(&g, &sol2);
    bottom_time = CycleTimer::currentSeconds() - start;
    //Run reference implementation
    start = CycleTimer::currentSeconds();
    reference_bfs_bottom_up(&g, &sol4);
    ref_bottom_time = CycleTimer::currentSeconds() - start;
    std::cout << "Testing Correctness of Bottom Up\n";
    for (int j=0; j<g.num_nodes; j++) {
        if (sol2.distances[j] != sol4.distances[j]) {
            fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, sol2.distances[j], sol4.distances[j]);
            bus_check = false;
            break;
        }
    }
#ifdef USE_HYBRID_FUNCTION
    start = CycleTimer::currentSeconds();
    bfs_hybrid(&g, &sol3);
    hybrid_time = CycleTimer::currentSeconds() - start;
    //Run reference implementation
    start = CycleTimer::currentSeconds();
    reference_bfs_hybrid(&g, &sol4);
    ref_hybrid_time = CycleTimer::currentSeconds() - start;
    std::cout << "Testing Correctness of Hybrid\n";
    for (int j=0; j<g.num_nodes; j++) {
        if (sol3.distances[j] != sol4.distances[j]) {
            fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, sol3.distances[j], sol4.distances[j]);
            hs_check = false;
            break;
        }
    }
#endif
    char buf[1024];
    char ref_buf[1024];
#ifdef USE_HYBRID_FUNCTION
    sprintf(buf, "%4d:     %.4f     %.4f     %.4f\n",
            thread_count, top_time, bottom_time, hybrid_time);
    sprintf(ref_buf, "%4d:     %.4f     %.4f     %.4f\n",
            thread_count, ref_top_time, ref_bottom_time, ref_hybrid_time);
#else
     sprintf(buf, "%4d:     %.4f     %.4f\n",
            thread_count, top_time, bottom_time);
     sprintf(ref_buf, "%4d:     %.4f     %.4f\n",
            thread_count, ref_top_time, ref_bottom_time);
#endif
    timing << buf;
    ref_timing << ref_buf;
    if (!tds_check)
        std::cout << "Top Down Search is not Correct" << std::endl;
    if (!bus_check)
        std::cout << "Bottom Up Search is not Correct" << std::endl;
#ifdef USE_HYBRID_FUNCTION
    if (!hs_check)
        std::cout << "Hybrid Search is not Correct" << std::endl;
#endif
    printf("----------------------------------------------------------\n");
    std::cout << "Timing Summary" << std::endl;
    std::cout << timing.str();
    printf("----------------------------------------------------------\n");
    std::cout << "Reference Summary" << std::endl;
    std::cout << ref_timing.str();
    printf("----------------------------------------------------------\n");
    */

    return 0;
}
