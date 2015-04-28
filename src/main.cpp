#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>

#include "CycleTimer.h"
#include "graph.h"
#include "cpu_bc.h"
//#include "bfs.h"

#define USE_BINARY_GRAPH 1

int main(int argc, char** argv) {

    int  num_threads = -1;
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
    if (USE_BINARY_GRAPH) {
        load_graph_binary(graph_filename.c_str(), &g);
    } else {
        load_graph(argv[1], &g);
        printf("storing binary form of graph!\n");
        store_graph_binary(graph_filename.append(".bin").c_str(), &g);
        exit(1);
    }
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", g.num_edges);
    printf("  Nodes: %d\n", g.num_nodes);

    solution sol1;
    compute_bc(&g, &sol1);
    print_solution(&sol1, g.num_nodes);

    //solution sol2;
    //compute_bc_openmp(&g, &sol2);
    //print_solution(&sol2, g.num_nodes);

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
