#ifndef SEQUENTIAL_BC_H
#define SEQUENTIAL_BC_H

struct graph;

struct solution {
    int *shortest_distance;
    int *num_shortest_path_through;
    double *dependency;
    double *bc_score;
};

void init_solution(struct solution *sol, int node_num);
void clear_solution(struct solution *sol);
void compute_bc(graph *g, solution *sol);
void print_solution(solution *sol, int num_nodes);
void compute_bc_openmp(graph *g, solution *sol);


#endif  /* SEQUENTIAL_BC_H */
