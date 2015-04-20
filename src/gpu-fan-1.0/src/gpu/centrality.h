#ifndef _CENTRALITY_H_
#define _CENTRALITY_H_

int bc(bool directed, int n_count, int e_count, int * h_v, int *h_e, float *h_cost);
int cc(int n_count, int e_count, int *v, int *e, float *res, bool ec);
int sc(bool directed, int n_count, int e_count, int * h_v, int *h_e, int *h_cost);

#endif
