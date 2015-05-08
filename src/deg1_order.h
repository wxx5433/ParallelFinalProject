#ifndef __DEG_ORD_H
#define __DEG_ORD_H

int preprocess(int *xadj, int* adj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp);
void order_graph (int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order);
void init();
#endif
