#ifndef __ARRAY_H__
#define __ARRAY_H__

#include"graph.h"
#include<stdio.h>

int create_adjacency_arrays(Graph &g, int *v, int *e);
void print_arrays(int len_v, int *v, int len_e, int *e, weight_t *w);


/* create two arrays from the graph data structure
 * input: g
 * output: 
   v, e -- two edge arrays
 * return:
 *  0 - success
 * -1 - failure
 */
int create_adjacency_arrays(Graph &g, int *v, int *e){
  int j=0;
  int e_count=g.getEdgeCount();
  int n_count=g.getNodeCount();

  for(int i=0; i<n_count; i++){
    sort(g.neighbors[i].begin(), g.neighbors[i].end());
    vector<int>::iterator it; 
    for(it=g.neighbors[i].begin();it!=g.neighbors[i].end(); it++){
      e[j]=*it;  
      v[j]=i;
      j++;
    }   
  }
  if(g.directed()){
    if(j!=e_count)
      return -1; 
   }   
  else{
    if(j!=2*e_count)
      return -1; 
  }
  return 0;
}

void print_arrays(int len_v, int *v, int len_e,  int *e, weight_t *w){
  for(int i=0; i<len_v; i++)
    cout<<v[i]<<" ";
  cout<<endl;
  for(int i=0; i<len_e; i++)
    cout<<e[i]<<" ";
  cout<<endl;
  if(w){
    for(int i=0; i<len_e; i++)
      cout<<w[i]<<" ";
    cout<<endl;
  }
}
#endif
