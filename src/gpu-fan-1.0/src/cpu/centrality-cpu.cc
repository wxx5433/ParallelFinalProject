/*
** Copyright (C) 2010 Zhiao Shi <zhiao.shi@accre.vanderbilt.edu>
**  
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 2 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software 
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include<stdio.h>
#include<unistd.h>
#include<cstdlib>
#include<string>
#include<cmath>
#include"graph.h"
#ifdef _PROFILE
#include <sys/time.h>
#endif 

using namespace std;

void print_usage(void){
  fprintf(stderr, "Options: \n");
  fprintf(stderr, "  -b (shortest path betweenness centrality)\n");
  fprintf(stderr, "  -c (closeness centrality)\n");
  fprintf(stderr, "  -e (eccentric centrality)\n");
  fprintf(stderr, "  -s (stress centrality)\n");
  fprintf(stderr, "  -g <FileName> (edge file name, required)\n");
  fprintf(stderr, "  -d (directed? optional, if given, yes. Otherwise, no)\n");
  fprintf(stderr, "  -o <FileName> (output file name, optional)\n");
  exit(1);
}

int main(int argc, char** argv){
  char c;
  string edge_file="";
  string output_f;
  Graph g;
  bool directed=false; 
  bool bc_flag=false, cc_flag=false, ec_flag=false, sc_flag=false;
#ifdef _PROFILE
  struct timeval start, finish ;
  int msec;
#endif

  if(argc<4){
    print_usage();
    exit(1);
  }

  while ((c=getopt(argc, argv, "bcesg:do:")) != -1 ) {
    switch(c){
      case 'b':
        bc_flag=true;
        break;
      case 'c':
        cc_flag=true;
        break;
      case 'e':
        ec_flag=true;
        break;
      case 's':
        sc_flag=true;
        break;
      case 'g':
        edge_file=optarg;
        output_f=edge_file+".cpu.output";
        break;
      case 'd':
        directed=true;
        break;
      case 'o':
        output_f=optarg;
        break;
      default:
        fprintf(stderr,"Unrecognized option\n");
        print_usage();
        break;
    }
  }

  if(edge_file.empty()){
    print_usage();
    exit(1);
  }

  vector<string> vs;
  vs.push_back(edge_file);
  if(g.readGraph(vs, false, false, directed)==-1){
    exit(1);
  };

  ofstream ofile;
  ofile.open(output_f.c_str(), ios::out);

  int n_count=g.getNodeCount();
  int e_count=g.getEdgeCount();

  if(bc_flag){
    vector<float> bc;
#ifdef _PROFILE
    gettimeofday (&start, NULL);
#endif
    g.shortestPathBetweenness(bc);
#ifdef _PROFILE
    gettimeofday (&finish, NULL);
#endif
    ofile<<"Shortest path betweenness centrality:"<<endl; 
    map<string,float> bc_map;
    for(int i=0; i<n_count; i++){
      bc_map[g.id2node(i)]=bc[i];
    } 
    map<string,float>::iterator it;
    for( it=bc_map.begin() ; it != bc_map.end(); it++ )
      ofile<<(*it).first << "\t" << (*it).second << endl;
#ifdef _PROFILE
    msec = finish.tv_sec * 1000 + finish.tv_usec / 1000;
    msec -= start.tv_sec * 1000 + start.tv_usec / 1000;
    ofile<<"Time: "<<msec<<" milliseconds"<<endl<<endl;
#endif
  }
  if(cc_flag || ec_flag){
    if(cc_flag){
      vector<float> cc;
      for(int i=0; i<n_count; i++){
        cc.push_back(0);
      }
#ifdef _PROFILE
    gettimeofday (&start, NULL);
#endif
      g.closeness(cc);
#ifdef _PROFILE
    gettimeofday (&finish, NULL);
#endif
      map<string,float> cc_map;
      for(int i=0; i<n_count; i++){
        cc_map[g.id2node(i)]=cc[i];
      }
      ofile<<"Closeness centrality:"<<endl;
      map<string,float>::iterator it;
      for( it=cc_map.begin() ; it != cc_map.end(); it++ )
        ofile<<(*it).first << "\t" << (*it).second << endl;
#ifdef _PROFILE
    msec = finish.tv_sec * 1000 + finish.tv_usec / 1000;
    msec -= start.tv_sec * 1000 + start.tv_usec / 1000;
    ofile<<"Time: "<<msec<<" milliseconds"<<endl<<endl;
#endif
    }
    if(ec_flag){
      vector<float> ec;
      for(int i=0; i<n_count; i++){
        ec.push_back(0);
      }
#ifdef _PROFILE
    gettimeofday (&start, NULL);
#endif
      g.eccentricity(ec); 
#ifdef _PROFILE
    gettimeofday (&finish, NULL);
#endif
      map<string,float> ec_map;
      for(int i=0; i<n_count; i++){
        ec_map[g.id2node(i)]=ec[i];
      }
      ofile<<"Eccentricity centrality:"<<endl;
      map<string,float>::iterator it;
      for( it=ec_map.begin() ; it != ec_map.end(); it++ )
        ofile<<(*it).first << "\t" << (*it).second << endl;
#ifdef _PROFILE
    msec = finish.tv_sec * 1000 + finish.tv_usec / 1000;
    msec -= start.tv_sec * 1000 + start.tv_usec / 1000;
    ofile<<"Time: "<<msec<<" milliseconds"<<endl<<endl;
#endif
    }
  }
  if(sc_flag){
    vector<int> sc;
    for(int i=0; i<n_count; i++){
      sc.push_back(0);
    }
#ifdef _PROFILE
    gettimeofday (&start, NULL);
#endif
    g.stress(sc);
#ifdef _PROFILE
    gettimeofday (&finish, NULL);
#endif
    map<string,int> sc_map;
    for(int i=0; i<n_count; i++){
      sc_map[g.id2node(i)]=sc[i];
    }
    ofile<<"Stress centrality:"<<endl;
    map<string,int>::iterator it;
    for( it=sc_map.begin() ; it != sc_map.end(); it++ )
      ofile<<(*it).first << "\t" << (*it).second << endl;
#ifdef _PROFILE
    msec = finish.tv_sec * 1000 + finish.tv_usec / 1000;
    msec -= start.tv_sec * 1000 + start.tv_usec / 1000;
    ofile<<"Time: "<<msec<<" milliseconds"<<endl<<endl;
#endif
  }
  ofile.close();
  return 0;
}
