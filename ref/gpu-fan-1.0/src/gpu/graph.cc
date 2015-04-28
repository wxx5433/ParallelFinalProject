#include "graph.h"

// default constructor
Graph::Graph(){
  digraph=false;
  edge_weighted=false;
  node_weighted=false;
}

Graph& Graph::operator=(const Graph& g){
  if(this!=&g){
    n_nodes=g.n_nodes;
	n_edges=g.n_edges;
	n_weight=g.n_weight;
	e_weight=g.e_weight;
	neighbors=g.neighbors;
	nodemap=g.nodemap;
	r_nodemap=g.r_nodemap;
	edgemap=g.edgemap;
	r_edgemap=g.r_edgemap;
  }
  return *this;
}

int Graph::getNodeCount() const{
  return n_nodes;
}

int Graph::getEdgeCount() const{
  return n_edges;
}

map<pair<int,int>,int>& Graph::getEdgeMap(){
  map<pair<int,int>,int>& em=edgemap;
  return em;
}

map<int, pair<int,int> >& Graph::getREdgeMap(){
  map<int, pair<int,int> >& em=r_edgemap;
  return em;
}

map<int, string>& Graph::getRNodeMap(){
  map<int, string>& em=r_nodemap;
  return em;
}

pair<int, int> Graph::id2edge(int eid){
  return r_edgemap[eid]; 
}

/* string tokenzier */
void Tokenize(const string& str, vector<string>& tokens,
   const string& delimiters = " "){
  string::size_type lastPos=str.find_first_not_of(delimiters, 0);
  string::size_type pos=str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)
  {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

/* find an item in a vector */
inline bool vectorFind(const int key, const vector<int>& vec) {
  for (unsigned int i = 0; i < vec.size(); i++)
    if (vec[i] == key)
      return true;
  return false;
}

/* find an item in a vector */
inline bool stringVectorFind(const string key, const vector<string>& vec) {
  for (unsigned int i = 0; i < vec.size(); i++)
    if (key.compare(vec[i]) == 0)
      return true;
  return false;
}

/* return 0 for success, -1 failure 
 * edge file  format:
   nodename1 nodename2 [weight]
   nodename3 nodename4 [weight]
   ...
  */
int Graph::readEdgeFile(string edge_file, bool edge_weighted, bool directed){
  string line;
  vector<string> tokens;
  int i;
  ifstream edgefile;
  map<string, int>::iterator nodemap_it;
  map< pair<int,int>, int>::iterator edgemap_it;
  int cur_node_num=0, cur_edge_num=0;
  int n1,n2;
  pair<int,int> edge;
  weight_t cur_weight=1; // default value

  edgefile.open(edge_file.c_str());
  if(!edgefile){
    cout<<"cannot open file: "<<edge_file<<endl;
	return -1;
  }

  while(getline(edgefile, line)){
    tokens.clear();
    Tokenize(line, tokens,"\t");
	if(tokens.size() < 2){
      cout<<"reading edge file: this is not an edge, continue..."<<endl;
	  continue;
	}
	else if(tokens.size()==3 && edge_weighted){
	  stringstream ss(tokens[2]);
	  ss>>cur_weight;
	}
    for(i=0; i<2; i++){
      nodemap_it=nodemap.find(tokens[i]);
      if(nodemap_it == nodemap.end()){ /*not found */
        nodemap.insert(make_pair(tokens[i],cur_node_num));
		r_nodemap.insert(make_pair(cur_node_num, tokens[i]));
		/* intialize node weight to INF */
		n_weight.push_back(INF);
		if(i==0)
		  n1=cur_node_num;
		else
		  n2=cur_node_num;
		/* initialize the neighbor vector for the new node */
		neighbors.push_back(vector<int>());
		cur_node_num++;
      }       
	  else{ /* the node is already added */
	    if(i==0)
          n1=nodemap_it->second; 
		else
          n2=nodemap_it->second; 
	  }
    }
	/* now adding the edge */
	edge=edgePair(n1,n2);
	/* the edge should be added previously, otherwise a duplicated edge */
	edgemap_it=edgemap.find(edge);
	if(edgemap_it != edgemap.end()){
      cout<<"reading edge file: duplicated edge"<<endl;
	}
	else{
	  edgemap.insert(make_pair(edge,cur_edge_num));
	  r_edgemap.insert(make_pair(cur_edge_num, edge));
	  e_weight.push_back((weight_t)cur_weight);
      /* update the neighbors */
	  neighbors[n1].push_back(n2);
	  if(!directed)
	    neighbors[n2].push_back(n1);
	  cur_edge_num++;
	}
  }
  edgefile.close();
  n_nodes=nodemap.size();
  n_edges=edgemap.size();
#ifdef DEBUG2
  cout<<"n_nodes="<<n_nodes<<endl;
  cout<<"n_edges="<<n_edges<<endl;
#endif
  return 0;
}

/* node file:
 * node_name weight
 * node_name weight
 */
int Graph::readNodeFile(string file_name){
  string line;
  vector<string> tokens;
  ifstream node_file;

  /* initially all node weight is INF */
  node_file.open(file_name.c_str());
  if(!node_file){
    cout<<"cannot open file "<<file_name<<endl;
	return -1;
  }
  while(getline(node_file, line)){
    tokens.clear();
    Tokenize(line, tokens,"\t");     
	if((int)tokens.size()<2){
      cout<<"Node weight not set."<<endl;
	  return -1;
	}
	/* if the gene is not found in the PPI
	    just ignore */
	if(nodemap.find(tokens[0])==nodemap.end())
	  continue;
    /* the first token is the node name */
	stringstream ss(tokens[1]);
    weight_t weight;
	ss>>weight;
    n_weight[node2id(tokens[0])]=weight;
#ifdef DEBUG2
	cout<<tokens[0]<<endl;
#endif
  } 
  node_file.close();
  return 0;
}

 /** return 0 for success, -1 failure 
  * vector<string> files: contains the name(s) of edge list file (required), node weight file (optional)
 */
int Graph::readGraph(vector<string> files, bool e_weighted, bool n_weighted, bool directed){
  edge_weighted=e_weighted;
  node_weighted=n_weighted;
  if(files.size()==0)
    return -1;
  if(readEdgeFile(files[0], edge_weighted, directed)==-1)
    return -1;
  if(node_weighted){
    if(files.size()<2){
	  cout<<"Error: no node weight file"<<endl;
	  return -1;
	}
	else
	  return readNodeFile(files[1]);
  }
  return 0;
}

/* print the degree sequence (increasing order)
 * to the stdout
 */
void Graph::printDegreeSeq(){
  vector<int> dseq;
  int n_nodes;
   
  n_nodes=getNodeCount();
  for(int i=0; i<n_nodes; i++){
    dseq.push_back(degree(i));
  }

  sort(dseq.begin(), dseq.end());
  
  cout<<"Degree sequence: ";
  for(vector<int>::iterator it=dseq.begin(); it!=dseq.end();
	  it++){
    cout<<*it<<" ";
  }
  cout<<endl;

}

/* return the degree of a node */
int Graph::degree(int node){
  if(node>=n_nodes){
    cout<<"no such node exists in the graph"<<endl;
	return -1;
  }
  return neighbors[node].size();
}

/* get edge id from edgemap */
int Graph::edge2id(int a, int b){
   map<pair<int,int>,int>::iterator id;
   pair<int, int> edge;
   
   edge=edgePair(a,b);
   id=edgemap.find(edge);
   if(id !=edgemap.end()){
	 return id->second;
   }
   else{
	 return -1;
   }
}

/* convert node name to id */
int Graph::node2id(string node){
  map<string,int>::iterator id;

  id=nodemap.find(node);
  if(id!=nodemap.end()){
    return id->second;
  }
  else{
   return -1;
  }
}

/* convert node id to name */
string Graph::id2node(int nid){
  map<int,string>::iterator it;

  it=r_nodemap.find(nid);
  if(it!=r_nodemap.end()){
    return it->second;
  }
  else
   return NULL;
}

pair<int, int> Graph::edgePair(int n1, int n2) {
  int temp;
  if(n1>n2){
    temp=n1;
    n1=n2;
    n2=temp;
  }
#ifdef DEBUG2
  cout<<"edge: "<<edge<<endl;
#endif
  return make_pair(n1,n2);
}

/* return the weight of edge connecting node n1 and n2 */   
weight_t Graph::getEdgeWeight(int n1, int n2){

  if(!isNeighbor(n1, n2))
    return INF;
  return e_weight[edge2id(n1,n2)];
}

weight_t Graph::getEdgeWeight(int e) const{
  return e_weight[e];
}

weight_t Graph::getNodeWeight(int n) const{
  if(n>n_nodes){
    cerr<<"node id > graph size, exiting..."<<endl;
	exit(1);
  }
  return n_weight[n];
}

bool Graph::isNeighbor(int n1, int n2){
  if(vectorFind(n2, neighbors[n1]))
    return true;
  else
    return false;
}

/* check if the edge weight contains negative number
 * return:
 *   true - there is negative edge weight
 *   false - otherwise
 */
bool Graph::checkEdgeWeight(){
  if(*min_element(e_weight.begin(), e_weight.end())>=0)
    return true;
  else
    return false;
}
