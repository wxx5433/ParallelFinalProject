#!/usr/bin/env python

# generate random scale-free networks 
# that follow Barabasi-Albert's pereferential attachment model

from networkx import *
from networkx.generators.random_graphs import *
from networkx.drawing import *
import string
import sys 

n=10000 
m=10

name="graph_"+str(n)+"_"+str(m);
G=barabasi_albert_graph(n,m)
write_edgelist(G,path=name,delimiter="\t",data=False)
