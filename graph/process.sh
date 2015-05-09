#!/bin/bash

tmp_file_sort="tmp.file.sort"
#tmp_file_sort=$1
tmp_file_sort_list="tmp.file.sort.list"

cat $1 | ./pre_process.py | sort -n -k1,1 -k2,2> $tmp_file_sort
num_edges=`wc -l $tmp_file_sort |  cut -d " " -f 1`
num_nodes=`tail -n 1 $tmp_file_sort |  cut -d " " -f 1`

printf "%s\t%s\n" "$num_nodes" "$num_edges" > $tmp_file_sort_list
cat $tmp_file_sort | ./process_graph_to_list.py >> $tmp_file_sort_list
cat $tmp_file_sort_list | ./process_format.py 

#rm -rf $tmp_file_sort
#rm -rf $tmp_file_sort_list
