#!/usr/bin/python

import sys

def printRes(src, tgt_list):
  #print src + "\t" + tgt_list[0],
  print "%d\t%d" %(src, tgt_list[0]),
  for i in xrange(len(tgt_list) - 1):
      print "%d" %tgt_list[i + 1],
  print

def main():
  prev_src = -1
  tgt_list = list()
  for line in sys.stdin:
    fields = line.strip().split(" ")
    src = int(fields[0]) # modified here for different graph
    tgt = int(fields[1]) # modified here for different graph
    if prev_src == -1:
        prev_src = src
        tgt_list.append(tgt)
    elif src == prev_src:
        tgt_list.append(tgt)
    else:
        printRes(prev_src, tgt_list)
        prev_src = src
        del tgt_list[:]
        tgt_list.append(tgt)

  printRes(prev_src, tgt_list)

if __name__ == "__main__":
  main()
