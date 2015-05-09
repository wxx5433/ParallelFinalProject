#!/usr/bin/python

import sys



def preProcess(uniq_set):
  for line in sys.stdin:
    fields = line.strip().split(" ")
    src = int(fields[0])
    tgt = int(fields[1])
    uniq_set.add((src, tgt))
    uniq_set.add((tgt, src))

def printResult(uniq_set):
  for (a, b) in uniq_set:
    print "%d %d"%(a, b)

def main():
  uniq_set = set()
  preProcess(uniq_set)
  printResult(uniq_set)

if __name__ == "__main__":
  main()
