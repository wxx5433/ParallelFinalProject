#!/usr/bin/python

import sys

def preProcess():
  for line in sys.stdin:
    fields = line.strip().split(" ")
    src = int(fields[0])
    tgt = int(fields[1])
    print "%d %d"%(src, tgt)
    print "%d %d"%(tgt, src)

def main():
  preProcess()

if __name__ == "__main__":
  main()
