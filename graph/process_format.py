#!/usr/bin/python
import sys

global_tgt_list = list()

def getMaxID():
    line = sys.stdin.readline()
    fields = line.strip().split("\t")
    maxID = int(fields[0])
    print maxID
    print fields[1]
    return maxID

def printStarts(maxID):
    starts = 0
    ID = 1
    for line in sys.stdin:
        fields = line.strip().split("\t")
        src = int(fields[0])
        tgt_list = fields[1].split(" ")
        global_tgt_list.append(tgt_list)
        while ID <= src:
            print starts
            ID += 1
        starts += len(tgt_list)
    while ID <= maxID:
        print starts
        ID += 1

def printEdges():
    for tgt_list in global_tgt_list:
        for tgt in tgt_list:
            print int(tgt)- 1

def main():
    print "AdjacencyGraph"
    maxID = getMaxID()
    printStarts(maxID)
    printEdges()

if __name__ == "__main__":
    main()
