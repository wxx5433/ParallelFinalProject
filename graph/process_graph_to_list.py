#!/usr/bin/python

import sys


def printRes(src, tgt_list):
    #print src + "\t" + tgt_list[0],
    print "%d\t%d" %(src, tgt_list[0]),
    for i in xrange(len(tgt_list) - 1):
        print "%d" %tgt_list[i + 1],
    print

def main():
    flag = False
    prev_src = -1
    tgt_list = list()
    for line in sys.stdin:
        if flag == True:
            fields = line.strip().split(" ")
            tgt = int(fields[0])
            src = int(fields[1])
            if prev_src == -1:
                prev_src = src
                tgt_list.append(tgt)
            elif src == prev_src:
                tgt_list.append(tgt)
            else:
                printRes(prev_src, tgt_list)
                prev_src = src
                #tgt_list.clear()
                del tgt_list[:]
                tgt_list.append(tgt)

        elif line.startswith("%") == False:
            print line.strip()
            flag = True
    printRes(prev_src, tgt_list)

if __name__ == "__main__":
    main()
