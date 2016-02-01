#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4  
'''
Tools for generating an machinefile containing all nodes that are being used below a given threshold
This file can be used with mpi.
'''
import subprocess # to get cluster information with ganglia
import decimal # to convert string to float

def getFreeNodes(thresholdUsage):
# Checks for the currently running processes
# and returns a list of the nodes that are being used less than a
# certain threshold

    proc = subprocess.Popen(['ganglia', 'load_one'], stdout = subprocess.PIPE)
    (out, err) = proc.communicate()
    lines = out.split('\n')
    freeNodes = []
    for line in lines:
        if line: # if line is non-empty
            info = line.split('\t')
            node = info[0].strip()
            usage = float(decimal.Decimal(info[1]))
            if "compute-" in node and usage <= thresholdUsage:
                freeNodes.append(node)
    freeNodes.sort()
    return freeNodes

def printMachineFile(fileName, freeNodes, nRepeats):
# Creates a machine file with name fileName usable by mpi with the names of the nodes
# in freeNodes. Each name is repeated nRepeats
# times
    file = open(fileName, 'w')
    try:
        for node in freeNodes:
            for i in xrange(nRepeats):
                file.write(node + '\n')
    finally:
        file.close()


def main():
    freeNodes = ['compute-0-0', 'compute-0-1']
    printMachineFile("mfile", freeNodes, 2)


if __name__ == "__main__": 
    main()

