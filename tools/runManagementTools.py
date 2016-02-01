#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4  
'''
Tools for running HiFiLES cases automatically
'''

import os
import time # to pause runs while processors become available

# Importing modules from 'tools' folder
from inputFileTools import createInputFile # used to create input files
from machinefileTools import printMachineFile, getFreeNodes # used to create machinefile files

def product(*args, **kwds):
    # From https://docs.python.org/2/library/itertools.html#itertools.product
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def setupRunFolder(curFolder, execName, referenceFolder):
# Creates a folder with name caseName containing:
# HiFiLES executable
# pre-processed geometry matrices (.HiFiLES.bin files) from a reference folder
# meshes present in referenceFolder

    # Copy HiFiLES executable to current folder
    if os.system('cp ' + execName + ' ' + curFolder) != 0:
        raise (execName + ' does not exist')
    
    # Copy binary matrix files for this case from the reference folder
    if os.system('cp ' + os.path.join(referenceFolder, '*.HiFiLES.bin') + ' ' + curFolder) != 0:
        raise ('Binary matrix files could not be copied from ' + referenceFolder + ' to ' + curFolder)
    
    # Copy meshes from reference folder
    os.system('cp ' + os.path.join(referenceFolder, '*.msh') + ' ' + curFolder)
    os.system('cp ' + os.path.join(referenceFolder, '*.neu') + ' ' + curFolder)
    
def assembleRuns(runs):
# Creates an array of dictionaries that can be readily used to create input files with the createInputFile function
    """
    Inputs: 
    runs: dictionary with variables to be modified as keys, and arrays of corresponding entries as values
    Output:
    runOptions: array of dictionaries
    """
    
    # Get all keys and values for the dictionary
    keys = [item \
            for singleRun in runs \
            for item in singleRun[0]]
    values = [singleRun[1] for singleRun in runs]
        
    
    # Generate all possible cases
    cases = list(product(*values)) # finds all combinations of the desired inputs; equivalent to list(itertols.product(*values)) in Python 2.6+
    
    # Understand which variables are modified together
    varList = [run[0] for run in runs]
    # Stores the index of varList where the key is found
    # keyLocation[i] = j means that keys[i] is in runs[j][0]
    keyLocation = [index \
                   for key in keys \
                   for (index,vars) in enumerate(varList) \
                   if key in vars]

    # Assemble the cases
    runOptions = []
    for case in cases:
        # ensure that fullCase[i] = case[keyLocation[i]] to enable the creation of the dictionary
        fullCase = [case[i] for i in keyLocation]
        runOptions.append( dict( zip(keys, fullCase) ) )
        
    return runOptions


def main():
    '''
    Run a series of cases with different input files
    '''
    # Specify number of processors to use per simulation
    numNodesPerCase = 2
    
    # Specify prefix of folder where all cases will be stored
    casePrefix = 'newRuns'
    
    # Specify paths to folders and files
    os.environ['HIFILES_HOME'] = '/home/mlopez14/HiFiLES-solver'
    homeFolder = os.environ['HIFILES_HOME']
    referenceFolder = os.path.join(homeFolder, 'bin', 'sense')
    referenceInputFile = os.path.join(homeFolder, 'testcases', 'navier-stokes', 'cylinder', 'input_cylinder_visc')
    execName = os.path.join(homeFolder, 'bin', 'HiFiLES')
    
    
    # Specify where summary of cases will be written
    caseFolder = os.path.join(homeFolder, 'runs', casePrefix)
    caseFileName = os.path.join(caseFolder, casePrefix + '_case_file')
    
    # Write configuration for the runs in the case file
    caseFile = open(caseFileName, 'a')
    try:
        caseFile.write('homeFolder=' + homeFolder + '\n')
        caseFile.write('referenceFolder=' + referenceFolder + '\n')
        caseFile.write('referenceInputFile=' + referenceInputFile + '\n')
        caseFile.write('execName=' + execName + '\n')
    finally:
        caseFile.close()

    # Create list of variable ranges desired
    runs = []
    runs.append( (['viscous'], [1]) )
    runs.append( (['dt'], [1e-5]) )
    runs.append( (['order'], [3]) )
    runs.append( (['plot_freq'], [1e8]) )
    runs.append( (['Mach_free_stream', 'Mach_c_ic'], [0.21, 0.2]) )
    runs.append( (['Re_free_stream', 'Re_c_ic'], [20, 21]) )

    runOptions = assembleRuns(runs)
    
    # Generate all input files necessary
    for (index, options) in enumerate(runOptions):
        
        # Pause runs while processors become free
        while True:
        # Gather free nodes
            freeNodes = ['compute-0-0', 'compute-0-1']
            # freeNodes = getFreeNodes(50)
        
            # Check if the number of free nodes is greater than the number of requested nodes
            if len(freeNodes) < numNodesPerCase:
                time.sleep(5)
            else:
                caseFile = open(caseFileName, 'a')
                try:
                    caseFile.write(casePrefix + str(index) + ': ' + str(options) + '\n')
                finally:
                    caseFile.close()
                break

        newFolder = os.path.join(caseFolder, str(index))
    
        os.system('mkdir -p ' + newFolder)
        
        newInputFile = os.path.join(newFolder, 'input_file' )
        createInputFile(referenceInputFile, newInputFile, options)
        
        mfileName = os.path.join(newFolder, 'mfile')
        printMachineFile(mfileName, freeNodes, 2)
        
        setupRunFolder(newFolder, execName, referenceFolder)


if __name__ == "__main__": 
    main()