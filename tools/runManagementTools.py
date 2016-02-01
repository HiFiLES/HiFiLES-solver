#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4  
'''
Tools for running HiFiLES cases automatically
'''

import os

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

    # Copy HiFiLES executable to current folder
    if os.system('cp ' + execName + ' ' + curFolder) != 0:
        raise (execName + ' does not exist')
    
    # Copy binary matrix files for this case from the reference folder
    if os.system('cp ' + os.path.join(referenceFolder, '*.HiFiLES.bin' + ' ' + curFolder)) != 0:
        raise ('Binary matrix files could not be copied from ' + referenceFolder + ' to ' + curFolder)
    
def assembleRuns(runs):
# Creates an array of dictionaries that can be readily used to create input files with the createInputFile function
    """
    Inputs: 
    runs: dictionary with variables to be modified as keys, and arrays of corresponding entries as values
    Output:
    runOptions: array of dictionaries
    """
    
    # Get all keys and values for the dictionary
    keys = [item for singleRun in runs for item in singleRun[0]]
    values = [singleRun[1] for singleRun in runs]
        
    
    # Generate all possible cases
    cases = list(product(*values)) # finds all combinations of the desired inputs; equivalent to list(itertols.product(*values)) in Python 2.6+
    
    # Understand which variables are modified together
    varList = [run[0] for run in runs]
    # Stores the index of varList where the key is found
    # keyLocation[i] = j means that keys[i] is in runs[j][0]
    keyLocation = [index for key in keys for (index,vars) in enumerate(varList) if key in vars]

    # Assemble the cases
    runOptions = []
    for case in cases:
        # ensure that fullCase[i] = case[keyLocation[i]] to enable the creation of the dictionary
        fullCase = [case[i] for i in keyLocation]
        runOptions.append( dict( zip(keys, fullCase) ) )
        
    return runOptions


def main():
    options = dict()
    options['viscous'] = 0
    options['dt'] = 1e-5
    options['order'] = 2
    options['plot_freq'] = 0
    options['Mach_free_stream'] = 0.1
    options['Mach_c_ic'] = options['Mach_free_stream']
    options['Re_free_stream'] = 200
    options['Re_c_ic'] = options['Re_free_stream']
    
    os.environ['HIFILES_HOME'] = '/home/mlopez14/HiFiLES-solver'
    
    caseName = 'new'
    homeFolder = os.environ['HIFILES_HOME']
    referenceFolder = os.path.join(homeFolder, 'bin', 'sense')
    
    referenceInputFile = os.path.join(homeFolder, 'testcases', 'navier-stokes', 'cylinder', 'input_cylinder_visc')
    

    caseName = 'new'
    newFolder = os.path.join(homeFolder, 'runs', caseName)
    
    os.system('mkdir -p ' + newFolder)
    
    newInputFile = os.path.join(newFolder, 'input_file_' + caseName)
    
    execName = os.path.join(homeFolder, 'bin', 'HiFiLES')
    
    mfileName = os.path.join(newFolder, 'mfile')
    
#     freeNodes = getFreeNodes(50)
    freeNodes = ['compute-0-0', 'compute-0-1']
    
#     printMachineFile(mfileName, freeNodes, 2)
#     
#     createInputFile(referenceInputFile, newInputFile, options)
#     
#     setupRunFolder(newFolder, execName, referenceFolder)



    runs = []
    # Create list of variable ranges desired
    runs.append( (['viscous'], [0]) )
    runs.append( (['dt'], [1e-5]) )
    runs.append( (['order'], [2]) )
    runs.append( (['plot_freq'], [0]) )
    runs.append( (['Mach_free_stream', 'Mach_c_ic'], [0.1, 0.2]) )
    runs.append( (['Re_free_stream', 'Re_free_stream'], [200, 300]) )

    runOptions = assembleRuns(runs)
    
    # Generate all input files necessary
    for (index, options) in enumerate(runOptions):
        newInputFile = os.path.join(newFolder, 'input_file_' + caseName + '_' + str(index))
        createInputFile(referenceInputFile, newInputFile, options)
    
    print runOptions


if __name__ == "__main__": 
    main()