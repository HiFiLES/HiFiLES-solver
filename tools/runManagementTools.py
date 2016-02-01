#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4  
'''
Tools for running HiFiLES cases automatically
'''

import os
from inputFileTools import createInputFile # used to create input files
from machinefileTools import printMachineFile, getFreeNodes


def setupRunFolder(curFolder, execName, referenceFolder):
# Creates a folder with name caseName containing:
# HiFiLES executable
# pre-processed geometry matrices (.HiFiLES.bin files) from a reference folder
    """   
    Inputs:
    caseName: string; name of the new folder where all the data needed for the run will be stored
    homeFolder: full path of main directory of HiFiLES; executable is assumbed to be in $homeFolder/bin
    referenceFolder: full path of folder that contains the pre-processed geometry files of a reference run: must have same mesh as current run
    options: dict; each key corresponds to a variable in the input file that will be modified with its corresponding value
    """
    
    # Copy HiFiLES executable to current folder
    if os.system('cp ' + execName + ' ' + curFolder) != 0:
        raise (execName + ' does not exist')
    
    # Copy binary matrix files for this case from the reference folder
    if os.system('cp ' + os.path.join(referenceFolder, '*.HiFiLES.bin' + ' ' + curFolder)) != 0:
        raise ('Binary matrix files could not be copied from ' + referenceFolder + ' to ' + curFolder)
    
    

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
    
    printMachineFile(mfileName, freeNodes, 2)
    
    createInputFile(referenceInputFile, newInputFile, options)
    
    setupRunFolder(newFolder, execName, referenceFolder)


if __name__ == "__main__": 
    main()