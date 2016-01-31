#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4  
'''
Generates a HiFiLES input file using the template input file values as defaults
'''
import os, subprocess, decimal


def createInputFile(templateFile, newInputFile, options):
    """   
    Inputs:
    templateFile: string; name of the reference input file
    options: dictionary; each key-value pair corresponds to an input file variable to be modified
    """
    
    # Open the reference configuration file
    inputFile = open(os.path.join(main_dir, configFileName),'r')
    try:
        inputFileContents = [line for line in inputFile]
    finally:
        inputFile.close()
    
    # Copy the contents of the default configuration file to the new file
    newInputFileContents = inputFileContents
    
    # Then modify the contents
    currentLine = 0
    for line in inputFileContents:
        # For each line, figure out if it corresponds to a variable-value definition
        lineSplit = line.split('=', 1)
        # If the returned array has two elements, it is likely a variable-value pair
        if len(lineSplit) == 2:
            variable = lineSplit[0]
            # If it is indeed a variable that needs to be modified, update the new config File
            if variable in options:
                newInputFileContents[currentLine] = \
                    variable + '=' + "\"" + options[variable] + "\"\n"
        currentLine += 1
    
    # Write the contents to a file
    newConfigFile = open(os.path.join(main_dir, newConfigFileName),'w')
    try:
        newConfigFile.write(''.join(newInputFileContents))
        os.chmod(newConfigFile.name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
    finally:
        newConfigFile.close()

