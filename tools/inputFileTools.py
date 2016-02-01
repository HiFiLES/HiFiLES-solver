#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4  
'''
Tools for generating a HiFiLES input file using the template input file values as defaults
'''


def getFileContents(fileToRead):
    """   
    Inputs:
    fileToRead: string; name of the file whose contents will be returned in an array of strings; one entry per line
    
    Outputs:
    fileContents: array of strings; each entry contains an entry of the fileToRead
    """
    # Open the reference configuration file
    file = open(fileToRead,'r')
    try:
        fileContents = [line for line in file]
    finally:
        file.close()
    
    return fileContents[:]


def writeFileContents(content, file):
    """   
    Inputs:
    file: string; name of the file in which contents will be written
    content: array of strings; same as that returned by getFileContents; one entry per line; 
            each line should end with character '\n' for it to be a line
    
    Outputs:
    creates a file with name of string contained in variable file
    """
    newFile = open(file,'w')
    try:
        newFile.write(''.join(content))
    finally:
        newFile.close()

def createInputFile(templateFile, newInputFile, options):
    """   
    Inputs:
    templateFile: string; name of the reference input file
    newInputFile: string; name of the input file to be created
    options: dictionary; each key-value pair corresponds to an input file variable to be modified
    """
    
    inputFileContents = getFileContents(templateFile)
    
    newInputFileContents = inputFileContents
    
    # Modify the contents
    currentLine = 0
    for line in inputFileContents:
        lineSplit = line.split('//', 1)
        variable = lineSplit[0].strip()

        # If it is indeed a variable that needs to be modified, update the new input file
        if variable in options:
            # modify the line below the variable
            newInputFileContents[currentLine + 1] = str(options[variable]) + "\n"
        currentLine += 1
    
    # Write the contents to a file
    writeFileContents(newInputFileContents, newInputFile)
        
        
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
    
    createInputFile("input_cylinder_visc", "case_1", options)


if __name__ == "__main__": 
    main()

