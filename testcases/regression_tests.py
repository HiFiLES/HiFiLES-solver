#!/usr/bin/env python

# \file parallel_regression.py
# \brief Python script for automated regression testing of HiFiLES examples
# \author - Original code: Aniket C. Aranake, Alejandro Campos, Thomas D. Economon.
#         - Current development: Aerospace Computing Laboratory (ACL) directed
#                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
# \version 1.0.0
#
# HiFiLES (High Fidelity Large Eddy Simulation).
# Copyright (C) 2013 Aerospace Computing Laboratory.

import sys,time, os, subprocess, datetime, signal, os.path, stat

class testcase:

  def __init__(self,tag_in):

    datestamp = time.strftime("%Y%m%d", time.gmtime())
    self.tag  = "%s_%s"%(tag_in,datestamp)  # Input, string tag that identifies this run

    # Configuration file path/filename
    self.cfg_dir  = "/home/fpalacios"
    self.cfg_file = "default.cfg"

    # The test condition. These must be set after initialization
    self.test_iter = 1
    self.test_vals = []  

    # These can be optionally varied 
    self.HiFiLES_dir     = "/home/fpalacios"
    self.HiFiLES_exec    = "default"
    self.mpi_cmd         = "mpirun -n 4"
    self.timeout         = 300
    self.tol             = 0.001
    self.outputdir       = "/home/fpalacios"

  def run_test(self):

    passed       = True
    exceed_tol   = False
    timed_out    = False
    iter_missing = True
    start_solver = True

    # Adjust the number of iterations in the config file   
    self.do_adjust_iter()

    # Assemble the shell command to run HiFiLES
    self.HiFiLES_exec = os.path.join("$HIFILES_RUN", self.HiFiLES_exec)
    command_base = "%s %s %s > outputfile"%(self.mpi_cmd, self.HiFiLES_exec, self.cfg_file)
    command      = "%s"%(command_base)

    # Run HiFiLES
    cur_dir = os.path.join('./',self.cfg_dir) 
    os.chdir(cur_dir)
    os.system('cp $HIFILES_HOME/bin/mfile .')
    start   = datetime.datetime.now()
    print("\nPath at terminal when executing this file")
    print(command)
    process = subprocess.Popen(command, shell=True)  # This line launches HiFiLES

    while process.poll() is None:
      time.sleep(0.1)
      now = datetime.datetime.now()
      if (now - start).seconds> self.timeout:
        try:
          process.kill()
          os.system('killall %s' % self.HiFiLES_exec)   # In case of parallel execution
        except AttributeError: # popen.kill apparently fails on some versions of subprocess... the killall command should take care of things!
          pass
        timed_out = True
        passed    = False

    # Examine the output
    f = open('outputfile','r')
    output = f.readlines()
    delta_vals = []
    sim_vals = []
    if not timed_out:
      start_solver = False
      for line in output:
        if not start_solver: # Don't bother parsing anything before --Setting initial conditions---
          if line.find('Setting initial conditions') > -1:
            start_solver=True
        else:   # Found the --Setting initial conditions--- line; parse the input
          raw_data = line.split()
          try:
            iter_number = int(raw_data[0])
            data        = raw_data[1:]    # Take the last 4 columns for comparison
          except ValueError:
            continue
          except IndexError:
            continue
         
          if iter_number == self.test_iter:  # Found the iteration number we're checking for
            iter_missing = False
            if not len(self.test_vals)==len(data):   # something went wrong... probably bad input
              print "Error in test_vals!"
              passed = False
              break
            for j in range(len(data)):
              sim_vals.append( float(data[j]) )
              delta_vals.append( abs(float(data[j])-self.test_vals[j]) )
              if delta_vals[j] > self.tol:
                exceed_tol = True
                passed     = False
            break
          else:
            iter_missing = True

      if not start_solver:
        passed = False
        
      if iter_missing:
        passed = False

    print '=========================================================\n'

    if passed:
      print "%s: PASSED"%self.tag
    else:
      print "%s: FAILED"%self.tag

    print 'execution command: %s'%command

    if timed_out:
      print 'ERROR: Execution timed out. timeout=%d'%self.timeout

    if exceed_tol:
      print 'ERROR: Difference between computed input and test_vals exceeded tolerance. TOL=%f'%self.tol

    if not start_solver:
      print 'ERROR: The code was not able to get to the "Begin solver" section.'

    if iter_missing:
      print 'ERROR: The iteration number %d could not be found.'%self.test_iter

    print 'test_iter=%d, test_vals: '%self.test_iter,
    for j in self.test_vals:
      print '%f '%j,
    print '\n',

    print 'sim_vals: ',
    for j in sim_vals:
      print '%f '%j,
    print '\n',
  
    print 'delta_vals: ',
    for j in delta_vals:
      print '%f '%j,
    print '\n'
    
    os.chdir('../../../')
    return passed

  def do_adjust_iter(self):
  
    # Read the cfg file
    self.cfg_file = os.path.join(os.environ['HIFILES_HOME'], self.cfg_dir, self.cfg_file)
    file_in = open(self.cfg_file, 'r')
    lines   = file_in.readlines()
    file_in.close()
  
    # Rewrite the file with a .autotest extension
    self.cfg_file = "%s.autotest"%self.cfg_file
    file_out = open(self.cfg_file,'w')
    for line in lines:
      if line.find("EXT_ITER")==-1:
        file_out.write(line)
      else:
        file_out.write("EXT_ITER=%d\n"%(self.test_iter+1))
    file_out.close()


def createConfigFile(configFileName, options, newConfigFileName, main_dir):
    """Updates the configuration file used to compile HiFiLES.
    
    Inputs:
    configFile: string; name of the reference configuration file
    options: dictionary; each key-value pair corresponds to a configuration file variable to be modified
    
    Output: newConfigFileContents: array of strings with the modified variables
    """
    
    # Open the reference configuration file
    configFile = open(os.path.join(main_dir, configFileName),'r')
    try:
        configFileContents = [line for line in configFile]
    finally:
	configFile.close()
    
    # Copy the contents of the default configuration file to the new file
    newConfigFileContents = configFileContents
    
    # Then modify the contents
    currentLine = 0
    for line in configFileContents:
        # For each line, figure out if it corresponds to a variable-value definition
        lineSplit = line.split('=', 1)
        # If the returned array has two elements, it is likely a variable-value pair
        if len(lineSplit) == 2:
            variable = lineSplit[0]
            # If it is indeed a variable that needs to be modified, update the new config File
            if variable in options:
                newConfigFileContents[currentLine] = \
                    variable + '=' + "\"" + options[variable] + "\"\n"
        currentLine += 1
    
    # Write the contents to a file
    newConfigFile = open(os.path.join(main_dir, newConfigFileName),'w')
    try:
        newConfigFile.write(''.join(newConfigFileContents))
	os.chmod(newConfigFile.name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
    finally:
	newConfigFile.close()

def main():
    '''This program runs HiFiLES and ensures that the output matches specified values. This will be used to do nightly checks to make sure nothing is broken. '''

    # Build HiFiLES_CFD using a working configure_run.sh script as reference
    # For debugging purposes, uncomment the two lines below to hard-code the environment variables
    # HIFILES_HOME (location of the main git directory) and HIFILES_RUN (location where the HiFiLES executable is
    # os.environ['HIFILES_HOME'] = '/home/mlopez14/HiFiLES-solver'
    os.environ['HIFILES_RUN'] = os.environ['HIFILES_HOME'] + '/bin' 
  
    # Name of the environment variable where the path to HiFiLES is stored
    hifilesLocationVar = 'HIFILES_HOME'
    # Name of the reference configuration script (path to MPI, CUDA, etc. must be set in the script below)
    configFileName = 'configure_run.sh'

    try:
        main_dir = os.environ[hifilesLocationVar]
    except:
        print('Error: ' + hifilesLocationVar + ' is not an environment variable')
        return
    # Go to HiFiLES's main directory
    os.chdir(os.environ['HIFILES_HOME'])

    # Loop through all possible configurations: serial or not, CPU or GPU
    confiFileOptions = dict()
    testReport = dict() # Where the results of whether or not a test passes will be recorded
    
    for platform in ["GPU","CPU"]: 
        for parallel in ["YES","NO"]:
            
            # Remove the HiFiLES binary and previous makefiles
            os.system('rm ' + os.environ['HIFILES_RUN'] + '/HiFiLES')
            os.system('rm Makefile')
            
            testName = platform + "_Parralel_" + parallel
            testResults = [] # This array will store a 1 or 0 for every test
            confiFileOptions['NODE'] = platform
            confiFileOptions['PARALLEL'] = parallel
            
            # Specify the name of the new configuration file
            newConfigFileName = "NODE-" + confiFileOptions['NODE'] + "_" + "PARALLEL-" + \
                                 confiFileOptions['PARALLEL'] + "-configFile.sh"
            print(newConfigFileName)
                # Modify the reference configuration file
            createConfigFile(configFileName, confiFileOptions, newConfigFileName, main_dir)

            # Execute the configuration script
            os.system("cd " + main_dir + "; bash " + newConfigFileName)
            
            if not os.path.exists("Makefile"):
                print('Could not configure HiFiLES in NODE: ' + platform + ' and PARALLEL: ' + parallel)
                continue
            
            # Make the executable
            os.system("cd " + main_dir + "; make clean; make -j")
            #os.system("cd " + main_dir + "; make -j")
            
            if not os.path.exists(os.environ['HIFILES_RUN'] + "/HiFiLES"):
                print('Could not build HiFiLES in NODE: ' + platform + ' and PARALLEL: ' + parallel)
                continue

            if parallel == "NO":
                mpi_command = ""
            else:
                mpi_command = "mpirun -n 4 -machinefile mfile"
                
   ##########################
   ###  Compressible N-S  ###
   ##########################

            # Cylinder
            cylinder              = testcase('cylinder')
            cylinder.cfg_dir      = "testcases/navier-stokes/cylinder"
            cylinder.cfg_file     = "input_cylinder_visc"
            cylinder.test_iter    = 25
            cylinder.test_vals    = [0.180251,  1.152697,  0.270985,  10.072776,  17.702310,  -0.097602]
            cylinder.HiFiLES_exec = "HiFiLES"
            cylinder.timeout      = 1600
            cylinder.tol          = 0.00001
            cylinder.mpi_cmd      = mpi_command;
            testResults.append( cylinder.run_test() )
   
            # Taylor-Green vortex
            tgv              = testcase('tgv')
            tgv.cfg_dir      = "testcases/navier-stokes/Taylor_Green_vortex"
            tgv.cfg_file     = "input_TGV_SD_hex"
            tgv.test_iter    = 25
            tgv.test_vals    = [0.00013215,0.05076817,0.05076814,0.06456282,0.07476870,0.00000000,0.00000000,0.00000000]
            tgv.HiFiLES_exec = "HiFiLES"
            tgv.timeout      = 1600
            tgv.tol          = 0.00001
            tgv.mpi_cmd      = mpi_command;
            testResults.append( tgv.run_test() )
            
            # Store the test results
            testReport[testName] = testResults

    print(testReport)
        
if __name__=="__main__":
    main()

