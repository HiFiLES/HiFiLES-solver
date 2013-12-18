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

import sys,time, os, subprocess, datetime, signal, os.path

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
    self.timeout     = 300
    self.tol         = 0.001
    self.outputdir   = "/home/fpalacios"

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
    command_base = "%s %s > outputfile"%(self.HiFiLES_exec, self.cfg_file)
    command      = "%s"%(command_base)

    # Run HiFiLES
    os.chdir(os.path.join('./',self.cfg_dir)) 
    start   = datetime.datetime.now()
    print("Path at terminal when executing this file")
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

if __name__=="__main__":
  '''This program runs HiFiLES and ensures that the output matches specified values. This will be used to do nightly checks to make sure nothing is broken. '''

  # Build HiFiLES_CFD in serial using the right Makefile.in
  os.system('make clean')
  os.system('make')
  
  os.chdir(os.environ['HIFILES_RUN'])
  if not os.path.exists("./HiFiLES"):
    print 'Could not build HiFiLES'
    sys.exit(1)

  os.chdir(os.environ['HIFILES_HOME'])

  ##########################
  ###  Compressible N-S  ###
  ##########################

  # Laminar flat plate
  cylinder              = testcase('cylinder')
  cylinder.cfg_dir      = "testcases/navier-stokes/cylinder"
  cylinder.cfg_file     = "input_cylinder_visc"
  cylinder.test_iter    = 25
  cylinder.test_vals    = [0.19358277,1.35384226,0.20833461,10.48882915]
  cylinder.HiFiLES_exec = "HiFiLES"
  cylinder.timeout      = 1600
  cylinder.tol          = 0.00001
  passed1               = cylinder.run_test()
  
  # Taylor-Green vortex
  tgv              = testcase('tgv')
  tgv.cfg_dir      = cylinder.HiFiLES_dir+"/testcases/navier-stokes/Taylor_Green_vortex"
  tgv.cfg_file     = "input_TGV_SD_hex"
  tgv.test_iter    = 30
  tgv.test_vals    = [0.00017975,0.05079293,0.05079307,0.06456811,0.05711717]
  tgv.HiFiLES_exec = "HiFiLES"
  tgv.timeout      = 1600
  tgv.tol          = 0.00001
  passed1          = tgv.run_test()

  if (passed1):
    sys.exit(0)
  else:
    sys.exit(1)

