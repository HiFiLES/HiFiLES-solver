#!/usr/bin/env python

import sys
import numpy
import pylab
import re
from math import log, sqrt, pi

def TGV_dissrate(tke,time,oldtke,oldtime,vol):
  ##### Calculate dissipation rate of TKE = -d(TKE)/dt
  dtkedt = -1.*(tke-oldtke)/(time-oldtime)/vol
  return dtkedt

def TGV_tkeplot(data):
  ##### Plot time series of Turbulent kinetic energy

  # Normalise by initial TKE
  tke_0 = data[0,1]

  plot1 = pylab.figure()
  pylab.title('Time series of Turbulent kinetic energy')
  pylab.xlabel('Time (s)')
  pylab.ylabel('Turbulent kinetic energy/TKE_0')
  pylab.plot(data[:,0], data[:,1]/tke_0, linestyle='solid')
  pylab.axis([min(data[:,0]),max(data[:,0]),0.0,1.05])
  pylab.axis([0,20,0,1.01])
  pylab.savefig('tke.pdf')
  return

def TGV_dissrateplot(data,dns,DG32,DG64,CDG48):
  ##### Plot time series of Turbulent kinetic energy
  plot1 = pylab.figure()
  pylab.title('Time series of TKE Dissipation Rate')
  pylab.xlabel('Time (s)')
  pylab.ylabel('Dissipation Rate')
  pylab.plot(data[:,0], data[:,2], linestyle='solid')
  pylab.plot(data[:,0], data[:,3], linestyle='solid')
  pylab.plot(CDG48[:,0], CDG48[:,1], linestyle='dotted',color='black')
  pylab.plot(DG32[:,0], DG32[:,1], linestyle='dashed',color='black')
  pylab.plot(DG64[:,0], DG64[:,1], linestyle='dashdot',color='black')
  pylab.plot(dns[:,0], dns[:,1], marker = 'o', markerfacecolor='none', markersize=6, markeredgecolor='black', linestyle='none')
  pylab.axis([0,20,-0.001,0.02])
  pylab.legend(('-dk/dt','vort','DG-48x4','DG-32x2','DG-64x2','DNS'), loc='best')
  pylab.savefig('dissrate.pdf')
  return

#########################################################################

def main():

    data = []
    vol = 8.*pi**3
    ##### Read in history file
    statfile = open('history.plt', 'r')

		# Skip header
    for i in range(3):
      statfile.readline()

    # Set tke at previous timestep
    index = 0

    # viscosity
    nu = 6.25E-04

    for line in statfile:
      time = float(line.split(', ')[14])
      tke = float(line.split(', ')[11]) # kinetic energy
      eps = float(line.split(', ')[12])/vol*2.*nu # vorticity-based dissipation rate

      # (Dissipation rate = nu*(|vorticity|/vol)**2
      # STRICTLY TRUE ONLY FOR INCOMPRESSIBLE FLOWS)

      # Calculate dissrate as time derivative of tke
      if (index == 0):
        oldtke = tke
        oldtime = -0.01
      dtkedt = TGV_dissrate(tke,time,oldtke,oldtime,vol)
      data.append((time, tke, dtkedt, eps))
      oldtime = data[index][0]
      oldtke = data[index][1]
      index += 1

    ##### Write to numpy data file
    data = numpy.array(data)
    numpy.save('dissrate', data)

    ##### Read in Beck's dissipation rate comparison data
    dns = []; DG32 = []; DG64 = []; DG128 = []
    beck = open('data/Beck-TGV-dissrate-DNS2-1600.dat', 'r')
    for line in beck:
      dns.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
    dns = numpy.array(dns)

    beck = open('data/Beck-TGV-dissrate-DG-32x2-1600.dat', 'r')
    for line in beck:
      DG32.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
    DG32 = numpy.array(DG32)

    beck = open('data/Beck-TGV-dissrate-DG-64x2-1600.dat', 'r')
    for line in beck:
      DG64.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
    DG64 = numpy.array(DG64)

    beck = open('data/Beck-TGV-dissrate-DG-128x2-1600.dat', 'r')
    for line in beck:
      DG128.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
    DG128 = numpy.array(DG128)

    ##### Read in Chapelier's dissipation rate comparison data
    CDG48 = []
    chap = open('data/Chapelier-TGV-dissrate-DG48x4-1600.dat', 'r')
    for line in chap:
      CDG48.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
    CDG48 = numpy.array(CDG48)
    for i in range(len(CDG48[:,1])):
      CDG48[i,1] *= (2.*nu)

    ##### Plot tke
    TGV_tkeplot(data)

    ##### Plot dissipation rates
    TGV_dissrateplot(data,dns,DG32,DG64,CDG48)


if __name__ == '__main__':
    main()

