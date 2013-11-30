#!/usr/bin/env python

import sys
import numpy
import pylab
from math import pi

#################################################

##### Read statfile

def TGV_readstatfile(statfile):

# Skip header
  statfile.readline()

  vol = 8.*pi**3
  nu = 6.25E-04
  index = 0
  data = []

  for line in statfile:
    time = float(line.split(' ')[0])
    tke = float(line.split(' ')[1]) # kinetic energy

    # Calculate dissrate as time derivative of tke

    # At time = 0, set values at 'previous' timestep
    if (index == 0):
      oldtke = tke
      oldtime = -0.01

    # First-order upwind approximation to time derivative
    dtkedt = -1.*derivative(tke,time,oldtke,oldtime,vol)
    # Append data to output array
    data.append((time, tke, dtkedt))
    # Set 'old' values for next timestep
    oldtime = data[index][0]
    oldtke = data[index][1]
    index += 1

  # End of statfile

  # Convert data to a more usable format
  data = numpy.array(data)

  return data

#################################################

##### Calculate time derivative of TKE

def derivative(tke,time,oldtke,oldtime,vol):

  dtkedt = (tke-oldtke)/(time-oldtime)/vol

  return dtkedt

#################################################

##### Read comparison data

def TGV_readcompdata():

  # Read in Debonis's TKE comparison data
  bb13 = []
  deb = open('data/Debonis-TGV-tke-DNS-1600.dat', 'r')
  for line in deb:
    bb13.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
  bb13 = numpy.array(bb13)

  # Read in Beck's dissipation rate comparison data
  dns = []
  beck = open('data/Beck-TGV-dissrate-DNS2-1600.dat', 'r')
  for line in beck:
    dns.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
  dns = numpy.array(dns)

  DG644 = []
  beck = open('data/Beck-TGV-dissrate-DG-64x4-1600.dat', 'r')
  for line in beck:
    DG644.append((float(line.split('  ')[0]), float(line.split('  ')[-1])))
  DG644 = numpy.array(DG644)

  return dns, bb13, DG644

#################################################

##### Plot time series of Turbulent kinetic energy

def TGV_tkeplot(data,bb13):

  # Normalise by initial TKE
  tke_0 = data[0,1]
  bb13_0 = bb13[0,1]

  plot1 = pylab.figure()
  pylab.title('Time series of Turbulent kinetic energy')
  pylab.xlabel('Time (s)')
  pylab.ylabel('Turbulent kinetic energy/TKE_0')
  pylab.plot(data[:,0], data[:,1]/tke_0, linestyle='solid')
  pylab.plot(bb13[:,0], bb13[:,1]/bb13_0, marker = 'o', markerfacecolor='none', markersize=6, markeredgecolor='black', linestyle='none')
  pylab.axis([min(data[:,0]),max(data[:,0]),0.0,1.05])
  pylab.axis([0,20,0,1.03])
  pylab.legend(('SD-16x4','DNS'), loc='best')
  leg = pylab.gca().get_legend()
  ltext=leg.get_texts()
  pylab.setp(ltext,fontsize=14,color='black')
  leg.get_frame().set_edgecolor('white')
  leg.get_frame().set_alpha(0)
  pylab.savefig('tke.pdf')
  return

#################################################

##### Plot time series of Turbulent kinetic energy

def TGV_dissrateplot(data,dns,DG644):

  plot1 = pylab.figure()
  pylab.title('Time series of TKE Dissipation Rate')
  pylab.xlabel('Time (s)')
  pylab.ylabel('Dissipation Rate')
  pylab.plot(data[:,0], data[:,2], linestyle='solid')
  pylab.plot(DG644[:,0], DG644[:,1], linestyle='dashed',color='black')
  pylab.plot(dns[:,0], dns[:,1], marker = 'o', markerfacecolor='none', markersize=6, markeredgecolor='black', linestyle='none')
  pylab.axis([0,20,0,0.015])
  pylab.legend(('SD-16x4','Beck-DG-64x4','DNS'), loc='best')
  leg = pylab.gca().get_legend()
  ltext=leg.get_texts()
  pylab.setp(ltext,fontsize=14,color='black')
  leg.get_frame().set_edgecolor('white')
  leg.get_frame().set_alpha(0)
  pylab.savefig('dissrate.pdf')

  return

#################################################

def main():

    ##### Read in statfile
    statfile = open('statfile.dat', 'r')
    data = TGV_readstatfile(statfile)

    # Read comparison data
    dns, bb13, DG644 = TGV_readcompdata()

    # Plot TKE and dissipation rate
    TGV_tkeplot(data,bb13)
    TGV_dissrateplot(data,dns,DG644)

if __name__ == '__main__':
    main()

