Instructions for running Navier-Stokes tests:

-----------------------------------------------------

1. Cylinder



-----------------------------------------------------

2. Taylor-Green vortex at Re=1600

This test has an initial condition of the form

u = sin(x)cos(y)cos(z),
v = -cos(x)sin(y)cos(z),
w = 0,
p = p_0 + rho/16*(cos(2x) + cos(2y)))(cos(2z) + 2),
e = p/(gamma-1) + 0.5*rho*(u^2 + v^2 + w^2),

In a triply periodic box of edge length 2 pi.
The initial condition is an analytical solution of the inviscid Navier-Stokes equations
which spontaneously decays into a quasi-realistic turbulent energy cascade
The simulation is set to run for 20 seconds which allows for the development of a wide range of scales.

The integral of kinetic energy is computed during the run and output every 10 timesteps to
an ASCII file called statfile.dat. The Python script plotstats.py reads in this file and plots two graphs:
(1), the evolution of the volume-averaged kinetic energy vs time, and
(2), the dissipation rate of volume-averaged kinetic energy, given by the negative time derivative of (1).
(1) includes reference DNS data from Debonis (2013).
(2) includes DNS data and data from a Discontinuous Galerkin simulation, both from Beck and Gassner (2012).

The simulation is performed on a coarse mesh of 16x16x16 hexahedral elements at 4th order accuracy.
At this level of resolution, the match to the reference data is not very good.
If you wish to obtain a much better fit to the data, try running the test using the 32x32x32 mesh provided,
but this will require 8 times more computational time, so run it in parallel (at least 4 processors recommended).

TODO: If Python is not installed on your system, write an equivalent Matlab script.

-----------------------------------------------------

3. High Order Workshop Problem C1.4. Laminar Boundary Layer on a Flat Plate
(see http://www.as.dlr.de/hiocfd/case_c1.4.html)

Flow Conditions:
  Mach Number: 0.5
  Angle of Attack: 0.0
  Reynolds Number: 1000000
  Dynamic Viscosity: Constant
  
Initial Condition:
  Uniform Flow
  
Meshes:
Mesh    Elements        dy_min    Growth rate
  a0         140        7.5E-4          2.251
  a1         560       3.75E-4          1.481
  a2        2240      1.875E-4          1.213
  a3        8960      9.375E-5          1.101
  a4       35840     4.6875E-5          1.049
  a5      143360    2.34375E-5          1.024
  
List of working cases (using Discontinous Galerkin):
Mesh    Polynomial Order        dt
  a2                   3    2.0E-6
  a2                   4    1.0E-6
  a4                   3    2.0E-7
  
Notes:
  Use provided input file: input_flatplate_a
