HiFiLES: High Fidelity Large Eddy Simulation
=======

At the Aerospace Computing Laboratory we believe that high-order numerical schemes have the potential to advance CFD beyond the current plateau of second-order methods and RANS turbulence modeling, ushering in new levels of accuracy and computational efficiency in turbulent flow simulations. HiFiLES (High Fidelity Large Eddy Simulation) is released as a freely available tool to unify the research community, promoting the advancement and wider adoption of high-order methods. The code is designed as an ideal base for further development on a variety of architectures.


HiFiLES Introduction
=======

High-order numerical methods for flow simulations capture complex phenomena like vortices and separation regions using fewer degrees of freedom than their low-order counterparts. The High Fidelity (HiFi) provided by the schemes, combined with turbulence models for small scales and wall interactions, gives rise to a powerful Large Eddy Simulation (LES) software package. HiFiLES is an open-source, high-order, compressible flow solver for unstructured grids built from the ground up to take full advantage of parallel computing architectures. It is specially well-suited for Graphical Processing Unit (GPU) architectures. HiFiLES is written in C++. The code uses the MPI protocol to run on multiple processors, and CUDA to harness GPU performance.

HiFiLES Dev. Ver. 0.1.0 Beta contains the following capabilities:

	- High-order compressible Navier-Stokes and Euler equations solver in 2D and 3D with support for triangular, quadratic, hexahedral, prismatic, and tetrahedral elements. Implementation for spatial orders of accuracy 2 through 4 have been verified.
	- Numerical scheme: Energy-Stable Flux Reconstruction.
	- Time advancement: explicit time-stepping with low-storage RK45 method (4th order) or forward Euler (1st order). Local time-stepping when running on CPUs.
	- Boundary conditions: Wall: no-slip isothermal, no-slip adiabatic, and symmetry (slip wall). Inflow and outflow: characteristic, supersonic, subsonic. Periodic.
	- High-order surface representation.
	- Mesh format compatibility: neutral (.neu) and Gmsh (.msh).
	- Large Eddy Simulation: Sub-grid Scale Models: Smagorinsky, WALE, similarity, and combinations of these. Wall models: log-law, three-layer Breuer-Rodi.
	- Parallelization: MPI, and GPU (strong scalability 88% of ideal for up to 16 GPUs; weak scalability above 90% of ideal for up to 16 GPUs)


HiFiLES Installation
=======






HiFiLES Developers
=======

Antony Jameson
Thomas V. Jones Professor of Engineering, Department of Aeronautics & Astronautics. Stanford University.

Francisco Palacios
Engineering Research Associate, Department of Aeronautics & Astronautics. Stanford University.

Jonathan Bull
Postdoc Scholar, Department of Aeronautics & Astronautics. Stanford University.

Kartikey Asthana, Thomas Economon, Manuel Lopez, David Manosalvas, Joshua Romero, Abhishek Sheshadri, Jerry Watkins
Ph.D. Candidates (developers in alphabetical order), Department of Aeronautics & Astronautics. Stanford University.

Past Developers:
Patrice Castonguay, Antony Jameson, Peter Vincent, David Williams.