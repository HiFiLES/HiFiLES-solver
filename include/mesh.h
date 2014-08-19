/*!
 * \file mesh.h
 * \brief  - Class to control mesh-related activities (motion, adaptation, etc.)
 * \author - Current development: Aerospace Computing Laboratory (ACL) directed
 *                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
 * \version 1.0.0
 *
 * HiFiLES (High Fidelity Large Eddy Simulation).
 * Copyright (C) 2013 Aerospace Computing Laboratory.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <map>
#include <float.h>
#include <map>
#include <set>

#include "global.h"
#include "input.h"
#include "error.h"
#include "array.h"
#include "eles.h"
#include "solution.h"
#include "funcs.h"
#include "matrix_structure.hpp"
#include "vector_structure.hpp"
#include "linear_solvers_structure.hpp"

#ifdef _TECIO
#include "TECIO.h"
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#ifdef _GPU
#include "../include/util.h"
#endif

class mesh
{
public:
  // #### constructors ####

  /** default constructor */
  mesh(void);

  /** default destructor */
  ~mesh(void);

  // #### methods ####

  /** Mesh motion wrapper */
  void move(int _iter, int in_rk_step, int n_rk_steps);

  /** peform prescribed mesh motion using linear elasticity method */
  void deform(void);

  /** peform prescribed mesh motion using rigid translation/rotation */
  void rigid_move(void);

  /** Perturb the mesh points (test case for freestream preservation) */
  void perturb(void);

  /** update grid velocity & apply to eles */
  void set_grid_velocity(solution *FlowSol, double dt);

  /** update the mesh: re-set spts, transforms, etc. */
  void update(void);

  /** setup information for boundary motion */
  //void setup_boundaries(array<int> bctype);

  /** write out mesh to file */
  void write_mesh(double sim_time);

  /** write out mesh in Gambit .neu format */
  void write_mesh_gambit(double sim_time);

  /** write out mesh in Gmsh .msh format */
  void write_mesh_gmsh(double sim_time);

  // #### members ####

  /** Basic parameters of mesh */
  //unsigned long
  int n_eles, n_verts, n_dims, n_verts_global, n_cells_global;
  int iter;

  /** arrays which define the basic mesh geometry */
  array<double> xv_0;//, xv;
  array< array<double> > xv;
  array<int> c2v,c2n_v,ctype,bctype_c,ic2icg,iv2ivg,ic2loc_c,
  f2c,f2loc_f,c2f,c2e,f2v,f2n_v,e2v,v2n_e,v2c,v2n_c,v2ctype,v2spt;
  array<array<int> > v2e;

  /** #### Boundary information #### */

  int n_bnds, n_moving_bnds, n_faces;
  array<int> nBndPts;
  array<int> v2bc;

  /** vertex id = boundpts(bc_id)(ivert) */
  array<array<int> > boundPts;

  /** Global cell ID & local face ID of boundary cells */
  array<array<int> > bccells, bcfaces;

  /*! number of cells/faces on boundary */
  array<int> bc_ncells;

  /** Store motion flag for each boundary
     (currently 0=fixed, 1=moving, -1=volume) */
  array<int> bound_flags;

  /** HiFiLES 'bcflag' for each boundary */
  array<int> bc_list;

  /** replacing get_bc_num() from geometry.cpp */
  map<string,int> bc_num;

  /** inverse of bc_name */
  map<int,string> bc_string;

  // nBndPts.setup(n_bnds); boundPts.setup(nBnds,nPtsPerBnd);

  array<double> vel_old,vel_new, xv_new;

  array< array<double> > grid_vel;

  void setup(solution *in_FlowSol, array<double> &in_xv, array<int> &in_c2v, array<int> &in_c2n_v, array<int> &in_iv2ivg, array<int> &in_ctype);

private:

  // ---- Variables ----

  bool start;
  array<double> xv_nm1, xv_nm2, xv_nm3;//, xv_new, vel_old, vel_new;

  /** Global stiffness matrix for linear elasticity solution */
  CSysMatrix StiffnessMatrix;
  CSysVector LinSysRes;
  CSysVector LinSysSol;

  /// Copy of the pointer to the Flow Solution structure
  struct solution *FlowSol;

  /** global stiffness psuedo-matrix for linear-elasticity mesh motion */
  array<array<double> > stiff_mat;

  unsigned long LinSolIters;
  int failedIts;
  double min_vol, min_length, solver_tolerance;
  double time, rk_time;
  int rk_step;

  /*! array of input parameters to control motion of all boundaries in mesh */
  array< array<double> > motion_params;

  array<double> displacement;

  // Coefficients for LS-RK45 time-stepping
  array<double> RK_a, RK_b, RK_c;

  // ---- Methods ----

  /** meant to check for any inverted cells (I think) and return minimum volume */
  double check_grid(solution *FlowSol);

  /** find minimum length in mesh */
  void set_min_length(void);

  /*! setup displacement array for all mesh vertices & initialize to 0 */
  void initialize_displacement();

  /** Set given/known displacements of vertices on moving boundaries in linear system */
  void set_boundary_displacements_eles(void);

  /*! Apply the boundary displacements to the flux points on the boundaries */
  void apply_boundary_displacements_fpts(void);

  /*! update displacements of mesh nodes using values computed from linear-elasticity solution in each ele */
  void update_displacements(void);

  /*! using the final displacements, update the list of vertex positions */
  void update_mesh_nodes(void);
};
