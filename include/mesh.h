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
  void move(int _iter, int time_level, solution *FlowSol);

  /** peform prescribed mesh motion using linear elasticity method */
  void deform(solution* FlowSol);

  /** peform prescribed mesh motion using rigid translation/rotation */
  void rigid_move(solution *FlowSol);

  /** Perturb the mesh points (test case for freestream preservation) */
  void perturb(solution *FlowSol);

  /** update grid velocity & apply to eles */
  void set_grid_velocity(solution *FlowSol, double dt);

  /** update the mesh: re-set spts, transforms, etc. */
  void update(solution *FlowSol);

  /** setup information for boundary motion */
  //void setup_boundaries(array<int> bctype);

  /** write out mesh to file */
  void write_mesh(int mesh_type, double sim_time);

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
  array<double> xv_0, xv,;
  array<int> c2v,c2n_v,ctype,bctype_c,ic2icg,iv2ivg,ic2loc_c,
  f2c,f2loc_f,c2f,c2e,f2v,f2n_v,e2v,v2n_e;
  array<array<int> > v2e;

  /** #### Boundary information #### */

  int n_bnds, n_faces;
  array<int> nBndPts;
  array<int> v2bc;

  /** vertex id = boundpts(bc_id)(ivert) */
  array<array<int> > boundPts;

  /** Store motion flag for each boundary
     (currently 0=fixed, 1=moving, -1=volume) */
  array<int> bound_flags;

  /** HiFiLES 'bcflag' for each boundary */
  array<int> bc_list;

  /** replacing get_bc_name() from geometry.cpp */
  map<string,int> bc_name;

  /** inverse of bc_name */
  map<int,string> bc_flag;

  // nBndPts.setup(n_bnds); boundPts.setup(nBnds,nPtsPerBnd);

  array<double> vel_old, vel_new, xv_new;
private:

  array<double> xv_nm1, xv_nm2, xv_nm3;//, xv_new, vel_old, vel_new;

  /** Global stiffness matrix for linear elasticity solution */
  CSysMatrix StiffnessMatrix;
  CSysVector LinSysRes;
  CSysVector LinSysSol;

  /** global stiffness psuedo-matrix for linear-elasticity mesh motion */
  array<array<double> > stiff_mat;

  unsigned long LinSolIters;
  int failedIts;
  double min_vol, min_length, solver_tolerance;
  double time;

  /** create individual-element stiffness matrix - triangles */
  bool set_2D_StiffMat_ele_tri(array<double> &stiffMat_ele,int ele_id);

  /** create individual-element stiffness matrix - quadrilaterals */
  bool set_2D_StiffMat_ele_quad(array<double> &stiffMat_ele,int ele_id);

  /** create individual-element stiffness matrix - tetrahedrons */
  //bool set_2D_StiffMat_ele_tet(array<double> &stiffMat_ele,int ele_id, solution *FlowSol);

  /** create individual-element stiffness matrix - hexahedrons */
  //bool set_2D_StiffMat_ele_hex(array<double> &stiffMat_ele,int ele_id, solution *FlowSol);

  /**
     * transfrom single-element stiffness matrix to nodal contributions in order to
     * add to global stiffness matrix
     */
  void add_StiffMat_EleTri(array<double> StiffMatrix_Elem, int id_pt_0,
                           int id_pt_1, int id_pt_2);


  void add_StiffMat_EleQuad(array<double> StiffMatrix_Elem, int id_pt_0,
                            int id_pt_1, int id_pt_2, int id_pt_3);

  /** Set given/known displacements of vertices on moving boundaries in linear system */
  void set_boundary_displacements(solution *FlowSol);

  /** meant to check for any inverted cells (I think) and return minimum volume */
  double check_grid(solution *FlowSol);

  /** transfer solution from LinSysSol to xv_new */
  void update_grid_coords(void);

  /** find minimum length in mesh */
  void set_min_length(void);
};
