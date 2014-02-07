/*!
 * \file solution.h
 * \brief _____________________________
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL) directed
 *                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
 * \version 1.0.0
 *
 * HiFiLES (High Fidelity Large Eddy Simulation).
 * Copyright (C) 2013 Aerospace Computing Laboratory.
 */

#pragma once

#include "array.h"
#include <string>
#include "input.h"
#include "eles.h"
#include "eles_tris.h"
#include "eles_quads.h"
#include "eles_hexas.h"
#include "eles_tets.h"
#include "eles_pris.h"
#include "int_inters.h"
#include "bdy_inters.h"

#ifdef _MPI
#include "mpi.h"
#include "mpi_inters.h"
#endif

class int_inters; /*!< Forwards declaration */
class bdy_inters; /*!< Forwards declaration */
class mpi_inters; /*!< Forwards declaration */

struct solution {
		
	int viscous;
  double time;
  double ene_hist;
  double grad_ene_hist;
  
  array<int> num_f_per_c;
  
	int n_ele_types;
  int n_dims;
  
  int num_eles;
  int num_verts;
  int num_edges;
  int num_inters;
  
	int n_steps;
	int adv_type;
	int plot_freq;
	int restart_dump_freq;
  int ini_iter;
	
	int write_type;
	
	array<eles*> mesh_eles;
	eles_quads mesh_eles_quads;
	eles_tris mesh_eles_tris;
	eles_hexas mesh_eles_hexas;
	eles_tets mesh_eles_tets;
	eles_pris mesh_eles_pris;
	
	int n_int_inter_types;
	int n_bdy_inter_types;
	
	array<int_inters> mesh_int_inters;
	array<bdy_inters> mesh_bdy_inters;
  
  int rank;

  /*! No-slip wall flux point coordinates for wall models. */

  array< array<double> > loc_noslip_bdy;
  
  /*! Diagnostics. */
  
	array<double> body_force;
  
  /*! Plotting related. */
  
  int p_res;
  int num_pnodes;
  
  array<int> ele2vert, ele2n_vert, ele_type;
  array<int> ele2face, ele2edge;
  array<int> inter2loc_inter, inter2ele;
  
  array< array<double> >  pos_pnode;
  array<double> plotq_pnodes;
  array<int> factor_pnode;
  
  array<int> c2ctype_c;
  
#ifdef _MPI
  
  int nproc;
  
	int n_mpi_inter_types;
	array<mpi_inters> mesh_mpi_inters;
  array<int> error_states;
  
  int n_mpi_inters;
  int n_mpi_pnodes;
  array<int> inter_mpi2inter;
  
  array<int> mpi_pnode2pnode;
  array<int> mpi_pnodes_part;
  
  array<double> out_buffer_plotq, in_buffer_plotq;
  array<int> out_buffer_pnode, in_buffer_pnode;

  array< array<double> > loc_noslip_bdy_global;
  
#endif
  
};
