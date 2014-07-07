/*!
 * \file solution.h
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 * \version 0.1.0
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 * Copyright (C) 2014 Aerospace Computing Laboratory (ACL).
 *
 * HiFiLES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HiFiLES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HiFiLES.  If not, see <http://www.gnu.org/licenses/>.
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
  int num_cells_global;
  
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

  /*! Diagnostic output quantities. */
  
  array<double> body_force;
  array<double> inv_force;
  array<double> vis_force;
  array<double> norm_residual;
  array<double> integral_quantities;
  double coeff_lift;
  double coeff_drag;

  /*! Plotting resolution. */
  
  int p_res;
  
#ifdef _MPI
  
  int nproc;
  
  int n_mpi_inter_types;
  array<mpi_inters> mesh_mpi_inters;
  array<int> error_states;
  
  int n_mpi_inters;
    
  /*! No-slip wall flux point coordinates for wall models. */

	array< array<double> > loc_noslip_bdy_global;

#endif
  
};
