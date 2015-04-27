/*!
 * \file inters.h
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

#include "inters.h"
#include "Array.h"

#ifdef _MPI
#include "mpi.h"
#endif

class inters
{
public:

  // #### constructors ####

  // default constructor

  inters();

  // default destructor

  ~inters();

  // #### methods ####

  /*! setup inters */
  void setup_inters(int in_n_inters, int in_inter_type);

  /*! Set normal flux to be normal * f_r */
  void right_flux(Array<double> &f_r, Array<double> &norm, Array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Rusanov flux */
  void rusanov_flux(Array<double> &u_l, Array<double> &u_r, Array<double> &v_g, Array<double> &f_l, Array<double> &f_r, Array<double> &norm, Array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Roe flux */
  void roe_flux(Array<double> &u_l, Array<double> &u_r, Array<double> &v_g, Array<double> &norm, Array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Lax-Friedrich flux (works only for wave equation) */
  void lax_friedrich(Array<double> &u_l, Array<double> &u_r, Array<double> &norm, Array<double> &fn, int n_dims, int n_fields, double lambda, Array<double>& wave_speed);

  /*! Compute common viscous flux using LDG formulation */
  void ldg_flux(int flux_spec, Array<double> &u_l, Array<double> &u_r, Array<double> &f_l, Array<double> &f_r, Array<double> &norm, Array<double> &fn, int n_dims, int n_fields, double tau, double pen_fact);

  /*! Compute common solution using LDG formulation */
  void ldg_solution(int flux_spec, Array<double> &u_l, Array<double> &u_r, Array<double> &u_c, double pen_fact, Array<double>& norm);

	/*! get look up table for flux point connectivity based on rotation tag */
	void get_lut(int in_rot_tag);
	
  /*! Compute common flux at boundaries using convective flux formulation */
  void convective_flux_boundary(Array<double> &f_l, Array<double> &f_r, Array<double> &norm, Array<double> &fn, int n_dims, int n_fields);

	protected:

	// #### members ####

	int inters_type; // segment, quad or tri

	int order;
	int viscous;
	int LES;
  int wall_model;
	int n_inters;
	int n_fpts_per_inter;
	int n_fields;
	int n_dims;
  int motion;       //!< Mesh motion flag
	
	Array<double*> disu_fpts_l;
	Array<double*> delta_disu_fpts_l;
	Array<double*> norm_tconf_fpts_l;
	//Array<double*> norm_tconvisf_fpts_l;
	Array<double*> detjac_fpts_l;
	Array<double*> tdA_fpts_l;
	Array<double*> norm_fpts;
	Array<double*> pos_fpts;
  Array<double*> pos_dyn_fpts;

  Array<double> pos_disu_fpts_l;
  Array<double*> grad_disu_fpts_l;
  Array<double*> normal_disu_fpts_l;

  Array<double> temp_u_l;
  Array<double> temp_u_r;

  Array<double> temp_grad_u_l;
  Array<double> temp_grad_u_r;

  Array<double> temp_normal_u_l;

  Array<double> temp_pos_u_l;

  Array<double> temp_f_l;
  Array<double> temp_f_r;

  Array<double> temp_fn_l;
  Array<double> temp_fn_r;

  Array<double> temp_f;

  Array<double> temp_loc;

	// LES and wall model quantities
	Array<double*> sgsf_fpts_l;
	Array<double*> sgsf_fpts_r;
	Array<double> temp_sgsf_l;
	Array<double> temp_sgsf_r;

  Array<int> lut;

  Array<double> v_l, v_r, um, du;

  // Dynamic grid variables:
  // Note: grid velocity is continuous across interfaces
  Array<double*> ndA_dyn_fpts_l;
  Array<double*> norm_dyn_fpts;
  Array<double*> J_dyn_fpts_l;
  Array<double*> grid_vel_fpts;
  Array<double*> disu_GCL_fpts_l;
  Array<double*> norm_tconf_GCL_fpts_l;

  double temp_u_GCL_l;
  double temp_f_GCL_l;

  Array<double> temp_v;
  Array<double> temp_fn_ref_l;
  Array<double> temp_fn_ref_r;
};
