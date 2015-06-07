/*!
 * \file eles_hexas.h
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

#include "eles.h"
#include "array.h"

class eles_hexas: public eles
{	
public:

  // #### constructors ####

  // default constructor

  eles_hexas();

  // #### methods ####

  /*! set shape */
  //void set_shape(array<int> &in_n_spts_per_ele);

  void set_connectivity_plot();

  /*! set location of 1d solution points in standard interval (required for tensor product elements)*/
  void set_loc_1d_upts(void);

  /*! set location of 1d shape points in standard interval (required for tensor product elements)*/
  void set_loc_1d_spts(array<double> &loc_1d_spts, int in_n_1d_spts);

  /*! set location of solution points */
  void set_loc_upts(void);

  /*! set location of flux points */
  void set_tloc_fpts(void);

  /*! set location and weight of interface cubature points */
  void set_inters_cubpts(void);

  /*! set location and weight of volume cubature points */
  void set_volume_cubpts(void);

  /*! set location of plot points */
  void set_loc_ppts(void);

  /*! set transformed normals at flux points */
  void set_tnorm_fpts(void);

  //#### helper methods ####

  void setup_ele_type_specific(void);

  /*! read restart info */
  int read_restart_info(ifstream& restart_file);

  /*! write restart info */
  void write_restart_info(ofstream& restart_file);

  /*! Compute interface jacobian determinant on face */
  double compute_inter_detjac_inters_cubpts(int in_inter, array<double> d_pos);

  /*! evaluate nodal basis */
  double eval_nodal_basis(int in_index, array<double> in_loc);

  /*! evaluate nodal basis */
  double eval_nodal_basis_restart(int in_index, array<double> in_loc);

  /*! evaluate derivative of nodal basis */
  double eval_d_nodal_basis(int in_index, int in_cpnt, array<double> in_loc);

  /*! evaluate divergence of vcjh basis */
  double eval_div_vcjh_basis(int in_index, array<double>& loc);

  void fill_opp_3(array<double>& opp_3);

  /*! evaluate nodal shape basis */
  double eval_nodal_s_basis(int in_index, array<double> in_loc, int in_n_spts);

  /*! evaluate derivative of nodal shape basis */
  void eval_d_nodal_s_basis(array<double> &d_nodal_s_basis, array<double> in_loc, int in_n_spts);

  /*! Compute the filter matrix for subgrid-scale models */
  void compute_filter_upts(void);

  /*! Matrix of filter weights at solution points in 1D */
  array<double> filter_upts_1D;

  /*! Calculate element volume */
  double calc_ele_vol(double& detjac);

  /*! Element reference length calculation */
  double calc_h_ref_specific(int in_ele);

protected:

  /*! evaluate Vandermonde matrix */
  void set_vandermonde(void);

  // members
  array<double> vandermonde;
  array<double> inv_vandermonde;

  /*! return position of 1d solution point */
  double get_loc_1d_upt(int in_index);

  /*! location of solution points in standard interval (tensor product elements only)*/
  array<double> loc_1d_upts;

  /*! location of solution points in standard interval (tensor product elements only)*/
  array<double> loc_1d_upts_rest;
};
