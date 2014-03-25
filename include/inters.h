/*!
 * \file inters.h
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

#include "inters.h"
#include "array.h"

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
  void right_flux(array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Rusanov flux */
  void rusanov_flux(array<double> &q_l, array<double> &q_r, array<double> &f_l, array<double> &f_r, array<double> v_g, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Roe flux */
  void roe_flux(array<double> &q_l, array<double> &q_r, array<double> v_g, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Lax-Friedrich flux (works only for wave equation) */
  void lax_friedrich(array<double> &u_l, array<double> &u_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double lambda, array<double>& wave_speed);

  /*! Compute common viscous flux using LDG formulation */
  void ldg_flux(int flux_spec, array<double> &u_l, array<double> &u_r, array<double> &f_l, array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double tau, double pen_fact);

  /*! Compute common solution using LDG formulation */
  void ldg_solution(int flux_spec, array<double> &u_l, array<double> &u_r, array<double> &u_c, double pen_fact, array<double>& norm);

  /*! get look up table for flux point connectivity based on rotation tag */
  void get_lut(int in_rot_tag);

  /*! Compute common flux at boundaries using convective flux formulation */
  void convective_flux_boundary(array<double> &f_l, array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields);


protected:

  // #### members ####

  int inters_type; // segment, quad or tri

  int order;
  int viscous;
  int LES;
  int n_inters;
  int n_fpts_per_inter;
  int n_fields;
  int n_dims;
  int motion;

  array<double*> disu_fpts_l;
  array<double*> delta_disu_fpts_l;
  array<double*> norm_tconf_fpts_l;
  //array<double*> norm_tconvisf_fpts_l;
  array<double*> detjac_fpts_l;
  array<double*> mag_tnorm_dot_inv_detjac_mul_jac_fpts_l;
  array<double*> norm_fpts;
  array<double*> loc_fpts;

  array<double> pos_disu_fpts_l;

  array<double*> grad_disu_fpts_l;
  array<double*> normal_disu_fpts_l;

  array<double> temp_u_l;
  array<double> temp_u_r;

  // Note: grid velocity is continuous across interfaces
  array<double*> vel_fpts;
  array<double> temp_v;

  array<double> temp_grad_u_l;
  array<double> temp_grad_u_r;

  array<double> temp_normal_u_l;

  array<double> temp_pos_u_l;

  array<double> temp_f_l;
  array<double> temp_f_r;

  array<double> temp_fn_l;
  array<double> temp_fn_r;

  array<double> temp_f;

  array<double> temp_loc;

  // LES and wall model quantities
  array<double*> sgsf_fpts_l;
  array<double> temp_sgsf_l;

  array<int> lut;

  array<double> v_l, v_r, um, du;

};
