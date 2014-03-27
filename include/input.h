/*!
 * \file input.h
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

#include <iostream>
#include <fstream>
#include "array.h"

class input
{	
public:

  // #### constructors ####

  // default constructor

  input();

  ~input();

  // #### methods ####


  void set_vcjh_scheme_tri(int in_vcjh_scheme_tri);
  void set_vcjh_scheme_hexa(int in_vcjh_scheme_hexa);
  void set_vcjh_scheme_pri_1d(int in_vcjh_scheme_pri_1d);

  void set_order(int in_order);
  void set_c(double in_c_tri, double in_c_quad);
  void set_dt(double in_dt);

  void setup(ifstream& in_run_input_file, int rank);
  
  void reset(int c_ind, int p_ind, int grid_ind, int vis_ind, int tau_ind, int dev_ind, int dim_ind);

  // #### members ####

  double gamma;

  int viscous;
  int equation;

  int n_diagnostic_fields;
  array<string> diagnostic_fields;
  int n_integral_quantities;
  array<string> integral_quantities;

  double prandtl;

  double tau;
  double pen_fact;
  double fix_vis;
  double diff_coeff;
  double const_src_term;

  int order;
  int inters_cub_order;
  int volume_cub_order;

  int test_case;
  array<double> wave_speed;
  double lambda;

  double dt;
  int dt_type;
  double CFL;
  int n_steps;
  int plot_freq;
  int restart_dump_freq;
  int adv_type;

  int LES;
  int filter_type;
	double filter_ratio;
	int SGS_model;
	int wall_model;
	double wall_layer_t;

  int monitor_res_freq;
  int monitor_force_freq;
  int monitor_integrals_freq;
  int res_norm_type; // 0:infinity norm, 1:L1 norm, 2:L2 norm
  int error_norm_type; // 0:infinity norm, 1:L1 norm, 2:L2 norm
  int res_norm_field;

  int restart_flag;
  int restart_iter;
  int n_restart_files;

  int ic_form;


  // boundary_conditions
  double rho_bound;
  array<double> v_bound;
  double p_bound;
  double p_total_bound;
  double T_total_bound;

  int mesh_format;
  string mesh_file;

  double dx_cyclic;
  double dy_cyclic;
  double dz_cyclic;


  int p_res;
  int write_type;

  int upts_type_tri;
  int fpts_type_tri;
  int vcjh_scheme_tri;
  double c_tri;
  int sparse_tri;

  int upts_type_quad;
  int vcjh_scheme_quad;
  double eta_quad;
  double c_quad;
  int sparse_quad;

  int upts_type_hexa;
  int vcjh_scheme_hexa;
  double eta_hexa;
  int sparse_hexa;

  int upts_type_tet;
  int fpts_type_tet;
  int vcjh_scheme_tet;
  double c_tet;
  double eta_tet;
  int sparse_tet;

  int upts_type_pri_tri;
  int upts_type_pri_1d;
  int vcjh_scheme_pri_1d;
  double eta_pri;
  int sparse_pri;

  int riemann_solve_type;
  int vis_riemann_solve_type;

  //new
  double S_gas;
  double T_gas;
  double R_gas;
  double mu_gas;

  double c_sth;
  double mu_inf;
  double rt_inf;

  double Mach_free_stream;
  double nx_free_stream;
  double ny_free_stream;
  double nz_free_stream;
  double Re_free_stream;
  double L_free_stream;
  double rho_free_stream;
  double p_free_stream;
  double T_free_stream;
  double u_free_stream;
  double v_free_stream;
  double w_free_stream;
  double mu_free_stream;

  double T_ref;
  double L_ref;
  double R_ref;
  double uvw_ref;
  double rho_ref;
  double p_ref;
  double mu_ref;
  double time_ref;
  
  double Mach_wall;
  double nx_wall;
  double ny_wall;
  double nz_wall;

  array<double> v_wall;
  double uvw_wall;
  double T_wall;
  
  double Mach_c_ic;
  double nx_c_ic;
  double ny_c_ic;
  double nz_c_ic;
  double Re_c_ic;
  double rho_c_ic;
  double p_c_ic;
  double T_c_ic;
  double uvw_c_ic;
  double u_c_ic;
  double v_c_ic;
  double w_c_ic;
  double mu_c_ic;

  double a_init, b_init;
  int bis_ind, file_lines;
  int device_num;
  int forcing;
  array<double> x_coeffs;
  array<double> y_coeffs;
  array<double> z_coeffs;
  int perturb_ic;

};
