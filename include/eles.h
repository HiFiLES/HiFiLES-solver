/*!
 * \file eles.h
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
#include "input.h"

#if defined _GPU
#include "cuda_runtime_api.h"
#include "cusparse_v2.h"
#endif

class eles
{	
public:

  // #### constructors ####

  // default constructor

  eles();

  // default destructor

  ~eles();

  // #### methods ####

  /*! setup */
  void setup(int in_n_eles, int in_max_s_spts_per_ele);

  /*! setup initial conditions */
  void set_ics(double& time);

  /*! read data from restart file */
  void read_restart_data(ifstream& restart_file);

  /*! write data to restart file */
  void write_restart_data(ofstream& restart_file);

  /*! write extra restart file containing x,y,z of solution points instead of solution data */
  void write_restart_mesh(ofstream& restart_file);

	/*! move all to from cpu to gpu */
	void mv_all_cpu_gpu(void);

	/*! move wall distance array to from cpu to gpu */
	void mv_wall_distance_cpu_gpu(void);

  /*! move wall distance magnitude array to from cpu to gpu */
  void mv_wall_distance_mag_cpu_gpu(void);

	/*! copy transformed discontinuous solution at solution points to cpu */
	void cp_disu_upts_gpu_cpu(void);

	/*! copy transformed discontinuous solution at solution points to gpu */
  void cp_disu_upts_cpu_gpu(void);

  void cp_grad_disu_upts_gpu_cpu(void);

  /*! copy determinant of jacobian at solution points to cpu */
  void cp_detjac_upts_gpu_cpu(void);

  /*! copy divergence at solution points to cpu */
  void cp_div_tconf_upts_gpu_cpu(void);

  /*! copy local time stepping reference length at solution points to cpu */
  void cp_h_ref_gpu_cpu(void);

  /*! copy source term at solution points to cpu */
  void cp_src_upts_gpu_cpu(void);

  /*! copy elemental sensor values to cpu */
  void cp_sensor_gpu_cpu(void);

  /*! copy AV co-eff values at solution points to cpu */
  void cp_epsilon_upts_gpu_cpu(void);

  /*! remove transformed discontinuous solution at solution points from cpu */
  void rm_disu_upts_cpu(void);

  /*! remove determinant of jacobian at solution points from cpu */
  void rm_detjac_upts_cpu(void);

  /*! calculate the discontinuous solution at the flux points */
  void extrapolate_solution(int in_disu_upts_from);

  /*! Calculate terms for some LES models */
  void calc_sgs_terms(int in_disu_upts_from);

  /*! calculate transformed discontinuous inviscid flux at solution points */
  void evaluate_invFlux(int in_disu_upts_from);
  
  /*! calculate divergence of transformed discontinuous flux at solution points */
  void calculate_divergence(int in_div_tconf_upts_to);
  
  /*! calculate normal transformed discontinuous flux at flux points */
  void extrapolate_totalFlux(void);
  
  /*! calculate subgrid-scale flux at flux points */
  void evaluate_sgsFlux(void);

  /*! calculate divergence of transformed continuous flux at solution points */
  void calculate_corrected_divergence(int in_div_tconf_upts_to);
  
  /*! calculate uncorrected transformed gradient of the discontinuous solution at the solution points */
  void calculate_gradient(int in_disu_upts_from);

  /*! calculate corrected gradient of the discontinuous solution at solution points */
  void correct_gradient(void);

  /*! calculate corrected gradient of the discontinuous solution at flux points */
  void extrapolate_corrected_gradient(void);

  /*! calculate corrected gradient of solution at flux points */
  //void extrapolate_corrected_gradient(void);

  /*! calculate transformed discontinuous viscous flux at solution points */
  void evaluate_viscFlux(int in_disu_upts_from);

  /*! calculate divergence of transformed discontinuous viscous flux at solution points */
  //void calc_div_tdisvisf_upts(int in_div_tconinvf_upts_to);

  /*! calculate normal transformed discontinuous viscous flux at flux points */
  //void calc_norm_tdisvisf_fpts(void);

  /*! calculate divergence of transformed continuous viscous flux at solution points */
  //void calc_div_tconvisf_upts(int in_div_tconinvf_upts_to);

  /*! calculate source term for SA turbulence model at solution points */
  void calc_src_upts_SA(int in_disu_upts_from);
  
  /*! advance solution using a runge-kutta scheme */
  void AdvanceSolution(int in_step, int adv_type);

  /*! Calculate element local timestep */
  double calc_dt_local(int in_ele);

  /*! get number of elements */
  int get_n_eles(void);

  // get number of ppts_per_ele
  int get_n_ppts_per_ele(void);

  // get number of peles_per_ele
  int get_n_peles_per_ele(void);

  // get number of verts_per_ele
  int get_n_verts_per_ele(void);

  /*! get number of solution points per element */
  int get_n_upts_per_ele(void);

  /*! get element type */
  int get_ele_type(void);

  /*! get number of dimensions */
  int get_n_dims(void);

  /*! get number of fields */
  int get_n_fields(void);
  
  /*! set shape */
  void set_shape(int in_max_n_spts_per_ele);

  /*! set shape node */
  void set_shape_node(int in_spt, int in_ele, array<double>& in_pos);

  /*! Set new position of shape node in dynamic domain */
  void set_dynamic_shape_node(int in_spt, int in_ele, array<double> &in_pos);

  /*! set bc type */
  void set_bctype(int in_ele, int in_inter, int in_bctype);

  /*! set bc type */
  void set_bdy_ele2ele(void);

  /*! set number of shape points */
  void set_n_spts(int in_ele, int in_n_spts);

  /*!  set global element number */
  void set_ele2global_ele(int in_ele, int in_global_ele);

  /*! get a pointer to the transformed discontinuous solution at a flux point */
  double* get_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);
  
  /*! get a pointer to the normal transformed continuous flux at a flux point */
  double* get_norm_tconf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);

  /*! get a pointer to the determinant of the jacobian at a flux point (static->computational) */
  double* get_detjac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get a pointer to the determinant of the jacobian at a flux point (dynamic->static) */
  double* get_detjac_dyn_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get pointer to the equivalent of 'dA' (face area) at a flux point in static physical space */
  double* get_tdA_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get pointer to the equivalent of 'dA' (face area) at a flux point in dynamic physical space */
  double* get_ndA_dyn_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get a pointer to the normal at a flux point */
  double* get_norm_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

  /*! get a pointer to the normal at a flux point in dynamic space */
  double* get_norm_dyn_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

  /*! get a CPU pointer to the coordinates at a flux point */
  double* get_loc_fpts_ptr_cpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

  /*! get a GPU pointer to the coordinates at a flux point */
  double* get_loc_fpts_ptr_gpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

  /*! get a CPU pointer to the dynamic physical coordinates at a flux point */
  double* get_pos_dyn_fpts_ptr_cpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

  /*! get a pointer to delta of the transformed discontinuous solution at a flux point */
  double* get_delta_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);

  /*! get a pointer to gradient of discontinuous solution at a flux point */
  double* get_grad_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_field, int in_ele);

  /*! get a pointer to gradient of discontinuous solution at a flux point */
  double* get_normal_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele, array<double> temp_loc, double temp_pos[3]);
  
  /*! get a pointer to the normal transformed continuous viscous flux at a flux point */
  //double* get_norm_tconvisf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);
  
  /*! get a pointer to the subgrid-scale flux at a flux point */
  double* get_sgsf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_dim, int in_ele);

  /*! set opp_0 */
  void set_opp_0(int in_sparse);
  
  /*! set opp_1 */
  void set_opp_1(int in_sparse);

  /*! set opp_2 */
  void set_opp_2(int in_sparse);

  /*! set opp_3 */
  void set_opp_3(int in_sparse);

  /*! set opp_4 */
  void set_opp_4(int in_sparse);

  /*! set opp_5 */
  void set_opp_5(int in_sparse);

  /*! set opp_6 */
  void set_opp_6(int in_sparse);

  /*! set opp_p */
  void set_opp_p(void);

  /*! set opp_p */
  void set_opp_inters_cubpts(void);

  /*! set opp_p */
  void set_opp_volume_cubpts(void);

  /*! set opp_r */
  void set_opp_r(void);

  /*! calculate position of the plot points */
  void calc_pos_ppts(int in_ele, array<double>& out_pos_ppts);

  void set_rank(int in_rank);

  virtual void set_connectivity_plot()=0;

  void set_disu_upts_to_zero_other_levels(void);

  array<int> get_connectivity_plot();

  /*! calculate solution at the plot points */
  void calc_disu_ppts(int in_ele, array<double>& out_disu_ppts);

  /*! calculate gradient of solution at the plot points */
  void calc_grad_disu_ppts(int in_ele, array<double>& out_grad_disu_ppts);

  /*! calculate sensor at the plot points */
  void calc_sensor_ppts(int in_ele, array<double>& out_sensor_ppts);

  /*! calculate AV-co-efficients at the plot points */
  void calc_epsilon_ppts(int in_ele, array<double>& out_epsilon_ppts);

  /*! calculate time-averaged diagnostic fields at the plot points */
  void calc_time_average_ppts(int in_ele, array<double>& out_disu_average_ppts);

  /*! calculate diagnostic fields at the plot points */
  void calc_diagnostic_fields_ppts(int in_ele, array<double>& in_disu_ppts, array<double>& in_grad_disu_ppts, array<double>& in_sensor_ppts, array<double> &in_epsilon_ppts, array<double>& out_diag_field_ppts, double& time);

  /*! calculate position of a solution point */
  void calc_pos_upt(int in_upt, int in_ele, array<double>& out_pos);

  /*! get physical position of a flux point */
  void calc_pos_fpt(int in_fpt, int in_ele, array<double>& out_pos);

  /*! returns position of a solution point */
  double get_loc_upt(int in_upt, int in_dim);

  /*! set transforms */
  void set_transforms(void);
       
  /*! set transforms at the interface cubature points */
  void set_transforms_inters_cubpts(void);

  /*! set transforms at the volume cubature points */
  void set_transforms_vol_cubpts(void);

	/*! Calculate distance of solution points to no-slip wall */
	void calc_wall_distance(int n_seg_noslip_inters, int n_tri_noslip_inters, int n_quad_noslip_inters, array< array<double> > loc_noslip_bdy);

	/*! Calculate distance of solution points to no-slip wall in parallel */
	void calc_wall_distance_parallel(array<int> n_seg_noslip_inters, array<int> n_tri_noslip_inters, array<int> n_quad_noslip_inters, array< array<double> > loc_noslip_bdy_global, int nproc);

  /*! calculate position */
  void calc_pos(array<double> in_loc, int in_ele, array<double>& out_pos);

  /*! calculate derivative of position */
  void calc_d_pos(array<double> in_loc, int in_ele, array<double>& out_d_pos);

  /*! calculate derivative of position at a solution point (using pre-computed gradients) */
  void calc_d_pos_upt(int in_upt, int in_ele, array<double>& out_d_pos);

  /*! calculate derivative of position at a flux point (using pre-computed gradients) */
  void calc_d_pos_fpt(int in_fpt, int in_ele, array<double>& out_d_pos);
  
  // #### virtual methods ####

  virtual void setup_ele_type_specific()=0;

  /*! prototype for element reference length calculation */
  virtual double calc_h_ref_specific(int in_eles) = 0;

  virtual int read_restart_info(ifstream& restart_file)=0;

  virtual void write_restart_info(ofstream& restart_file)=0;

  /*! Compute interface jacobian determinant on face */
  virtual double compute_inter_detjac_inters_cubpts(int in_inter, array<double> d_pos)=0;

  /*! evaluate nodal basis */
  virtual double eval_nodal_basis(int in_index, array<double> in_loc)=0;

  /*! evaluate nodal basis for restart file*/
  virtual double eval_nodal_basis_restart(int in_index, array<double> in_loc)=0;

  /*! evaluate derivative of nodal basis */
  virtual double eval_d_nodal_basis(int in_index, int in_cpnt, array<double> in_loc)=0;

  virtual void fill_opp_3(array<double>& opp_3)=0;

  /*! evaluate divergence of vcjh basis */
  //virtual double eval_div_vcjh_basis(int in_index, array<double>& loc)=0;

  /*! evaluate nodal shape basis */
  virtual double eval_nodal_s_basis(int in_index, array<double> in_loc, int in_n_spts)=0;

  /*! evaluate derivative of nodal shape basis */
  virtual void eval_d_nodal_s_basis(array<double> &d_nodal_s_basis, array<double> in_loc, int in_n_spts)=0;

  /*! Calculate SGS flux */
  void calc_sgsf_upts(array<double>& temp_u, array<double>& temp_grad_u, double& detjac, int ele, int upt, array<double>& temp_sgsf);

  /*! rotate velocity components to surface*/
  array<double> calc_rotation_matrix(array<double>& norm);

  /*! calculate wall shear stress using LES wall model*/
  void calc_wall_stress(double rho, array<double>& urot, double ene, double mu, double Pr, double gamma, double y, array<double>& tau_wall, double q_wall);

  /*! Wall function calculator for Breuer-Rodi wall model */
  double wallfn_br(double yplus, double A, double B, double E, double kappa);

  /*! Calculate element volume */
  virtual double calc_ele_vol(double& detjac)=0;

  double compute_res_upts(int in_norm_type, int in_field);

  /*! calculate body forcing at solution points */
  void evaluate_body_force(int in_file_num);

  /*! Compute volume integral of diagnostic quantities */
  void CalcIntegralQuantities(int n_integral_quantities, array <double>& integral_quantities);

  /*! Compute time-average diagnostic quantities */
  void CalcTimeAverageQuantities(double& time);

  void compute_wall_forces(array<double>& inv_force, array<double>& vis_force, double& temp_cl, double& temp_cd, ofstream& coeff_file, bool write_forces);

  array<double> compute_error(int in_norm_type, double& time);
  
  array<double> get_pointwise_error(array<double>& sol, array<double>& grad_sol, array<double>& loc, double& time, int in_norm_type);

  /*! calculate position of a point in physical (dynamic) space from (r,s,t) coordinates*/
  void calc_pos_dyn(array<double> in_loc, int in_ele, array<double> &out_pos);

  /*!
   * Calculate dynamic position of solution point
   * \param[in] in_upt - ID of solution point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_pos_dyn_fpt(int in_fpt, int in_ele, array<double> &out_pos);

  /*!
   * Calculate dynamic position of flux point
   * \param[in] in_upt - ID of solution point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_pos_dyn_upt(int in_upt, int in_ele, array<double> &out_pos);

  /*!
   * Calculate dynamic position of plot point
   * \param[in] in_ppt - ID of plot point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_pos_dyn_ppt(int in_ppt, int in_ele, array<double> &out_pos);

  /*!
   * Calculate dynamic position of volume cubature point
   * \param[in] in_cubpt - ID of cubature point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_pos_dyn_vol_cubpt(int in_cubpt, int in_ele, array<double> &out_pos);

  /*!
   * Calculate dynamic position of interface cubature point
   * \param[in] in_cubpt - ID of cubature point on element face to evaluate at
   * \param[in] in_face - local face ID within element
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_pos_dyn_inters_cubpt(int in_cubpt, int in_face, int in_ele, array<double> &out_pos);

  /*!
   * Calculate derivative of dynamic position wrt reference (initial,static) position
   * \param[in] in_loc - position of point in computational space
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_d_pos_dyn(array<double> in_loc, int in_ele, array<double> &out_d_pos);

  /*!
   * Calculate derivative of dynamic position wrt reference (initial,static) position at fpt
   * \param[in] in_fpt - ID of flux point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_d_pos_dyn_fpt(int in_fpt, int in_ele, array<double> &out_d_pos);

  /*!
   * Calculate derivative of dynamic position wrt reference (initial,static) position at upt
   * \param[in] in_upt - ID of solution point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_d_pos_dyn_upt(int in_upt, int in_ele, array<double> &out_d_pos);

  /*!
   * Calculate derivative of dynamic position wrt reference (initial,static) position at
     volume cubature point
   * \param[in] in_cubpt - ID of cubature point within element to evaluate at
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_d_pos_dyn_vol_cubpt(int in_cubpt, int in_ele, array<double> &out_d_pos);

  /*!
   * Calculate derivative of dynamic position wrt reference (initial,static) position at
     volume cubature point
   * \param[in] in_cubpt - ID of cubature point on face to evaluate at
   * \param[in] in_face - local face ID within element
   * \param[in] in_ele - local element ID
   * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
   */
  void calc_d_pos_dyn_inters_cubpt(int in_cubpt, int in_face, int in_ele, array<double> &out_d_pos);

  /*! pre-computing shape basis contributions at flux points for more efficient access */
  void store_nodal_s_basis_fpts(void);

  /*! pre-computing shape basis contributions at solution points for more efficient access */
  void store_nodal_s_basis_upts(void);

  /*! pre-computing shape basis contributions at plot points for more efficient access */
  void store_nodal_s_basis_ppts(void);

  /*! pre-computing shape basis contributions at plot points for more efficient access */
  void store_nodal_s_basis_vol_cubpts(void);

  /*! pre-computing shape basis contributions at plot points for more efficient access */
  void store_nodal_s_basis_inters_cubpts(void);

  /*! pre-computing shape basis deriavative contributions at flux points for more efficient access */
  void store_d_nodal_s_basis_fpts(void);

  /*! pre-computing shape basis derivative contributions at solution points for more efficient access */
  void store_d_nodal_s_basis_upts(void);

  /*! pre-computing shape basis derivative contributions at solution points for more efficient access */
  void store_d_nodal_s_basis_vol_cubpts(void);

  /*! pre-computing shape basis derivative contributions at solution points for more efficient access */
  void store_d_nodal_s_basis_inters_cubpts(void);

  /*! initialize arrays for storing grid velocities */
  void initialize_grid_vel(int in_max_n_spts_per_ele);

  /*! set grid velocity on element shape points */
  void set_grid_vel_spt(int in_ele, int in_spt, array<double> in_vel);

  /*! interpolate grid velocity from shape points to flux points */
  void set_grid_vel_fpts(int in_rk_step);

  /*! interpolate grid velocity from shape points to solution points */
  void set_grid_vel_upts(int in_rk_step);

  /*! interpolate grid velocity from shape points to plot points */
  void set_grid_vel_ppts(void);

  /*! Get array of grid velocity at all plot points */
  array<double> get_grid_vel_ppts(void);

  /*! Get pointer to grid velocity at a flux point */
  double *get_grid_vel_fpts_ptr(int in_ele, int in_ele_local_inter, int in_inter_local_fpt, int in_dim);

  /*! Set the transformation variables for dynamic-physical -> static-physical frames */
  void set_transforms_dynamic(void);

  /* --- Geometric Conservation Law (GCL) Funcitons --- */
  /*! Update the dynamic transformation variables with the GCL-corrected Jacobian determinant */
  void correct_dynamic_transforms(void);

  /*! GCL Residual-Calculation Steps */
  void evaluate_GCL_flux(int in_disu_upts_from);
  void extrapolate_GCL_solution(int in_disu_upts_from);
  void extrapolate_GCL_flux(void);
  void calculate_divergence_GCL(int in_div_tconf_upts_to);
  void calculate_corrected_divergence_GCL(int in_div_tconf_upts_to);

  double *get_disu_GCL_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);
  /* --------------------------------------------------- */

  /*! Set the time step for the current iteration */
  void set_dt(int in_step, int adv_type);

#ifdef _GPU
  void cp_transforms_gpu_cpu(void);
  void cp_transforms_cpu_gpu(void);

  void perturb_shape(double rk_time);
  void rigid_move(double rk_time);

  void calc_grid_velocity(void);
  void rigid_grid_velocity(double rk_time);
  void perturb_grid_velocity(double rk_time);
#endif

  /* --- Shock capturing functions --- */

  void shock_capture_concentration(int in_disu_upts_from);
  void shock_capture_concentration_cpu(int in_n_eles, int in_n_upts_per_ele, int in_n_fields, int in_order, int in_ele_type, int in_artif_type, double s0, double kappa, double* in_disu_upts_ptr, double* in_inv_vandermonde_ptr, double* in_inv_vandermonde2D_ptr, double* in_vandermonde2D_ptr, double* concentration_array_ptr, double* out_sensor, double* sigma);

protected:

  // #### members ####

  /// flag to avoid re-setting-up transform arrays
  bool first_time;

  /*! mesh motion flag */
  int motion;

  /*! viscous flag */
  int viscous;

  /*! LES flag */
  int LES;

  /*! SGS model */
  int sgs_model;

  /*! LES filter flag */
  int filter;

  /*! near-wall model */
  int wall_model;

  /*! number of elements */
  int n_eles;

  /*! number of elements that have a boundary face*/
  int n_bdy_eles;

  /*!  number of dimensions */
  int n_dims;

  /*!  number of prognostic fields */
  int n_fields;

  /*!  number of diagnostic fields */
  int n_diagnostic_fields;

  /*!  number of time averaged diagnostic fields */
  int n_average_fields;

  /*! order of solution polynomials */
  int order;

  /*! order of interface cubature rule */
  int inters_cub_order;

  /*! order of interface cubature rule */
  int volume_cub_order;

  /*! order of solution polynomials in restart file*/
  int order_rest;

  /*! number of solution points per element */
  int n_upts_per_ele;

  /*! number of solution points per element */
  int n_upts_per_ele_rest;

  /*! number of flux points per element */
  int n_fpts_per_ele;

  /*! number of vertices per element */
  int n_verts_per_ele;

  array<int> connectivity_plot;

  /*! plotting resolution */
  int p_res;

  /*! solution point type */
  int upts_type;

  /*! flux point type */
  int fpts_type;

  /*! number of plot points per element */
  int n_ppts_per_ele;

  /*! number of plot elements per element */
  int n_peles_per_ele;

  /*! Global cell number of element */
  array<int> ele2global_ele;

  /*! Global cell number of element */
  array<int> bdy_ele2ele;

  /*! Boundary condition type of faces */
  array<int> bctype;

  /*! number of shape points per element */
  array<int> n_spts_per_ele;

  /*! transformed normal at flux points */
  array<double> tnorm_fpts;

  /*! transformed normal at flux points */
  array< array<double> > tnorm_inters_cubpts;

  /*! location of solution points in standard element */
  array<double> loc_upts;

  /*! location of solution points in standard element */
  array<double> loc_upts_rest;

  /*! location of flux points in standard element */
  array<double> tloc_fpts;

  /*! location of interface cubature points in standard element */
  array< array<double> > loc_inters_cubpts;

  /*! weight of interface cubature points in standard element */
  array< array<double> > weight_inters_cubpts;

  /*! location of volume cubature points in standard element */
  array<double> loc_volume_cubpts;

  /*! weight of cubature points in standard element */
  array<double> weight_volume_cubpts;

  /*! transformed normal at cubature points */
	array< array<double> > tnorm_cubpts;

	/*! location of plot points in standard element */
	array<double> loc_ppts;
	
	/*! location of shape points in standard element (simplex elements only)*/
	array<double> loc_spts;
	
	/*! number of interfaces per element */
	int n_inters_per_ele;
	
	/*! number of flux points per interface */
	array<int> n_fpts_per_inter; 

	/*! number of cubature points per interface */
	array<int> n_cubpts_per_inter; 

	/*! number of cubature points per interface */
	int n_cubpts_per_ele; 

	/*! element type (0=>quad,1=>tri,2=>tet,3=>pri,4=>hex) */
	int ele_type; 
	
	/*! order of polynomials defining shapes */
	int s_order;
	
  /*! maximum number of shape points used by any element */
  int max_n_spts_per_ele;

  /*! position of shape points (mesh vertices) in static-physical domain */
	array<double> shape;
	
  /*! position of shape points (mesh vertices) in dynamic-physical domain */
  array<double> shape_dyn;

  /*!
  Description: Mesh velocity at shape points \n
  indexing: (in_ele)(in_spt, in_dim) \n
  */
  array<double> vel_spts;

  /*!
  Description: Mesh velocity at flux points (interpolated using shape basis funcs) \n
  indexing: (in_dim, in_fpt, in_ele) \n
  */
  array<double> grid_vel_upts, grid_vel_fpts, vel_ppts;

  /*! nodal shape basis contributions at flux points */
  array<double> nodal_s_basis_fpts;

  /*! nodal shape basis contributions at solution points */
  array<double> nodal_s_basis_upts;

  /*! nodal shape basis contributions at output plot points */
  array<double> nodal_s_basis_ppts;

  /*! nodal shape basis contributions at output plot points */
  array<double> nodal_s_basis_vol_cubpts;

  /*! nodal shape basis contributions at output plot points */
  array<array<double> > nodal_s_basis_inters_cubpts;

  /*! nodal shape basis derivative contributions at flux points */
  array<double> d_nodal_s_basis_fpts;

  /*! nodal shape basis derivative contributions at solution points */
  array<double> d_nodal_s_basis_upts;

  /*! nodal shape basis contributions at output plot points */
  array<double> d_nodal_s_basis_vol_cubpts;

  /*! nodal shape basis contributions at output plot points */
  array<array<double> > d_nodal_s_basis_inters_cubpts;

	/*! temporary solution storage at a single solution point */
	array<double> temp_u;

  /*! temporary grid velocity storage at a single solution point */
  array<double> temp_v;

  /*! temporary grid velocity storage at a single solution point (transformed to static frame) */
  array<double> temp_v_ref;

  /*! temporary flux storage for GCL at a single solution point (transformed to static frame) */
  array<double> temp_f_GCL;

  /*! temporary flux storage for GCL at a single solution point (transformed to static frame) */
  array<double> temp_f_ref_GCL;

  /*! constansts for RK time-stepping */
  array<double> RK_a, RK_b, RK_c;

	/*! temporary solution gradient storage */
	array<double> temp_grad_u;

	/*! Matrix of filter weights at solution points */
	array<double> filter_upts;

	/*! extra arrays for similarity model: Leonard tensors, velocity/energy products */
	array<double> Lu, Le, uu, ue;

	/*! temporary flux storage */
	array<double> temp_f;

  /*! temporary flux storage */
  array<double> temp_f_ref;

	/*! temporary subgrid-scale flux storage */
	array<double> temp_sgsf;

  /*! temporary subgrid-scale flux storage for dynamic->static transformation */
  array<double> temp_sgsf_ref;
	
	/*! storage for distance of solution points to nearest no-slip boundary */
	array<double> wall_distance;
  array<double> wall_distance_mag;

	array<double> twall;

	/*! number of storage levels for time-integration scheme */
	int n_adv_levels;
	
  /*! determinant of Jacobian (transformation matrix) at solution points
   *  (J = |G|) */
	array<double> detjac_upts;
	
  /*! determinant of Jacobian (transformation matrix) at flux points
   *  (J = |G|) */
	array<double> detjac_fpts;

  /*! determinant of jacobian at volume cubature points. TODO: what is this really? */
	array< array<double> > vol_detjac_inters_cubpts;

	/*! determinant of volume jacobian at cubature points. TODO: what is this really? */
	array< array<double> > vol_detjac_vol_cubpts;

  /*! Full vector-transform matrix from static physical->computational frame, at solution points
   *  [Determinant of Jacobian times inverse of Jacobian] [J*G^-1] */
  array<double> JGinv_upts;
	
  /*! Full vector-transform matrix from static physical->computational frame, at flux points
   *  [Determinant of Jacobian times inverse of Jacobian] [J*G^-1] */
  array<double> JGinv_fpts;
	
  /*! Magnitude of transformed face-area normal vector from computational -> static-physical frame
   *  [magntiude of (normal dot inverse static transformation matrix)] [ |J*(G^-1)*(n*dA)| ] */
  array<double> tdA_fpts;

	/*! determinant of interface jacobian at flux points */
	array< array<double> > inter_detjac_inters_cubpts;

	/*! normal at flux points*/
	array<double> norm_fpts;
	
  /*! static-physical coordinates at flux points*/
  array<double> pos_fpts;

  /*! static-physical coordinates at solution points*/
  array<double> pos_upts;

  /*! normal at interface cubature points*/
  array< array<double> > norm_inters_cubpts;

  /*! determinant of dynamic jacobian at solution points ( |G| ) */
  array<double> J_dyn_upts;

  /*! determinant of dynamic jacobian at flux points ( |G| ) */
  array<double> J_dyn_fpts;

  /*! Dynamic transformation matrix at solution points ( |G|*G^-1 ) */
  array<double>  JGinv_dyn_upts;

  /*! Dynamic->Static transformation matrix at flux points ( |G|*G^-1 ) */
  array<double>  JGinv_dyn_fpts;

  /*! Static->Dynamic transformation matrix at flux points ( G/|G| ) */
  array<double>  JinvG_dyn_fpts;

  /*! transformed gradient of determinant of dynamic jacobian at solution points */
  array<double> tgrad_J_dyn_upts;

  /*! transformed gradient of determinant of dynamic jacobian at flux points */
  array<double> tgrad_J_dyn_fpts;

  /*! normal at flux points in dynamic mesh */
  array<double> norm_dyn_fpts;

  /*! physical coordinates at flux points in dynamic mesh */
  array<double> dyn_pos_fpts, dyn_pos_upts;

  /*! magnitude of transformed face-area normal vector from static-physical -> dynamic-physical frame
   *  [magntiude of (normal dot inverse dynamic transformation matrix)] [ |J*(G^-1)*(n*dA)| ] */
  array<double> ndA_dyn_fpts;

  /*!
        description: transformed discontinuous solution at the solution points
        indexing: \n
        matrix mapping:
        */
  array< array<double> > disu_upts;

	/*!
	running time-averaged diagnostic fields at solution points
	*/
	array<double> disu_average_upts;

	/*!
	time (in secs) until start of time average period for above diagnostic fields
	*/
  double spinup_time;

	/*!
	filtered solution at solution points for similarity and SVV LES models
	*/
	array<double> disuf_upts;

  /*! position at the plot points */
  array< array<double> > pos_ppts;

	/*!
	description: transformed discontinuous solution at the flux points \n
	indexing: (in_fpt, in_field, in_ele) \n
	matrix mapping: (in_fpt || in_field, in_ele)
	*/
	array<double> disu_fpts;

	/*!
	description: transformed discontinuous flux at the solution points \n
	indexing: (in_upt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_upt, in_dim || in_field, in_ele)
	*/
	array<double> tdisf_upts;
	
	/*!
	description: subgrid-scale flux at the solution points \n
	indexing: (in_upt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_upt, in_dim || in_field, in_ele)
	*/
	array<double> sgsf_upts;

	/*!
	description: subgrid-scale flux at the flux points \n
	indexing: (in_fpt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_fpt, in_dim || in_field, in_ele)
	*/
	array<double> sgsf_fpts;

	/*!
	normal transformed discontinuous flux at the flux points
	indexing: \n
	matrix mapping:
	*/
	array<double> norm_tdisf_fpts;
	
	/*!
	normal transformed continuous flux at the flux points
	indexing: \n
	matrix mapping:
	*/
	array<double> norm_tconf_fpts;
	
	/*!
	divergence of transformed continuous flux at the solution points
	indexing: \n
	matrix mapping:
	*/
	array< array<double> > div_tconf_upts;
	
	/*! delta of the transformed discontinuous solution at the flux points   */
	array<double> delta_disu_fpts;

	/*! gradient of discontinuous solution at solution points */
	array<double> grad_disu_upts;
	
	/*! gradient of discontinuous solution at flux points */
	array<double> grad_disu_fpts;

	/*! transformed discontinuous viscous flux at the solution points */
	//array<double> tdisvisf_upts;
	
	/*! normal transformed discontinuous viscous flux at the flux points */
	//array<double> norm_tdisvisf_fpts;
	
	/*! normal transformed continuous viscous flux at the flux points */
	//array<double> norm_tconvisf_fpts;
		
	/*! transformed gradient of determinant of jacobian at solution points */
	array<double> tgrad_detjac_upts;

  /*! source term for SA turbulence model at solution points */
  array<double> src_upts;

  array<double> d_nodal_s_basis;

  // TODO: change naming (comments) to reflect reuse

  /*!
   * description: discontinuous solution for GCL at the solution points \n
   * indexing: (in_upt, in_ele) \n
   */
  array< array<double> > Jbar_upts;

  /*!
   * description: discontinuous solution for GCL at the flux points \n
   * indexing: (in_fpt, in_ele) \n
   */
  array< array<double> > Jbar_fpts;

  /*!
   * description: transformed discontinuous flux for GCL at the solution points \n
   * indexing: (in_upt, in_dim, in_ele) \n
   */
  array<double> tdisf_GCL_upts;

  /*!
   * description: transformed discontinuous flux for GCL at the flux points \n
   * indexing: (in_fpt, in_dim, in_ele) \n
   */
  array<double> tdisf_GCL_fpts;

  /*!
   * normal transformed discontinuous GCL flux at the flux points \n
   * indexing: (in_fpt, in_ele) \n
   */
  array<double> norm_tdisf_GCL_fpts;

  /*!
   * normal transformed continuous GCL flux at the flux points \n
   * indexing: (in_fpt, in_ele) \n
   */
  array<double> norm_tconf_GCL_fpts;

  /*!
   * divergence of transformed continuous GCL flux at the solution points
   * indexing: (in_upt, in_ele) \n
   */
  array< array<double> > div_tconf_GCL_upts;

#ifdef _GPU
  cusparseHandle_t handle;
#endif

  /*! operator to go from transformed discontinuous solution at the solution points to transformed discontinuous solution at the flux points */
  array<double> opp_0;
  array<double> opp_0_data;
  array<int> opp_0_cols;
  array<int> opp_0_b;
  array<int> opp_0_e;
  int opp_0_sparse;

#ifdef _GPU
  array<double> opp_0_ell_data;
  array<int> opp_0_ell_indices;
  int opp_0_nnz_per_row;
#endif

  /*! operator to go from transformed discontinuous inviscid flux at the solution points to divergence of transformed discontinuous inviscid flux at the solution points */
  array< array<double> > opp_1;
  array< array<double> > opp_1_data;
  array< array<int> > opp_1_cols;
  array< array<int> > opp_1_b;
  array< array<int> > opp_1_e;
  int opp_1_sparse;
#ifdef _GPU
  array< array<double> > opp_1_ell_data;
  array< array<int> > opp_1_ell_indices;
  array<int> opp_1_nnz_per_row;
#endif

  /*! operator to go from transformed discontinuous inviscid flux at the solution points to normal transformed discontinuous inviscid flux at the flux points */
  array< array<double> > opp_2;
  array< array<double> > opp_2_data;
  array< array<int> > opp_2_cols;
  array< array<int> > opp_2_b;
  array< array<int> > opp_2_e;
  int opp_2_sparse;
#ifdef _GPU
  array< array<double> > opp_2_ell_data;
  array< array<int> > opp_2_ell_indices;
  array<int> opp_2_nnz_per_row;
#endif

  /*! operator to go from normal correction inviscid flux at the flux points to divergence of correction inviscid flux at the solution points*/
  array<double> opp_3;
  array<double> opp_3_data;
  array<int> opp_3_cols;
  array<int> opp_3_b;
  array<int> opp_3_e;
  int opp_3_sparse;
#ifdef _GPU
  array<double> opp_3_ell_data;
  array<int> opp_3_ell_indices;
  int opp_3_nnz_per_row;
#endif

  /*! operator to go from transformed solution at solution points to transformed gradient of transformed solution at solution points */
  array< array<double> >  opp_4;
  array< array<double> >  opp_4_data;
  array< array<int> > opp_4_cols;
  array< array<int> > opp_4_b;
  array< array<int> > opp_4_e;
  int opp_4_sparse;
#ifdef _GPU
  array< array<double> > opp_4_ell_data;
  array< array<int> > opp_4_ell_indices;
  array< int > opp_4_nnz_per_row;
#endif

  /*! operator to go from transformed solution at flux points to transformed gradient of transformed solution at solution points */
  array< array<double> > opp_5;
  array< array<double> > opp_5_data;
  array< array<int> > opp_5_cols;
  array< array<int> > opp_5_b;
  array< array<int> > opp_5_e;
  int opp_5_sparse;
#ifdef _GPU
  array< array<double> > opp_5_ell_data;
  array< array<int> > opp_5_ell_indices;
  array<int> opp_5_nnz_per_row;
#endif

  /*! operator to go from transformed solution at solution points to transformed gradient of transformed solution at flux points */
  array<double> opp_6;
  array<double> opp_6_data;
  array<int> opp_6_cols;
  array<int> opp_6_b;
  array<int> opp_6_e;
  int opp_6_sparse;
#ifdef _GPU
  array<double> opp_6_ell_data;
  array<int> opp_6_ell_indices;
  int opp_6_nnz_per_row;
#endif

  /*! operator to go from discontinuous solution at the solution points to discontinuous solution at the plot points */
  array<double> opp_p;

  array< array<double> > opp_inters_cubpts;
  array<double> opp_volume_cubpts;

  /*! operator to go from discontinuous solution at the restart points to discontinuous solution at the solutoin points */
  array<double> opp_r;

  /*! dimensions for blas calls */
  int Arows, Acols;
  int Brows, Bcols;
  int Astride, Bstride, Cstride;

  /*! general settings for mkl sparse blas */
  char matdescra[6];

  /*! transpose setting for mkl sparse blas */
  char transa;

  /*! zero for mkl sparse blas */
  double zero;

  /*! one for mkl sparse blas */
  double one;

  /*! number of fields multiplied by number of elements */
  int n_fields_mul_n_eles;

  /*! number of dimensions multiplied by number of solution points per element */
  int n_dims_mul_n_upts_per_ele;

  int rank;
  int nproc;

  /*! mass flux through inlet */
  double mass_flux;

  /*! reference element length */
  array<double> h_ref;
  
  /*! element local timestep */
  array<double> dt_local;
  double dt_local_new;
  array<double> dt_local_mpi;

  /*! Artificial Viscosity variables */
  array<double> vandermonde;
  array<double> inv_vandermonde;
  array<double> vandermonde2D;
  array<double> inv_vandermonde2D;
  array<double> area_coord_upts;
  array<double> area_coord_fpts;
  array<double> epsilon;
  array<double> epsilon_upts;
  array<double> epsilon_fpts;
  array<double> concentration_array;
  array<double> sensor;
  array<double> sigma;

  array<double> min_dt_local;

  /*! Global cell number of element as in the code */
  array<int> ele2global_ele_code;

};
