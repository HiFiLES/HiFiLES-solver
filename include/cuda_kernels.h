/*!
 * \file cuda_kernels.h
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

#define PI 3.141592653589793

void RK45_update_kernel_wrapper(int n_upts_per_ele,int n_dims,int n_fields,int n_eles,double* disu0_upts_ptr,double* disu1_upts_ptr,double* div_tconf_upts_ptr, double* detjac_upts_ptr, double* src_upts_ptr, double* h_ref, double rk4a, double rk4b, double dt, double const_src, double CFL, double gamma, double mu_inf, int order, int viscous, int dt_type, int step);

void RK11_update_kernel_wrapper(int n_upts_per_ele,int n_dims,int n_fields,int n_eles,double* disu0_upts_ptr,double* div_tconf_upts_ptr, double* detjac_upts_ptr, double* src_upts_ptr, double* h_ref, double dt, double const_src, double CFL, double gamma, double mu_inf, int order, int viscous, int dt_type);

/*! wrapper for gpu kernel to calculate transformed discontinuous inviscid flux at solution points */
void evaluate_invFlux_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* disu_upts_ptr, double* out_tdisinvf_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double* grid_vel_upts_ptr, double gamma, int motion, int equation, double wave_speed_x, double wave_speed_y, double wave_speed_z, int turb_model);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points */

void calculate_common_invFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconinvf_fpts_l_ptr, double** norm_tconinvf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** tdA_dyn_fpts_l_ptr, double** tdA_dyn_fpts_r_ptr, double** detjac_dyn_fpts_l_ptr, double** detjac_dyn_fpts_r_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double **grid_vel_fpts_ptr, int riemann_solve_type, double **delta_disu_fpts_l_ptr, double **delta_disu_fpts_r_ptr, double gamma, double pen_fact, int viscous, int motion, int vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int turb_model);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries*/
void evaluate_boundaryConditions_invFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** norm_tconinvf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_l_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** loc_fpts_ptr, double** loc_dyn_fpts_ptr, double** grid_vel_fpts_ptr, int* boundary_type, double* bdy_params, int riemann_solve_type, double** delta_disu_fpts_l_ptr, double gamma, double R_ref, int viscous, int motion, int vis_riemann_solve_type, double time_bound, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int equation, int turb_model);


void transform_grad_disu_upts_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, int equation, int motion);

/*! wrapper for gpu kernel to calculate transformed discontinuous viscous flux at solution points */
void evaluate_viscFlux_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, int ele_type, int order, double filter_ratio, int LES, int motion, int sgs_model, int wall_model, double wall_thickness, double* wall_dist_ptr, double* twall_ptr, double* Leonard_mom_ptr, double* Leonard_energy_ptr, double* disu_upts_ptr, double* out_tdisvisf_upts_ptr, double* out_sgsf_upts_ptr, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int equation, double diff_coeff, int turb_model, double c_v1, double omega, double prandtl_t);

/*! wrapper for gpu kernel to calculate corrected gradient of solution at flux points */
/*
void extrapolate_corrected_gradient_gpu_kernel_wrapper(int n_fpts_per_ele, int n_dims, int n_fields, int n_eles, double* disu_fpts_ptr, double* delta_disu_fpts_ptr, double* out_grad_disu_fpts_ptr, double* detjac_fpts_ptr, double* JGinv_fpts_ptr, double* tgrad_detjac_fpts_ptr);
*/

/*! wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points */
void calculate_common_viscFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconvisf_fpts_l_ptr, double** norm_tconvisf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** tdA_dyn_fpts_l_ptr, double** tdA_dyn_fpts_r_ptr, double** detjac_dyn_fpts_l_ptr, double** detjac_dyn_fpts_r_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** sgsf_fpts_l_ptr, double** sgsf_fpts_r_ptr, int riemann_solve_type, int vis_riemann_solve_type, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int equation, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries*/
void evaluate_boundaryConditions_viscFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** grad_disu_fpts_l_ptr, double** norm_tconvisf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, double** loc_fpts_ptr, double** loc_dyn_fpts_ptr, double** sgsf_fpts_ptr, int* boundary_type, double* bdy_params, double** delta_disu_fpts_l_ptr, int riemann_solve_type, int vis_riemann_solve_type, double R_ref, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, double time_bound, int equation, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t);

/*! wrapper for gpu kernel to calculate source term for SA turbulence model at solution points */
void calc_src_upts_SA_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* in_disu_upts_ptr, double* grad_disu_upts_ptr, double* wall_distance_mag_ptr, double* src_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, int in_fix_vis, double in_c_v1, double in_c_v2, double in_c_v3, double in_c_b1, double in_c_b2, double in_c_w2, double in_c_w3, double in_omega, double in_Kappa);

void evaluate_body_force_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* src_upts_ptr, double* body_force_ptr);

#ifdef _MPI

void pack_out_buffer_disu_gpu_kernel_wrapper(int n_fpts_per_inter,int n_inters,int n_fields,double** disu_fpts_l_ptr, double* out_buffer_disu_ptr);

void pack_out_buffer_grad_disu_gpu_kernel_wrapper(int n_fpts_per_inter,int n_inters,int n_fields,int n_dims, double** grad_disu_fpts_l_ptr, double* out_buffer_grad_disu_ptr);

void pack_out_buffer_sgsf_gpu_kernel_wrapper(int n_fpts_per_inter,int n_inters,int n_fields,int n_dims, double** sgsf_fpts_l_ptr, double* out_buffer_sgsf_ptr);

void calculate_common_invFlux_mpi_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconinvf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, int riemann_solve_type, double** delta_disu_fpts_l_ptr, double gamma, double pen_fact,  int viscous, int motion, int vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int turb_model);

void calculate_common_viscFlux_mpi_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconvisf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** sgsf_fpts_l_ptr, double** sgsf_fpts_r_ptr, int riemann_solve_type, int vis_riemann_solve_type, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t);


#endif

void bespoke_SPMV(int m, int n, int n_fields, int n_eles, double* opp_ell_data_ptr, int* opp_ell_indices_ptr, int nnz_per_row, double* b_ptr, double *c_ptr, int cell_type, int order, int add_flag);

/*! wrapper for gpu kernel to calculate Leonard tensors for similarity model */
void calc_similarity_model_kernel_wrapper(int flag, int n_fields, int n_upts_per_ele, int n_eles, int n_dims, double* disu_upts_ptr, double* disuf_upts_ptr, double* uu_ptr, double* ue_ptr, double* Leonard_mom_ptr, double* Leonard_energy_ptr);

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void rigid_motion_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* shape_dyn, double* motion_params, double rk_time);

/*! wrapper for gpu kernel to */
void perturb_shape_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* shape_dyn, double rk_time);

/*! wrapper for gpu kernel to */
void perturb_shape_points_gpu_kernel_wrapper(int n_dims, int n_verts, double* xv, double* xv_0, double rk_time);

/*! wrapper for gpu kernel to */
void push_back_xv_kernel_wrapper(int n_dims, int n_verts, double* xv_1, double* xv_2);

/*! Wrapper for gpu kernel to calculate the grid velocity at the shape points using backward-difference formula */
void calc_rigid_grid_vel_spts_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* motion_params, double* grid_vel, double rk_time);

/*! Wrapper for gpu kernel to calculate the grid velocity at the shape points using backward-difference formula */
void calc_perturb_grid_vel_spts_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* grid_vel, double rk_time);

/*! Wrapper for gpu kernel to calculate the grid velocity at the shape points using backward-difference formula */
void calc_grid_vel_spts_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape_dyn, double* grid_vel, double dt);

/*! Wrapper for gpu kernel to interpolate the grid veloicty at the shape points to either the solution or flux points */
void eval_grid_vel_pts_kernel_wrapper(int n_dims, int n_eles, int n_pts_per_ele, int max_n_spts_per_ele, int* n_spts_per_ele, double* nodal_s_basis_pts, double* grid_vel_spts, double* grid_vel_pts);

/*! Wrapper for GPU kernel to update coordinate transformation at flux points for moving grids */
void set_transforms_dynamic_fpts_kernel_wrapper(int n_fpts_per_ele, int n_eles, int n_dims, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_fpts_ptr, double* J_dyn_fpts_ptr, double* JGinv_fpts_ptr, double* JGinv_dyn_fpts_ptr, double* tdA_dyn_fpts_ptr, double* norm_fpts_ptr, double* norm_dyn_fpts_ptr, double *d_nodal_s_basis_fpts, double *shape_dyn);

/*! Wrapper for GPU kernel to update coordinate transformation at solution points for moving grids */
void set_transforms_dynamic_upts_kernel_wrapper(int n_upts_per_ele, int n_eles, int n_dims, int max_n_spts_per_ele, int *n_spts_per_ele, double* J_upts_ptr, double *J_dyn_upts_ptr, double *JGinv_upts_ptr, double *JGinv_dyn_upts_ptr, double *d_nodal_s_basis_upts, double *shape_dyn);

/*! Wrapper for GPU kernel to */
void push_back_shape_dyn_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int n_levels, int* n_spts_per_ele, double* shape_dyn);

/*! Wrapper for shock capturing */
void shock_capture_concentration_gpu_kernel_wrapper(int in_n_eles, int in_n_upts_per_ele, int in_n_fields, int in_order, int in_ele_type, int in_artif_type, double s0, double kappa, double* in_disu_upts_ptr, double* in_inv_vandermonde_ptr, double* in_inv_vandermonde2D_ptr, double* in_vandermonde2D_ptr, double* concentration_array_ptr, double* out_sensor, double* sigma);
