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

void RK45_update_kernel_wrapper(int in_n_upts_per_ele,int in_n_dims,int in_n_fields,int in_n_eles,double* in_disu0_upts_ptr,double* in_disu1_upts_ptr,double* in_div_tconf_upts_ptr, double* in_detjac_upts_ptr, double in_rk4a, double in_rk4b, double in_dt, double in_const_src_term);

void RK11_update_kernel_wrapper(int in_n_upts_per_ele,int in_n_dims,int in_n_fields,int in_n_eles,double* in_disu0_upts_ptr,double* in_div_tconf_upts_ptr, double* in_detjac_upts_ptr, double in_dt, double in_const_src_term);

/*! wrapper for gpu kernel to calculate transformed discontinuous inviscid flux at solution points */
void evaluate_invFlux_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisinvf_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, double* in_grid_vel_upts_ptr, double in_gamma, int in_motion, int equation, double wave_speed_x, double wave_speed_y, double wave_speed_z);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points */

void calculate_common_invFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_norm_tconinvf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_tdA_dyn_fpts_r_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double **in_grid_vel_fpts_ptr, int in_riemann_solve_type, double **in_delta_disu_fpts_l_ptr, double **in_delta_disu_fpts_r_ptr, double in_gamma, double in_pen_fact, int in_viscous, int in_motion, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries*/
void evaluate_boundaryConditions_invFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_loc_fpts_ptr, double** in_loc_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, int* in_boundary_type, double* in_bdy_params, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_R_ref, int in_viscous, int in_motion, int in_vis_riemann_solve_type, double in_time_bound, double in_wave_speed_x, double in_wave_speed_y, double in_wave_speed_z, double in_lambda, int in_equation);


void transform_grad_disu_upts_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, int equation, int in_motion);

/*! wrapper for gpu kernel to calculate transformed discontinuous viscous flux at solution points */
void evaluate_viscFlux_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, int in_ele_type, int in_order, double in_filter_ratio, int LES, int in_motion, int sgs_model, int wall_model, double in_wall_thickness, double* in_wall_dist_ptr, double* in_twall_ptr, double* in_Leonard_mom_ptr, double* in_Leonard_energy_ptr, double* in_turb_visc_ptr, double* in_dynamic_coeff_ptr, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* out_tdisvisf_upts_ptr, double* out_sgsf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_grad_disuf_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double diff_coeff);

/*! wrapper for gpu kernel to calculate corrected gradient of solution at flux points */
/*
void extrapolate_corrected_gradient_gpu_kernel_wrapper(int in_n_fpts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_disu_fpts_ptr, double* in_delta_disu_fpts_ptr, double* out_grad_disu_fpts_ptr, double* in_detjac_fpts_ptr, double* in_JGinv_fpts_ptr, double* in_tgrad_detjac_fpts_ptr);
*/

/*! wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points */
void calculate_common_viscFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconvisf_fpts_l_ptr, double** in_norm_tconvisf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_tdA_dyn_fpts_r_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_sgsf_fpts_l_ptr, double** in_sgsf_fpts_r_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double diff_coeff, int LES, int in_motion);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries*/
void evaluate_boundaryConditions_viscFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_grad_disu_fpts_l_ptr, double** in_norm_tconvisf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, double** in_loc_fpts_ptr, double** in_loc_dyn_fpts_ptr, double** in_sgsf_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_R_ref, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_time_bound, int in_equation, double in_diff_coeff, int LES, int in_motion);

#ifdef _MPI

void pack_out_buffer_disu_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,double** in_disu_fpts_l_ptr, double* in_out_buffer_disu_ptr);

void pack_out_buffer_grad_disu_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,int in_n_dims, double** in_grad_disu_fpts_l_ptr, double* in_out_buffer_grad_disu_ptr);

void pack_out_buffer_sgsf_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,int in_n_dims, double** in_sgsf_fpts_l_ptr, double* in_out_buffer_sgsf_ptr);

void calculate_common_invFlux_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_pen_fact,  int in_viscous, int in_motion, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda);

void calculate_common_viscFlux_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconvisf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_sgsf_fpts_l_ptr, double** in_sgsf_fpts_r_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_diff_coeff, int LES, int in_motion);


#endif

void bespoke_SPMV(int m, int n, int n_fields, int n_eles, double* opp_ell_data_ptr, int* opp_ell_indices_ptr, int nnz_per_row, double* b_ptr, double *c_ptr, int cell_type, int order, int add_flag);

/*! wrapper for gpu kernel to calculate Leonard tensors for similarity model */
void calc_similarity_model_kernel_wrapper(int flag, int in_n_fields, int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* in_uu_ptr, double* in_ue_ptr, double* in_Leonard_mom_ptr, double* in_Leonard_energy_ptr);

/*! Wrapper for GPU kernel to update coordinate transformation at flux points for moving grids */
void set_transforms_dynamic_fpts_kernel_wrapper(int in_n_fpts_per_ele, int in_n_eles, int in_n_dims, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_fpts_ptr, double* J_dyn_fpts_ptr, double* JGinv_fpts_ptr, double* JGinv_dyn_fpts_ptr, double* tdA_dyn_fpts_ptr, double* norm_fpts_ptr, double* norm_dyn_fpts_ptr, double *d_nodal_s_basis_fpts, double *shape_dyn);

/*! Wrapper for GPU kernel to update coordinate transformation at solution points for moving grids */
void set_transforms_dynamic_upts_kernel_wrapper(int in_n_upts_per_ele, int in_n_eles, int in_n_dims, int max_n_spts_per_ele, int *n_spts_per_ele, double* J_upts_ptr, double *J_dyn_upts_ptr, double *JGinv_upts_ptr, double *JGinv_dyn_upts_ptr, double *d_nodal_s_basis_upts, double *shape_dyn);
