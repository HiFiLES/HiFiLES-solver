/*!
 * \file cuda_kernels.h
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

void RK45_update_kernel_wrapper(int in_n_upts_per_ele,int in_n_dims,int in_n_fields,int in_n_eles,double* in_disu0_upts_ptr,double* in_disu1_upts_ptr,double* in_div_tconf_upts_ptr, double* in_detjac_upts_ptr, double in_rk4a, double in_rk4b, double in_dt, double in_const_src_term);

void RK11_update_kernel_wrapper(int in_n_upts_per_ele,int in_n_dims,int in_n_fields,int in_n_eles,double* in_disu0_upts_ptr,double* in_div_tconf_upts_ptr, double* in_detjac_upts_ptr, double in_dt, double in_const_src_term);

/*! wrapper for gpu kernel to calculate transformed discontinuous inviscid flux at solution points */
void calc_tdisinvf_upts_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisinvf_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr,double in_gamma, int equation, double wave_speed_x, double wave_speed_y, double wave_speed_z);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points */

void calc_norm_tconinvf_fpts_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_norm_tconinvf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double** in_delta_disu_fpts_r_ptr, double in_gamma, double in_pen_fact, int in_viscous, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries*/
void calc_norm_tconinvf_fpts_boundary_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_loc_fpts_ptr, int* in_boundary_type, double* in_bdy_params, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_R_ref, int in_viscous, int in_vis_riemann_solve_type, double in_time_bound, double in_wave_speed_x, double in_wave_speed_y, double in_wave_speed_z, double in_lambda, int in_equation);


void transform_grad_disu_upts_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, int equation);

/*! wrapper for gpu kernel to calculate transformed discontinuous viscous flux at solution points */
void calc_tdisvisf_upts_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, int in_ele_type, int in_order, double in_filter_ratio, int LES, int sgs_model, int wall_model, double in_wall_thickness, double* in_wall_dist_ptr, double* in_twall_ptr, double* in_Leonard_mom_ptr, double* in_Leonard_energy_ptr, double* in_disu_upts_ptr, double* out_tdisvisf_upts_ptr, double* out_sgsf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double diff_coeff);

/*! wrapper for gpu kernel to calculate corrected gradient of solution at flux points */
/*
void calc_cor_grad_disu_fpts_gpu_kernel_wrapper(int in_n_fpts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_disu_fpts_ptr, double* in_delta_disu_fpts_ptr, double* out_grad_disu_fpts_ptr, double* in_detjac_fpts_ptr, double* in_inv_detjac_mul_jac_fpts_ptr, double* in_tgrad_detjac_fpts_ptr);
*/

/*! wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points */
void calc_norm_tconvisf_fpts_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconvisf_fpts_l_ptr, double** in_norm_tconvisf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_sgsf_fpts_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double diff_coeff);

/*! wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries*/
void calc_norm_tconvisf_fpts_boundary_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_grad_disu_fpts_l_ptr, double** in_norm_tconvisf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_loc_fpts_ptr, double** in_sgsf_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_R_ref, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_time_bound, int in_equation, double in_diff_coeff);

#ifdef _MPI

void pack_out_buffer_disu_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,double** in_disu_fpts_l_ptr, double* in_out_buffer_disu_ptr);

void pack_out_buffer_grad_disu_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,int in_n_dims, double** in_grad_disu_fpts_l_ptr, double* in_out_buffer_grad_disu_ptr);

void calc_norm_tconinvf_fpts_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr,int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_pen_fact,  int in_viscous, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda);

void calc_norm_tconvisf_fpts_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconvisf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_diff_coeff);


#endif

void bespoke_SPMV(int m, int n, int n_fields, int n_eles, double* opp_ell_data_ptr, int* opp_ell_indices_ptr, int nnz_per_row, double* b_ptr, double *c_ptr, int cell_type, int order, int add_flag);

/*! wrapper for gpu kernel to calculate Leonard tensors for similarity model */
void calc_similarity_model_kernel_wrapper(int flag, int in_n_fields, int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* in_uu_ptr, double* in_ue_ptr, double* in_Leonard_mom_ptr, double* in_Leonard_energy_ptr);
