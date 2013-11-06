/*!
 * \file int_inters.h
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
#include "int_inters.h"
#include "array.h"
#include "solution.h"

struct solution; // forwards declaration

class int_inters: public inters
{
	public:
	
	// #### constructors ####
	
	// default constructor
	
	int_inters();

  // default destructor
 
  ~int_inters(); 

	// #### methods ####
	
	/*! setup inters */
	void setup(int in_n_inters, int in_inter_type, int in_run_type);

	/*! set interior interface */
	void set_interior(int in_inter, int in_ele_type_l, int in_ele_type_r, int in_ele_l, int in_ele_r, int in_local_inter_l, int in_local_inter_r, int rot_tag, int in_run_type, struct solution* FlowSol);

	/*! move all from cpu to gpu */
	void mv_all_cpu_gpu(void);

	/*! calculate normal transformed continuous inviscid flux at the flux points */
	void calc_norm_tconinvf_fpts(void);

	/*! calculate normal transformed continuous viscous flux at the flux points */
	void calc_norm_tconvisf_fpts(void);
	
	/*! calculate delta in transformed discontinuous solution at flux points */
	void calc_delta_disu_fpts(void);
	
	protected:

	// #### members ####
  //
	array<double*> disu_fpts_r;
	array<double*> delta_disu_fpts_r;
	array<double*> norm_tconf_fpts_r;
	//array<double*> norm_tconvisf_fpts_r;
	array<double*> detjac_fpts_r;
	array<double*> mag_tnorm_dot_inv_detjac_mul_jac_fpts_r;
	array<double*> grad_disu_fpts_r;
	
};
