/*!
 * \file int_inters.h
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
  void setup(int in_n_inters, int in_inter_type);

  /*! set interior interface */
  void set_interior(int in_inter, int in_ele_type_l, int in_ele_type_r, int in_ele_l, int in_ele_r, int in_local_inter_l, int in_local_inter_r, int rot_tag, struct solution* FlowSol);

  /*! move all from cpu to gpu */
  void mv_all_cpu_gpu(void);

  /*! calculate normal transformed continuous inviscid flux at the flux points */
  void calculate_common_invFlux(void);

  /*! calculate normal transformed continuous viscous flux at the flux points */
  void calculate_common_viscFlux(void);

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
  array<double*> tdA_fpts_r;
  array<double*> grad_disu_fpts_r;

  // Dynamic grid variables:
  array<double*> ndA_dyn_fpts_r;
  array<double*> J_dyn_fpts_r;
  array<double*> disu_GCL_fpts_r;
  array<double*> norm_tconf_GCL_fpts_r;

  double temp_u_GCL_r;
  double temp_f_GCL_r;
};
