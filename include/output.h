/*!
 * \file output.h
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
#include <string>
#include "input.h"
#include "eles.h"
#include "eles_tris.h"
#include "eles_quads.h"
#include "eles_hexas.h"
#include "eles_tets.h"
#include "eles_pris.h"
#include "int_inters.h"
#include "bdy_inters.h"
#include "solution.h"

#ifdef _MPI
#include "mpi.h"
#include "mpi_inters.h"
#endif

#ifdef _GPU
#include "util.h"
#endif

/*! write an output file in Tecplot ASCII format */
void write_tec(int in_file_num, struct solution* FlowSol);

/*! write an output file in VTK ASCII format */
void write_vtu(int in_file_num, struct solution* FlowSol);

/*! writing a restart file */
void write_restart(int in_file_num, struct solution* FlowSol);

/*! compute forces on wall faces*/
void CalcForces(int in_file_num, struct solution* FlowSol);

/*! compute integral diagnostic quantities */
void CalcIntegralQuantities(int in_file_num, struct solution* FlowSol);

/*! Calculate time averaged diagnostic quantities */
void CalcTimeAverageQuantities(struct solution* FlowSol);

/*! compute error */
void compute_error(int in_file_num, struct solution* FlowSol);

/*! calculate residual */
void CalcNormResidual(struct solution* FlowSol);

/*! monitor convergence of residual */
void HistoryOutput(int in_file_num, clock_t init, ofstream *write_hist, struct solution* FlowSol);

/*! check if the solution is bounded !*/
void check_stability(struct solution* FlowSol);

#ifdef _GPU
/*! copy solution and gradients from GPU to CPU for above routines !*/
void CopyGPUCPU(struct solution* FlowSol);
#endif

