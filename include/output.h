/*!
 * \file output.h
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

/*! write an output file containing evenly spaced points for calculating spectra */
void write_ftpoints(int in_file_num, struct solution* FlowSol);

/*! writing a restart file */
void write_restart(int in_file_num, struct solution* FlowSol);

/*! compute forces on wall faces*/
void CalcForces(int in_file_num, struct solution* FlowSol);

/*! compute integral diagnostic quantities */
void CalcIntegralQuantities(int in_file_num, struct solution* FlowSol);

/*! compute error */
void compute_error(int in_file_num, struct solution* FlowSol);

/*! monitor convergence of residual */
void CalcNormResidual(struct solution* FlowSol);

/*! monitor convergence of residual */
void HistoryOutput(int in_file_num, clock_t init, ofstream *write_hist, struct solution* FlowSol);

/*! check if the solution is bounded !*/
void check_stability(struct solution* FlowSol);

#ifdef _GPU
/*! copy solution and gradients from GPU to CPU for above routines !*/
void CopyGPUCPU(struct solution* FlowSol);
#endif

