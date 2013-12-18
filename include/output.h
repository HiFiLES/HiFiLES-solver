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


/*! write a continuous plot file */
void plot_continuous(struct solution* FlowSol);

void plotter_setup(struct solution* FlowSol);

/*! write an output file in Tecplot ASCII format.  Used in run mode. */
void write_tec(int in_file_num, struct solution* FlowSol);

/*! write an output file in Tecplot binary format.  Used in plot mode.*/
void write_tec_bin(int in_file_num, struct solution* FlowSol);

/*! write an output file in VTK ASCII format. Used in run mode. */
void write_vtu(int in_file_num, struct solution* FlowSol);

/*! write an output file in VTK binary format. Used in plot mode. */
void write_vtu_bin(int in_file_num, struct solution* FlowSol);

/*! writing a restart file */
void write_restart(int in_file_num, struct solution* FlowSol);

/*! compute forces on wall faces*/
void compute_forces(int in_file_num, double in_time, struct solution* FlowSol);

/*! compute diagnostics */
void CalcDiagnostics(int in_file_num, double in_time, struct solution* FlowSol);

/*! compute error */
void compute_error(int in_file_num, struct solution* FlowSol);

/*! monitor convergence of residual */
int monitor_residual(int in_file_num, struct solution* FlowSol);

/*! check if the solution is bounded !*/
void check_stability(struct solution* FlowSol);
