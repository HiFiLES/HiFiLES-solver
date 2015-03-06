/*!
 * \file global.h
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

#include "input.h"

/*! input 'run_input' has global scope */
extern input run_input;

/*! double 'pi' has global scope */
extern const double pi;

/** enumeration for cell type */
enum CTYPE {
    TRI     = 0,
    QUAD    = 1,
    TET     = 2,
    PRISM   = 3,
    HEX     = 4,
    PYRAMID = 5
};

/** enumeration for boundary conditions */
enum BCFLAG {
  SUB_IN_SIMP   = 1,
  SUB_OUT_SIMP  = 2,
  SUB_IN_CHAR   = 3,
  SUB_OUT_CHAR  = 4,
  SUP_IN        = 5,
  SUP_OUT       = 6,
  SLIP_WALL     = 7,
  CYCLIC        = 9,
  ISOTHERM_FIX  = 11,
  ADIABAT_FIX   = 12,
  ISOTHERM_MOVE = 13,
  ADIABAT_MOVE  = 14,
  CHAR          = 15,
  SLIP_WALL_DUAL= 16,
  AD_WALL       = 50
};

enum MOTION_TYPE {
  STATIC_MESH       = 0,
  LINEAR_ELASTICITY = 1,
  RIGID_MOTION      = 2,
  PERTURB_TEST      = 3,
  BLENDING          = 4
};

/** enumeration for mesh motion type */
enum {MOTION_DISABLED, MOTION_ENABLED};

/*! environment variable specifying location of HiFiLES repository */
extern const char* HIFILES_DIR;

/*! routine that mimics BLAS dgemm */
int dgemm(int Arows, int Bcols, int Acols, double alpha, double beta, double* a, double* b, double* c);

/*! routine that mimics BLAS daxpy */
int daxpy(int n, double alpha, double *x, double *y);
