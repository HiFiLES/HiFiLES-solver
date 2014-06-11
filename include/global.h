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
extern double pi;

/*! environment variable specifying location of HiFiLES repository */
extern const char* HIFILES_DIR;

/*! routine that mimics BLAS dgemm */
int dgemm(int Arows, int Bcols, int Acols, double alpha, double beta, double* a, double* b, double* c);

/*! routine that mimics BLAS daxpy */
int daxpy(int n, double alpha, double *x, double *y);
