/*!
 * \file global.cpp
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

#include "../include/global.h"
#include "../include/array.h"
#include <math.h>

using namespace std;

input run_input;
const double pi=4*atan(1);

const char* HIFILES_DIR = getenv("HIFILES_HOME");

/*! Routine to multiply matrices similar to BLAS's dgemm */
int dgemm(int Arows, int Bcols, int Acols, double alpha, double beta, double* a, double* b, double* c)
{
  /* Routine similar to blas dgemm but does not allow for transposes.

     Performs C := alpha*A*B + beta*C

     Just as an alternative to the BLAS routines in case a standalone implementation is required

     Arows - No. of rows of matrices A and C
     Bcols - No. of columns of matrices B and C
     Acols - No. of columns of A or No. of rows of B
  */

  #define A(I,J) a[(I) + (J)*Arows]
  #define B(I,J) b[(I) + (J)*Acols]
  #define C(I,J) c[(I) + (J)*Arows]

  int i,j,l;
  double temp;

  // Quick return if possible
  if (Arows == 0 || Bcols == 0 || ((alpha == 0. || Acols == 0) && beta == 1.))  {
      return 0;
  }

  // If alpha is zero.

  if (alpha == 0.) {
    if (beta == 0.) {
      for (j = 0; j < Bcols; j++)
        for (i = 0; i < Arows; i++)
          C(i,j) = 0.;
    }

    else {
      for (j = 0; j < Bcols; j++)
        for (i = 0; i < Arows; i++)
                  C(i,j) = beta * C(i,j);
    }
    return 0;
  }

  // Otherwise, perform full operation
  for (j = 0; j < Bcols; j++) {

    if (beta == 0.) {
      for (i = 0; i < Arows; i++)
        C(i,j) = 0.;
    }

    else if (beta != 1.) {
      for (i = 0; i < Arows; i++)
              C(i,j) = beta * C(i,j);
    }

    for (l = 0; l < Acols; l++) {
        temp = alpha*B(l,j);

        for (i = 0; i < Arows; i++)
          C(i,j) += temp * A(i,l);
    }
  }

  return 0;
}

/*! Routing to compute alpha*x + y for vectors x and y - similar to BLAS's daxpy */
int daxpy(int n, double alpha, double *x, double *y)
{
  // Error
  if(n == 0)
      return 1;

  // Very straightforward implementation - can be improved
  for(int i=0; i<n; i++)
    y[i] += alpha*x[i];

  return 0;
}

