/*!
 * \file global.cpp
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

#include "../include/global.h"
#include "../include/array.h"

using namespace std;

input run_input;
double pi=3.141592654;

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
  if (Arows == 0 || Bcols == 0 || (alpha == 0. || Acols == 0) && beta == 1.)  {
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
