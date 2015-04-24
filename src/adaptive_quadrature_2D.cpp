#include "../include/adaptive_quadrature_2D.h"

static double XVALUE; // global variable used when evaluating integral along y

static double (*INTEGRAND_2D)(double,double);
static double (*Y_LOWER_LIMIT) (double);
static double (*Y_UPPER_LIMIT) (double);
static double ABSERR = 1E-8, RELERR = 1E-7;
static int NOFUN = 0;

static double integrand_fixed_x(double y) {
  return INTEGRAND_2D(XVALUE,y);
}

// Integral evaluated at given value of x
static double integral_along_y(double x) {
  XVALUE = x;
  double temp_result,errest,flag;
  int nofun;

  quanc8(integrand_fixed_x,
         Y_LOWER_LIMIT(x), Y_UPPER_LIMIT(x),
         ABSERR, RELERR,
         temp_result, errest,nofun, flag);
  NOFUN += nofun; // keep track of total number of function evaluations
  return temp_result;
}

/* Function used to calculate double integrals
 * Creates a functor to evaluate the integral along a line
 * Then calls quanc8 with this functor to evaluate the double integral
 */
void quad2(double (*fun) (double x, double y),
           double a, double b,
           double (*y_lower_limit) (double), double (*y_upper_limit) (double),
           double abserr, double relerr,
           double& result, double& errest, int& nofun, double& flag) {
  INTEGRAND_2D = fun;
  Y_LOWER_LIMIT = y_lower_limit;
  Y_UPPER_LIMIT = y_upper_limit;
  ABSERR = abserr;
  RELERR = relerr;

  quanc8(integral_along_y, a, b,
         ABSERR, RELERR,
         result, errest, nofun,flag);

  nofun = NOFUN;

}


