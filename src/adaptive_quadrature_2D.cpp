#include "../include/adaptive_quadrature_2D.h"

static double XVALUE; // global variable used when evaluating integral along y

static double (*INTEGRAND_2D)(double,double);
static double (*Y_LOWER_LIMIT) (double);
static double (*Y_UPPER_LIMIT) (double);
static double X_LOWER_LIMIT;
static double X_UPPER_LIMIT;
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

  return temp_result;
}

// Integral over x and y
static double integral_along_x() {
  double result,errest,flag;
  int nofun;

  quanc8(integral_along_y, X_LOWER_LIMIT, X_UPPER_LIMIT,
         ABSERR, RELERR,
         result, errest, nofun,flag);

  return result;
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
  X_LOWER_LIMIT = a;
  X_UPPER_LIMIT = b;
  Y_LOWER_LIMIT = y_lower_limit;
  Y_UPPER_LIMIT = y_upper_limit;
  ABSERR = abserr;
  RELERR = relerr;

  result = integral_along_x();

  nofun = NOFUN;

}


//  ------------------------------------------------------------------//
//  -------Implementation of 3D Quadrature ---------------------------//
//  ------------------------------------------------------------------//
static double (*INTEGRAND_3D)(double, double, double);
static double (*Z_LOWER_LIMIT) (double, double);
static double (*Z_UPPER_LIMIT) (double, double);

static double Y_INTEGRAL_VALUE;
static double YVALUE;

static double integrand_fixed_xy(double z) {
  return INTEGRAND_3D(XVALUE, YVALUE, z);
}

// Evaluate integral along lines of z (called by integral along y)
static double integral_along_z(double x, double y) {
  YVALUE = y;
  double temp_result,errest,flag;
  int nofun;

  quanc8(integrand_fixed_xy,
         Z_LOWER_LIMIT(x, y), Z_UPPER_LIMIT(x, y),
         ABSERR, RELERR,
         temp_result, errest,nofun, flag);

  NOFUN += nofun; // keep track of total number of function evaluations
  return temp_result;
}

/* Function used to calculate triple integrals
 * Then calls quanc8 to evaluate integral over x, which depends on integral
 * over y, which depends on integral over z
 */
void quad3(double (*fun) (double x, double y, double z),
           double a, double b,
           double (*y_lower_limit) (double), double (*y_upper_limit) (double),
           double (*z_lower_limit) (double, double), double (*z_upper_limit) (double, double),
           double abserr, double relerr,
           double& result, double& errest, int& nofun, double& flag) {
  INTEGRAND_3D = fun;
  INTEGRAND_2D = integral_along_z;
  X_LOWER_LIMIT = a;
  X_UPPER_LIMIT = b;
  Y_LOWER_LIMIT = y_lower_limit;
  Y_UPPER_LIMIT = y_upper_limit;
  Z_LOWER_LIMIT = z_lower_limit;
  Z_UPPER_LIMIT = z_upper_limit;

  ABSERR = 1;//abserr;
  RELERR = 1;//relerr;

  result = integral_along_x();

  nofun = NOFUN;
}
