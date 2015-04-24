#ifndef ADAPTIVE_QUADRATURE_H
#define ADAPTIVE_QUADRATURE_H

void quanc8(double(*fun)(double), double a, double b,
            double abserr, double relerr,
            double& result, double& errest, int& nofun,double& flag);

void quad2(double (*fun) (double x, double y),
             double a, double b,
           double (*y_lower_limit) (double), double (*y_upper_limit) (double),
             double abserr, double relerr,
             double& result, double& errest, int& nofun, double& flag);

#endif // ADAPTIVE_QUADRATURE_H
