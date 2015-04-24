#include<math.h>

void quanc8(double(*fun)(double), double a, double b,
            double abserr, double relerr,
            double& result, double& errest, int& nofun,double& flag)
/*
   estimate the integral of fun(x) from a to b to a user provided tolerance.
   an automatic adaptive routine based on the 8-panel newton-cotes rule.

input:
   fun     the name of the integrand function subprogram fun(x).
   a       the lower limit of integration.
   b       the upper limit of integration.(b may be less than a.)
   relerr  a relative error tolerance. (should be non-negative)
   abserr  an absolute error tolerance. (should be non-negative)

output:
   result  an approximation to the integral hopefully satisfying the
           least stringent of the two error tolerances.
   errest  an estimate of the magnitude of the actual error.
   nofun   the number of function values used in calculation of result.
   flag    a reliability indicator.  if flag is zero, then result
           probably satisfies the error tolerance.  if flag is
           xxx.yyy , then  xxx = the number of intervals which have
           not converged and  0.yyy = the fraction of the interval
           left to do when the limit on  nofun  was approached.

comments:
   Alex Godunov (February 2007)
   the program is based on a fortran version of program quanc8.f
*/
{
    double w0,w1,w2,w3,w4,area,x0,f0,stone,step,cor11,temp;
    double qprev,qnow,qdiff,qleft,esterr,tolerr;
    double qright[32], f[17], x[17], fsave[9][31], xsave[9][31];
    double dabs,dmax1;
    int    levmin,levmax,levout,nomax,nofin,lev,nim,i,j;
    int    key;

//  ***   stage 1 ***   general initialization

    levmin = 1;
    levmax = 30;
    levout = 6;
    nomax = 5000;
    nofin = nomax - 8*(levmax - levout + 128);
//  trouble when nofun reaches nofin

    w0 =   3956.0 / 14175.0;
    w1 =  23552.0 / 14175.0;
    w2 =  -3712.0 / 14175.0;
    w3 =  41984.0 / 14175.0;
    w4 = -18160.0 / 14175.0;

//  initialize running sums to zero.

    flag   = 0.0;
    result = 0.0;
    cor11  = 0.0;
    errest = 0.0;
    area   = 0.0;
    nofun  = 0;
    if (a == b) return;

//  ***   stage 2 ***   initialization for first interval

    lev = 0;
    nim = 1;
    x0 = a;
    x[16] = b;
    qprev  = 0.0;
    f0 = fun(x0);
    stone = (b - a) / 16.0;
    x[8]  =  (x0    + x[16])   / 2.0;
    x[4]  =  (x0    + x[8])    / 2.0;
    x[12] =  (x[8]  + x[16])   / 2.0;
    x[2]  =  (x0    + x[4])    / 2.0;
    x[6]  =  (x[4]  + x[8])    / 2.0;
    x[10] =  (x[8]  + x[12])   / 2.0;
    x[14] =  (x[12] + x[16])   / 2.0;
    for (j=2; j<=16; j = j+2)
    {
      f[j] = fun(x[j]);
    }
    nofun = 9;

//  ***   stage 3 ***   central calculation

    while(nofun <= nomax)
    {
      x[1] = (x0 + x[2]) / 2.0;
      f[1] = fun(x[1]);
      for(j = 3; j<=15; j = j+2)
        {
          x[j] = (x[j-1] + x[j+1]) / 2.0;
          f[j] = fun(x[j]);
        }
      nofun = nofun + 8;
      step  = (x[16] - x0) / 16.0;
      qleft  = (w0*(f0 + f[8])  + w1*(f[1]+f[7])  + w2*(f[2]+f[6])
             + w3*(f[3]+f[5])  +  w4*f[4]) * step;
      qright[lev+1] = (w0*(f[8]+f[16])+w1*(f[9]+f[15])+w2*(f[10]+f[14])
                    + w3*(f[11]+f[13]) + w4*f[12]) * step;
      qnow  = qleft + qright[lev+1];
      qdiff = qnow  - qprev;
      area  = area  + qdiff;

//  ***   stage 4 *** interval convergence test

      esterr = fabs(qdiff) / 1023.0;
      if(abserr >= relerr*fabs(area))
        tolerr = abserr;
        else
        tolerr = relerr*fabs(area);
      tolerr = tolerr*(step/stone);

// multiple logic conditions for the convergence test
      key = 1;
      if (lev < levmin) key = 1;
        else if (lev >= levmax)
        key = 2;
        else if (nofun > nofin)
        key = 3;
        else if (esterr <= tolerr)
        key = 4;
        else
        key = 1;

      switch (key) {
// case 1 ********************************* (mark 50)
      case 1:
//      ***   stage 5   ***   no convergence
//      locate next interval.
        nim = 2*nim;
        lev = lev+1;

//      store right hand elements for future use.
        for(i=1; i<=8; i=i+1)
        {
          fsave[i][lev] = f[i+8];
          xsave[i][lev] = x[i+8];
        }

//      assemble left hand elements for immediate use.
        qprev = qleft;
        for(i=1; i<=8; i=i+1)
        {
          j = -i;
          f[2*j+18] = f[j+9];
          x[2*j+18] = x[j+9];
        }
        continue;  // go to start of stage 3 "central calculation"
      break;

// case 2 ********************************* (mark 62)
      case 2:
        flag = flag + 1.0;
      break;
// case 3 ********************************* (mark 60)
      case 3:
//    ***   stage 6   ***   trouble section
//    number of function values is about to exceed limit.
        nofin = 2*nofin;
        levmax = levout;
        flag = flag + (b - x0) / (b - a);
      break;
// case 4 ********************************* (continue mark 70)
      case 4:
      break;
// default ******************************** (continue mark 70)
      default:
      break;
// end case section ***********************
}

//   ***   stage 7   ***   interval converged
//   add contributions into running sums.

    result = result + qnow;
    errest = errest + esterr;
    cor11  = cor11  + qdiff / 1023.0;

//  locate next interval

    while (nim != 2*(nim/2))
    {
      nim = nim/2;
      lev = lev-1;
    }
    nim = nim + 1;
    if (lev <= 0) break;  // may exit futher calculation

//  assemble elements required for the next interval.

    qprev = qright[lev];
    x0 = x[16];
    f0 = f[16];
    for (i =1; i<=8; i=i+1)
    {
      f[2*i] = fsave[i][lev];
      x[2*i] = xsave[i][lev];
    }
}
//  *** end stage 3 ***   central calculation

//  ***   stage 8   ***   finalize and return

    result = result + cor11;

//  make sure errest not less than roundoff level.

    if (errest == 0.0) return;
    do
    {
     temp = fabs(result) + errest;
     errest = 2.0*errest;
    }
    while (temp == fabs(result));

    return;
}
