/*!
 * \file funcs.cpp
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

#include <iomanip>
#include <iostream>
#include <cmath>
#include <vector>

#if defined _ACCELERATE_BLAS
#include <Accelerate/Accelerate.h>
#endif

#if defined _MKL_BLAS
#include "mkl.h"
#include "mkl_spblas.h"
#endif

#if defined _STANDARD_BLAS
extern "C"
{
#include "cblas.h"
}
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#if defined _GPU
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "../include/cuda_kernels.h"
#endif


#include "../include/funcs.h"
//#include "../include/Array.h"
#include "../include/cubature_1d.h"
#include "../include/global.h"
#include "../include/eles.h"
#include "../include/eles_tris.h"
#include "../include/adaptive_quadrature_2D.h"
#include "../include/simplex_min_method.h"

using namespace std;

// #### local global variables ####
static Array<double> LOCAL_X0;
static int LOCAL_BASIS_INDEX;
static int LOCAL_ORDER;
static eles *LOCAL_ELE_OF_INTEREST;
static double LOCAL_BESSEL_X; // used in the besselIntegrand function
static double LOCAL_BESSEL_NU;

// #### global functions ####

// evaluate lagrange basis

double eval_lagrange(double in_r, int in_mode, Array<double>& in_loc_pts)
{
  int i;

  double dtemp_0;

  dtemp_0=1.0;

  for(i=0;i<in_loc_pts.get_dim(0);i++)
  {
    if(i!=in_mode)
    {
      dtemp_0=dtemp_0*((in_r-in_loc_pts(i))/(in_loc_pts(in_mode)-in_loc_pts(i)));
    }
  }

  return dtemp_0;
}

// evaluate derivative of lagrange basis

double eval_d_lagrange(double in_r, int in_mode, Array<double>& in_loc_pts)
{
  int i,j;

  double dtemp_0, dtemp_1, dtemp_2;

  dtemp_0=0.0;

  for(i=0;i<in_loc_pts.get_dim(0);i++)
  {
    if(i!=in_mode)
    {
      dtemp_1=1.0;
      dtemp_2=1.0;

      for(j=0;j<in_loc_pts.get_dim(0);j++)
      {
        if(j!=in_mode&&j!=i)
        {
          dtemp_1=dtemp_1*(in_r-in_loc_pts(j));
        }

        if(j!=in_mode)
        {
          dtemp_2=dtemp_2*(in_loc_pts(in_mode)-in_loc_pts(j));
        }
      }

      dtemp_0=dtemp_0+(dtemp_1/dtemp_2);
    }
  }

  return dtemp_0;
}

// evaluate second derivative of lagrange basis

double eval_dd_lagrange(double in_r, int in_mode, Array<double>& in_loc_pts)
{
  int i,j,k;

  double dtemp_0, dtemp_1, dtemp_2;

  dtemp_0=0.0;

  for(i=0;i<in_loc_pts.get_dim(0);i++)
  {
    if(i!=in_mode)
    {
      for(j=0;j<in_loc_pts.get_dim(0);j++)
      {
        if(j!=in_mode)
        {
          dtemp_1 =1.0;
          dtemp_2 =1.0;

          for(k=0;k<in_loc_pts.get_dim(0);k++)
          {
            if(k!=in_mode)
            {
              if( (k!=i) && (k!=j) )
              {
                dtemp_1 = dtemp_1*(in_r-in_loc_pts(k));
              }

              dtemp_2 = dtemp_2*(in_loc_pts(in_mode)-in_loc_pts(k));
            }
          }

          if(j!=i)
          {
            dtemp_0 = dtemp_0 + (dtemp_1/dtemp_2);
          }
        }
      }
    }
  }

  return dtemp_0;
}

// evaluate legendre basis

double eval_legendre(double in_r, int in_mode)
{
  double d_temp;

  if(in_mode==0)
  {
    d_temp=1.0;
  }
  else if(in_mode==1)
  {
    d_temp=in_r;
  }
  else
  {
    d_temp=((2*in_mode-1)*in_r*eval_legendre(in_r,in_mode-1)-(in_mode-1)*eval_legendre(in_r,in_mode-2))/in_mode;
  }

  return d_temp;
}

// evaluate derivative of legendre basis

double eval_d_legendre(double in_r, int in_mode)
{
  int par;
  double d_temp;

  if(in_mode==0)
  {
    d_temp=0;
  }
  else
  {
    if(in_r > -1.0 && in_r < 1.0)
    {
      d_temp=(in_mode*((in_r*eval_legendre(in_r,in_mode))-eval_legendre(in_r,in_mode-1)))/((in_r*in_r)-1.0);
    }
    else
    {
      if(in_r == -1.0)
      {
        d_temp = pow(-1.0, in_mode-1.0)*0.5*in_mode*(in_mode + 1.0);
      }
      if(in_r == 1.0)
      {
        d_temp = 0.5*in_mode*(in_mode + 1.0);
      }
    }
  }

  return d_temp;
}

// evaluate derivative of 1d vcjh basis

double eval_d_vcjh_1d(double in_r, int in_mode, int in_order, double in_eta)
{
  double dtemp_0;

  if(in_mode==0) // left correction function
  {
    if(in_order == 0)
    {
      // if (in_eta != 0)
      //     FatalError("P=0 only compatible with DG. Set VCJH scheme to 1 OR eta to 0.0")

      dtemp_0=0.5*pow(-1.0,in_order)*(eval_d_legendre(in_r,in_order)-((eval_d_legendre(in_r,in_order+1))/(1.0+in_eta)));
    }
    else
    {
      dtemp_0=0.5*pow(-1.0,in_order)*(eval_d_legendre(in_r,in_order)-(((in_eta*eval_d_legendre(in_r,in_order-1))+eval_d_legendre(in_r,in_order+1))/(1.0+in_eta)));
    }
  }
  else if(in_mode==1) // right correction function
  {
    if (in_order == 0)
    {
      //if (in_eta != 0)
      //    FatalError("P=0 only compatible with DG. Set VCJH scheme to 1 OR eta to 0.0")

      dtemp_0=0.5*(eval_d_legendre(in_r,in_order)+((eval_d_legendre(in_r,in_order+1))/(1.0+in_eta)));
    }
    else
    {
      dtemp_0=0.5*(eval_d_legendre(in_r,in_order)+(((in_eta*eval_d_legendre(in_r,in_order-1))+eval_d_legendre(in_r,in_order+1))/(1.0+in_eta)));
    }
  }

  return dtemp_0;
}

double eval_d_ofr_1d(double in_r, int in_mode, int in_order)
{
  double dtemp_0;
  double cVal, aP, eta;

  Array<double> loc_zeros_gL(in_order+2), loc_zeros_gR(in_order+2);

  // Append end points
  loc_zeros_gL(0) = -1.0;
  loc_zeros_gL(in_order+1) = 1.0;
  loc_zeros_gR(0) = -1.0;
  loc_zeros_gR(in_order+1) = 1.0;

  // zeros of correction function
  if(in_order == 1) {
    loc_zeros_gL(1) = -0.324936024976658;

    loc_zeros_gR(1) = 0.324936024976658;
  }
  else if(in_order == 2) {
    loc_zeros_gL(1) = -0.683006983995485;
    loc_zeros_gL(2) = 0.302192635873585;

    loc_zeros_gR(2) = 0.683006983995485;
    loc_zeros_gR(1) = -0.302192635873585;
  }
  else if(in_order == 3) {
    loc_zeros_gL(1) = -0.839877075575685;
    loc_zeros_gL(2) = -0.202221671675099;
    loc_zeros_gL(3) = 0.518569179742482;

    loc_zeros_gR(3) = 0.839877075575685;
    loc_zeros_gR(2) = 0.202221671675099;
    loc_zeros_gR(1) = -0.518569179742482;
  }
  else if(in_order == 4) {
    loc_zeros_gL(1) = -0.856985048185331;
    loc_zeros_gL(2) = -0.447652424946130;
    loc_zeros_gL(3) = 0.180019033571473;
    loc_zeros_gL(4) = 0.638102911955799;

    loc_zeros_gR(4) = 0.856985048185331;
    loc_zeros_gR(3) = 0.447652424946130;
    loc_zeros_gR(2) = -0.180019033571473;
    loc_zeros_gR(1) = -0.638102911955799;
  }
  else if(in_order == 5) {
    loc_zeros_gL(1) = -0.897887439354270;
    loc_zeros_gL(2) = -0.577293821014237;
    loc_zeros_gL(3) = -0.101190259640464;
    loc_zeros_gL(4) = 0.354120543898467;
    loc_zeros_gL(5) = 0.760380824360528;

    loc_zeros_gR(5) = 0.897887439354270;
    loc_zeros_gR(4) = 0.577293821014237;
    loc_zeros_gR(3) = 0.101190259640464;
    loc_zeros_gR(2) = -0.354120543898467;
    loc_zeros_gR(1) = -0.760380824360528;
  }
  else if(in_order == 6) { // P=6 not verified
    loc_zeros_gL(1) = -0.932638621602718;
    loc_zeros_gL(2) = -0.627949285295015;
    loc_zeros_gL(3) = -0.196972255400472;
    loc_zeros_gL(4) = 0.392803242695776;
    loc_zeros_gL(5) = 0.481615260763104;
    loc_zeros_gL(6) = 0.629467212278235;

    loc_zeros_gR(6) = 0.932638621602718;
    loc_zeros_gR(5) = 0.627949285295015;
    loc_zeros_gR(4) = 0.196972255400472;
    loc_zeros_gR(3) = -0.392803242695776;
    loc_zeros_gR(2) = -0.481615260763104;
    loc_zeros_gR(1) = -0.629467212278235;
  }
  else
    FatalError("OFR schemes have been obtained as yet for P = 1 to 6");

  if(in_mode==0) // left correction function
    dtemp_0 = eval_d_lagrange(in_r, 0, loc_zeros_gL);

  else if(in_mode==1) // right correction function
    dtemp_0 = eval_d_lagrange(in_r, in_order+1, loc_zeros_gR);

  return dtemp_0;
}

double eval_d_oesfr_1d(double in_r, int in_mode, int in_order)
{
  double dtemp_0;
  double cVal, aP, eta;

  // optimal 'c' value
  if (in_order == 1)
    cVal = 8.40e-3;
  else if (in_order == 2)
    cVal = 5.83e-4;
  else if (in_order == 3)
    cVal = 3.17e-5;
  else if (in_order == 4)
    cVal = 9.68e-7;
  else if (in_order == 5)
    cVal = 1.02e-8;
  else if (in_order == 6)
    cVal = 9.76e-11;
  else
    FatalError("ESFR schemes have been obtained as yet for P = 1 to 6");

  aP = (1.0/pow(2.0,in_order)) *factorial(2*in_order)/(factorial(in_order)*factorial(in_order));
  eta = cVal * (0.5*(2*in_order+1)) * (aP*factorial(in_order))*(aP*factorial(in_order));

  if(in_mode==0) // left correction function
    dtemp_0=0.5*pow(-1.0,in_order)*(eval_d_legendre(in_r,in_order)-(((eta*eval_d_legendre(in_r,in_order-1))+eval_d_legendre(in_r,in_order+1))/(1.0+eta)));

  else if(in_mode==1) // right correction function
    dtemp_0=0.5*(eval_d_legendre(in_r,in_order)+(((eta*eval_d_legendre(in_r,in_order-1))+eval_d_legendre(in_r,in_order+1))/(1.0+eta)));

  return dtemp_0;
}

void get_opp_3_tri(Array<double>& opp_3, Array<double>& loc_upts_tri, Array<double>& loc_1d_fpts, Array<double>& vandermonde_tri, Array<double>& inv_vandermonde_tri, int n_upts_per_tri, int order, double c_tri, int vcjh_scheme_tri)
{

  Array<double> Filt(n_upts_per_tri,n_upts_per_tri);
  Array<double> opp_3_dg(n_upts_per_tri, 3*(order+1));
  Array<double> m_temp;

  compute_filt_matrix_tri(Filt,vandermonde_tri,inv_vandermonde_tri,n_upts_per_tri,order,c_tri,vcjh_scheme_tri,loc_upts_tri);

  get_opp_3_dg(opp_3_dg, loc_upts_tri, loc_1d_fpts, n_upts_per_tri, order);
  m_temp = mult_Arrays(Filt,opp_3_dg);
  opp_3 = Array<double> (m_temp);
}

void get_opp_3_dg(Array<double>& opp_3_dg, Array<double>& loc_upts_tri, Array<double>& loc_1d_fpts, int n_upts_per_tri, int order)
{

  int i,j,k;
  Array<double> loc(2);

  for(i=0;i<3*(order+1);i++)
  {
    for(j=0;j<n_upts_per_tri;j++)
    {
      for(k=0;k<2;k++)
      {
        loc(k)=loc_upts_tri(k,j);
      }

      int edge = i/ (order+1);
      int edge_fpt = i%(order+1);

      opp_3_dg(j,i)=eval_div_dg_tri(loc,edge,edge_fpt,order,loc_1d_fpts);
    }
  }
}

// Compute a 1D modal filter matrix, given Vandermonde matrix and inverse
void compute_modal_filter_1d(Array <double>& filter_upts, Array<double>& vandermonde, Array<double>& inv_vandermonde, int N, int order)
{
  int i,j,ind=0;
  double Cp=0.1;     // filter strength coeff.
  double p=order;    // filter exponent
  double alpha, eta;
  Array <double> modal(N,N), mtemp(N,N);

  zero_Array(modal);
  zero_Array(filter_upts);

  // Exponential filter (SVV method) (similar to Meister et al 2009)

  // Full form: alpha = Cp*p*dt
  /*alpha = Cp*p;

  for(i=0;i<p+1;i++) {
    eta = i/(p+1.0);
    modal(ind,ind) = exp(-alpha*pow(eta,2*p));
    ind++;
  }*/

  // Gaussian filter in modal space (from SD3D)
  for(i=0;i<N;i++) {
    eta = i/double(N);
    modal(i,i) = exp(-pow(2.0*eta,2.0)/48.0);
  }

  // Sharp cutoff filter
  //modal(N-1,N-1) = 1.0;

  //cout<<"modal coeffs:"<<endl;
  //modal.print();

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,vandermonde.get_ptr_cpu(),N,modal.get_ptr_cpu(),N,0.0,mtemp.get_ptr_cpu(),N);

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,mtemp.get_ptr_cpu(),N,inv_vandermonde.get_ptr_cpu(),N,0.0,filter_upts.get_ptr_cpu(),N);

#else // inefficient matrix multiplication

  mtemp = mult_Arrays(inv_vandermonde,modal);
  filter_upts = mult_Arrays(mtemp,vandermonde);

#endif
}

// Compute a modal filter matrix for a triangular element, given Vandermonde matrix and inverse
void compute_modal_filter_tri(Array <double>& filter_upts, Array<double>& vandermonde, Array<double>& inv_vandermonde, int N, int order)
{
  int i,j,ind=0;
  double Cp=0.1;     // Dubiner SVV filter strength coeff.
  double p=order;    // filter exponent
  double alpha, eta;
  Array <double> modal(N,N), mtemp(N,N);

  zero_Array(modal);
  zero_Array(filter_upts);

  // Exponential filter (SVV method) (similar to Meister et al 2009)

  // Full form: alpha = Cp*(p+1)*dt/delta
  /*alpha = Cp*p;

  for(i=0;i<p+1;i++) {
    for(j=0;j<p-i+1;j++) {
      eta = (i+j)/(p+1.0);
      modal(ind,ind) = exp(-alpha*pow(eta,2*p));
      ind++;
    }
  }*/

  // Gaussian filter in modal space (from SD3D)
  for(i=0;i<N;i++) {
    eta = i/double(N);
    modal(i,i) = exp(-pow(2.0*eta,2.0)/48.0);
  }

  // Sharp modal cutoff filter
  //modal(N-1,N-1)=0.0;

  //cout<<"modal coeffs:"<<endl;
  //modal.print();

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,vandermonde.get_ptr_cpu(),N,modal.get_ptr_cpu(),N,0.0,mtemp.get_ptr_cpu(),N);

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,mtemp.get_ptr_cpu(),N,inv_vandermonde.get_ptr_cpu(),N,0.0,filter_upts.get_ptr_cpu(),N);

#else // inefficient matrix multiplication

  mtemp = mult_Arrays(inv_vandermonde,modal);
  filter_upts = mult_Arrays(mtemp,vandermonde);

#endif
}

// Compute a modal filter matrix for a tetrahedral element, given Vandermonde matrix and inverse
void compute_modal_filter_tet(Array <double>& filter_upts, Array<double>& vandermonde, Array<double>& inv_vandermonde, int N, int order)
{
  int i,j,k,ind=0;
  double Cp=0.1;     // Dubiner SVV filter strength coeff.
  double p=order;    // filter exponent
  double alpha, eta;
  Array <double> modal(N,N), mtemp(N,N);

  zero_Array(modal);
  zero_Array(filter_upts);

  // Exponential filter (SVV method) (similar to Meister et al 2009)

  // Full form: alpha = Cp*(p+1)*dt/delta
  /*alpha = Cp*p;

  for(i=0;i<p+1;i++) {
    for(j=0;j<p-i+1;j++) {
      for(k=0;k<p-i-j+1;k++) {
        eta = (i+j+k)/(p+1.0);
        modal(ind,ind) = exp(-alpha*pow(eta,2*p));
        ind++;
      }
    }
  }*/

  // Gaussian filter in modal space (from SD3D)
  for(i=0;i<N;i++) {
    eta = i/double(N);
    modal(i,i) = exp(-pow(2.0*eta,2.0)/48.0);
  }

  // Sharp modal cutoff filter
  //modal(N-1,N-1)=0.0;

  //cout<<"modal coeffs:"<<endl;
  //modal.print();

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,vandermonde.get_ptr_cpu(),N,modal.get_ptr_cpu(),N,0.0,mtemp.get_ptr_cpu(),N);

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,mtemp.get_ptr_cpu(),N,inv_vandermonde.get_ptr_cpu(),N,0.0,filter_upts.get_ptr_cpu(),N);

#else // inefficient matrix multiplication

  mtemp = mult_Arrays(inv_vandermonde,modal);
  filter_upts = mult_Arrays(mtemp,vandermonde);

#endif
}

void compute_filt_matrix_tri(Array<double>& Filt, Array<double>& vandermonde_tri, Array<double>& inv_vandermonde_tri, int n_upts_tri, int order, double c_tri, int vcjh_scheme_tri, Array<double>& loc_upts_tri)
{

  // -----------------
  // VCJH Filter
  // -----------------
  double ap;
  double c_plus;
  double c_plus_1d, c_sd_1d, c_hu_1d;

  Array<double> c_coeff(order+1);
  Array<double> mtemp_0, mtemp_1, mtemp_2;
  Array<double> K(n_upts_tri,n_upts_tri);
  Array<double> Identity(n_upts_tri,n_upts_tri);
  Array<double> Filt_dubiner(n_upts_tri,n_upts_tri);
  Array<double> Dr(n_upts_tri,n_upts_tri);
  Array<double> Ds(n_upts_tri,n_upts_tri);
  Array<double> tempr(n_upts_tri,n_upts_tri);
  Array<double> temps(n_upts_tri,n_upts_tri);
  Array<double> D_high_order_trans(n_upts_tri,n_upts_tri);
  Array<double> vandermonde_tri_trans(n_upts_tri,n_upts_tri);

  Array<Array <double> > D_high_order;
  Array<Array <double> > D_T_D;

  // 1D prep
  ap = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));

  c_sd_1d = (2*order)/((2*order+1)*(order+1)*(factorial(order)*ap)*(factorial(order)*ap));
  c_hu_1d = (2*(order+1))/((2*order+1)*order*(factorial(order)*ap)*(factorial(order)*ap));

  if(vcjh_scheme_tri>1)
  {
    if (order==2)
      c_plus_1d = 0.206;
    else if (order==3)
      c_plus_1d = 3.80e-3;
    else if (order==4)
      c_plus_1d = 4.67e-5;
    else if (order==5)
      c_plus_1d = 4.28e-7;
    else
      FatalError("C_plus scheme not implemented for this order");

    //2D
    if (order==2)
      c_plus = 3.13e-2;
    else if (order==3)
      c_plus = 4.67e-4;
    else if (order==4)
      c_plus = 6.55e-6;
    else
      FatalError("C_plus scheme not implemented for this order");
  }


  if (vcjh_scheme_tri==0)
  {
    //c_tri set by user
  }
  else if (vcjh_scheme_tri==1) // DG
  {
    c_tri = 0.;
  }
  else if (vcjh_scheme_tri==2) // SD-like
  {
    c_tri = (c_sd_1d/c_plus_1d)*c_plus;
  }
  else if (vcjh_scheme_tri==3) // HU-like
  {
    c_tri = (c_hu_1d/c_plus_1d)*c_plus;
  }
  else if (vcjh_scheme_tri==4) // Cplus scheme
  {
    c_tri = c_plus;
  }
  else
    FatalError("VCJH triangular scheme not recognized");

  //cout << "c_tri " << c_tri << endl;

  run_input.c_tri = c_tri;

  // Evaluate the derivative normalized of Dubiner basis at position in_loc
  for (int i=0;i<n_upts_tri;i++) {
    for (int j=0;j<n_upts_tri;j++) {
      tempr(i,j) = eval_dr_dubiner_basis_2d(loc_upts_tri(0,i),loc_upts_tri(1,i),j,order);
      temps(i,j) = eval_ds_dubiner_basis_2d(loc_upts_tri(0,i),loc_upts_tri(1,i),j,order);
    }
  }

  //Convert to nodal derivatives
  Dr = mult_Arrays(tempr,inv_vandermonde_tri);
  Ds = mult_Arrays(temps,inv_vandermonde_tri);

  //Create identity matrix
  zero_Array(Identity);

  for (int i=0;i<n_upts_tri;++i)
    Identity(i,i) = 1.;

  // Set Array with binomial coefficients multiplied by value of c
  for(int k=0; k<(order+1);k++) {
    c_coeff(k) = (1./n_upts_tri)*(factorial(order)/( factorial(k)*factorial(order-k) ));
    //cout << "k=" << k << "coeff= " << c_coeff(k) << endl;
  }

  // Initialize K to zero
  zero_Array(K);

  // Compute D_transpose*D
  D_high_order.setup(order+1);
  D_T_D.setup(order+1);

  for (int k=0;k<(order+1);k++)
  {
    int m = order-k;
    D_high_order(k) = Array<double> (Identity);
    for (int k2=0;k2<k;k2++)
      D_high_order(k) = mult_Arrays(D_high_order(k),Ds);
    for (int m2=0;m2<m;m2++)
      D_high_order(k) = mult_Arrays(D_high_order(k),Dr);
    //cout << "k=" << k << endl;
    //cout<<"D_high_order(k)"<<endl;
    //D_high_order(k).print();
    //cout << endl;

    D_high_order_trans = transpose_Array(D_high_order(k));
    D_T_D(k) = mult_Arrays(D_high_order_trans,D_high_order(k));

    //mtemp_2 = transpose_Array(vandermonde_tri);
    //mtemp_2 = mult_Arrays(mtemp_2,D_high_order(k));
    //mtemp_2 = mult_Arrays(mtemp_2,vandermonde_tri);
    //cout<<"V^T*D_high_order(k)*V"<<endl;
    //mtemp_2.print();
    //cout << endl;

    // Scale by c_coeff
    for (int i=0;i<n_upts_tri;i++) {
      for (int j=0;j<n_upts_tri;j++) {
        D_T_D(k)(i,j) = c_tri*c_coeff(k)*D_T_D(k)(i,j);
        K(i,j) += D_T_D(k)(i,j); //without jacobian scaling
      }
    }
  }

  //inverse mass matrix
  vandermonde_tri_trans = transpose_Array(vandermonde_tri);
  mtemp_0 = mult_Arrays(vandermonde_tri,vandermonde_tri_trans);

  //filter
  mtemp_1 = Array<double>(mtemp_0);
  mtemp_1 = mult_Arrays(mtemp_1,K);

  for (int i=0;i<n_upts_tri;i++)
    for (int j=0;j<n_upts_tri;j++)
      mtemp_1(i,j) += Identity(i,j);

  Filt = inv_Array(mtemp_1);
  Filt_dubiner = mult_Arrays(inv_vandermonde_tri,Filt);
  Filt_dubiner = mult_Arrays(Filt_dubiner,vandermonde_tri);

  //cout << "Filt" << endl;
  //Filt.print();
  //cout << endl;
  //cout << "Filt_dubiner" << endl;
  //Filt_dubiner.print();
  //cout << endl;

  /*
  // ------------------------
  // Diagonal filter
  // ------------------------
  //matrix Filt_dubiner(n_upts_tri,n_upts_tri);
  int n_upts_lower = (order+1)*order/2;

  double frac;

  if (vcjh_scheme_tri==0)
  {
    double c_1d = c_tri*2*order;
    double cp = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
    double kappa = (2*order+1)/2*(factorial(order)*cp)*(factorial(order)*cp);
    frac = 1./ (1+c_1d*kappa);
  }
  else if (vcjh_scheme_tri==1) // DG
  {
    frac = 1.0;
  }
  else if (vcjh_scheme_tri==2) // SD-like
  {
    frac = (order+1.)/(2.*order+1);
  }
  else if (vcjh_scheme_tri==3) // HU-like
  {
    frac = (order)/(2.*order+1);
  }
  else if (vcjh_scheme_tri==4) // Cplus scheme
  {
    if (order==2)
      c_tri = 4.3e-2;
    else if (order==3)
      c_tri = 6.4e-4;
    else if (order==4)
      c_tri = 5.3e-6;
    else
      FatalError("C_plus scheme not implemented for this order");

    double c_1d = c_tri*2*order;
    double cp = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
    double kappa = (2*order+1)/2*(factorial(order)*cp)*(factorial(order)*cp);
    frac = 1./ (1+c_1d*kappa);
  }
  else
    FatalError("VCJH triangular scheme not recognized");

  cout << "Filtering fraction=" << frac << endl;

  for (int j=0;j<n_upts_tri;j++) {
    for (int k=0;k<n_upts_tri;k++) {
      if (j==k) {
        if (j < n_upts_lower)
          Filt_dubiner(j,k) = 1.;
        else
          Filt_dubiner(j,k) = frac;
      }
      else {
        Filt_dubiner(j,k) = 0.;
      }
    }
  }

  Filt = mult_Arrays(vandermonde_tri,Filt_dubiner);
  Filt = mult_Arrays(Filt,inv_vandermonde_tri);

  cout << "Filt_dubiner_diag" << endl;
  Filt_dubiner.print();

  cout << "Filt_diag" << endl;
  Filt.print();
   */

}


double eval_div_dg_tri(Array<double> &in_loc , int in_edge, int in_edge_fpt, int in_order, Array<double> &in_loc_fpts_1d)
{
  int n_upts_tri = (in_order+1)*(in_order+2)/2;

  double r,s,t;
  double integral, edge_length, gdotn_at_cubpt;
  double div_vcjh_basis;

  Array<double> mtemp_0((in_order+1),(in_order+1));
  Array<double> gdotn((in_order+1),1);
  Array<double> coeff_gdotn((in_order+1),1);
  Array<double> coeff_divg(n_upts_tri,1);

  cubature_1d cub1d(10);  // TODO: CHECK STRENGTH

  if (in_edge==0)
    edge_length=2.;
  else if (in_edge==1)
    edge_length=2.*sqrt(2.);
  else if (in_edge==2)
    edge_length=2.;

  // Compute the coefficients of vjch basis in Dubiner basis
  // i.e. sigma_i = integral of (h cdot n)*L_i over the edge


  // 1. Construct a polynomial for g*n over the edge
  //    Store the coefficient of g*n in terms of jacobi basis in coeff_gdotn
  for (int i=0;i<(in_order+1);i++) {

    if (i==in_edge_fpt)
      gdotn(i,0) = 1.;
    else
      gdotn(i,0) = 0.;

    // map to [0..edge_length] interval
    t = (1.+in_loc_fpts_1d(i))/2.*edge_length;

    for (int j=0;j<(in_order+1);j++)
      mtemp_0(i,j) = eval_jacobi(t,0,0,j);
  }

  mtemp_0 = inv_Array(mtemp_0);
  coeff_gdotn = mult_Arrays(mtemp_0,gdotn);

  // 2. Perform the edge integrals to obtain coefficients sigma_i
  for (int i=0;i<n_upts_tri;i++)
  {
    integral = 0.;

    for (int j=0;j<cub1d.get_n_pts();j++)
    {
      // Get the position along the edge
      if (in_edge==0)
      {
        t = (cub1d.get_r(j)+1.)/2.*edge_length;
        r = -1 + t;
        s = -1;
      }
      else if (in_edge==1)
      {
        t = (cub1d.get_r(j)+1.)/2.*edge_length;
        r = 1 - t/edge_length*2;
        s = -1 + t/edge_length*2;
      }
      else if (in_edge==2)
      {
        t = (cub1d.get_r(j)+1.)/2.*edge_length;
        r = -1;
        s = 1 - t;
      }

      gdotn_at_cubpt = 0.;
      for (int k=0;k<(in_order+1);k++)
        gdotn_at_cubpt += coeff_gdotn(k,0)*eval_jacobi(t,0,0,k);

      integral += cub1d.get_weight(j)*eval_dubiner_basis_2d(r,s,i,in_order)*gdotn_at_cubpt;
    }
    coeff_divg(i,0) = integral*(edge_length)/2;
  }

  div_vcjh_basis = 0.;
  for (int i=0;i<n_upts_tri;i++)
    div_vcjh_basis += coeff_divg(i,0)*eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,in_order);

  return div_vcjh_basis;

}

// get intel mkl csr 4 Array format
void Array_to_mklcsr(Array<double>& in_Array, Array<double>& out_data, Array<int>& out_cols, Array<int>& out_b, Array<int>& out_e)
{
  int i,j;

  double tol=1e-24;
  int nnz=0;
  int pos=0;
  int new_row=0;

  Array<double> temp_data;
  Array<int> temp_cols, temp_b, temp_e;

  for(j=0;j<in_Array.get_dim(0);j++)
  {
    for(i=0;i<in_Array.get_dim(1);i++)
    {
      if((in_Array(j,i)*in_Array(j,i))>tol)
      {
        nnz++;
      }
    }
  }

  temp_data.setup(nnz);
  temp_cols.setup(nnz);
  temp_b.setup(in_Array.get_dim(0));
  temp_e.setup(in_Array.get_dim(0));

  pos=0;

  for(j=0;j<in_Array.get_dim(0);j++)
  {
    for(i=0;i<in_Array.get_dim(1);i++)
    {
      if((in_Array(j,i)*in_Array(j,i))>tol)
      {
        temp_data(pos)=in_Array(j,i);
        temp_cols(pos)=i+1;

        if(new_row==0)
        {
          temp_b(j)=pos+1;
          new_row=1;
        }

        pos++;
      }
    }

    new_row=0;
  }

  for(i=0;i<temp_e.get_dim(0)-1;i++)
  {
    temp_e(i)=temp_b(i+1);
  }

  temp_e(temp_e.get_dim(0)-1)=pos+1;

  out_data=temp_data;
  out_cols=temp_cols;
  out_b=temp_b;
  out_e=temp_e;
}

void Array_to_ellpack(Array<double>& in_Array, Array<double>& out_data, Array<int>& out_cols, int& nnz_per_row)
{

  double zero_tol = 1.0e-12;

  int n_rows = in_Array.get_dim(0);
  int n_cols = in_Array.get_dim(1);
  nnz_per_row = 0;
  int temp;

  for (int i=0;i<n_rows;i++)
  {
    temp = 0;
    for (int j=0;j<n_cols;j++)
    {
      //cout << "in_Array=" << in_Array(i,j) << endl;
      if (abs(in_Array(i,j)) >= zero_tol)
        temp++;
    }
    //cout << "temp= " << temp << endl;
    if (temp > nnz_per_row)
      nnz_per_row=temp;
  }
  //cout << "nnz_per_row=" << nnz_per_row << endl;

  out_data.setup(nnz_per_row*n_rows);
  out_cols.setup(nnz_per_row*n_rows);

  for (int i=0;i<nnz_per_row*n_rows;i++) {
    out_data(i) = 0.;
    out_cols(i) = 0.;
  }

  //cout << "nnz_per_row*n_rows=" << nnz_per_row*n_rows << endl;
  int index;
  for (int i=0;i<n_rows;i++)
  {
    int count=0;
    for (int j=0;j<n_cols;j++)
    {
      if ( abs(in_Array(i,j)) >= zero_tol)
      {
        index=i+count*n_rows;
        out_data(index) = in_Array(i,j);
        out_cols(index) = j;
        count++;
      }
    }
  }


}


Array<double> rs_to_ab(double in_r, double in_s)
    {
  Array<double> ab(2);

  if(in_s==1.0) // to avoid singularity
  {
    ab(0)=-1.0;
  }
  else
  {
    ab(0)=(2.0*((1.0+in_r)/(1.0-in_s)))-1.0;
  }

  ab(1)=in_s;

  return ab;
    }

#ifdef _GPU
/*
void Array_to_cusparse_csr(Array<double>& in_Array, cusparseHybMat_t &hyb_Array, cusparseHandle_t& handle)
{
  int n_rows = in_Array.get_dim(0);
  int n_cols = in_Array.get_dim(1);

  cout << "Converting to hybrid format" << endl;

  Array<int> nnz_per_row(n_rows);
  int nnzTotalDevHostPtr;

  //cusparseCreateHybMat(&hyb_Array);
  cusparseMatDescr_t mat_description;
  cusparseStatus_t status;

  status = cusparseCreateMatDescr(&mat_description);
  if (status != CUSPARSE_STATUS_SUCCESS){
    cout << "error create Mat Desc" << endl;
    exit(1);
  }

  cusparseSetMatType(mat_description,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(mat_description,CUSPARSE_INDEX_BASE_ZERO);

  cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, n_rows, n_cols,mat_description,in_Array.get_ptr_gpu(),n_rows,nnz_per_row.get_ptr_cpu(),&nnzTotalDevHostPtr);

  const double* gpu_ptr = in_Array.get_ptr_gpu();

  cusparseDdense2csr(handle,n_rows,n_cols,mat_description,gpu_ptr,n_rows,nnz_per_row.get_ptr_cpu(),hyb_Array,NULL,CUSPARSE_HYB_PARTITION_AUTO);
}
 */
#endif

Array<double> rst_to_abc(double in_r, double in_s, double in_t) // CHECK (PASSING)
    {
  Array<double> abc(3);

  if (in_s + in_t == 0.0)
  {
    abc(0) = -1.0;
  }
  else
  {
    abc(0) = -2.0*(1.0+in_r)/(in_s+in_t) - 1.0;

  }

  if (in_t == 1.0)
  {
    abc(1) = -1.0;
  }
  else
  {
    abc(1) = 2.0*(1.0+in_s)/(1.0-in_t) - 1.0;

  }
  abc(2) = in_t;

  return abc;
    }

// helper method to evaluate the gamma function for positive integers
double eval_gamma(int in_n)
{
  int i;
  double gamma_val;

  if(in_n==1)
    gamma_val=1;
  else
  {
    gamma_val=in_n-1;

    for(i=0;i<in_n-2;i++)
    {
      gamma_val=gamma_val*(in_n-2-i);
    }
  }

  return gamma_val;
}

// helper method to evaluate a normalized jacobi polynomial
double eval_jacobi(double in_r, int in_alpha, int in_beta, int in_mode)
{
  double jacobi;

  if(in_mode==0)
  {
    double dtemp_0, dtemp_1, dtemp_2;

    dtemp_0=pow(2.0,(-in_alpha-in_beta-1));
    dtemp_1=eval_gamma(in_alpha+in_beta+2);
    dtemp_2=eval_gamma(in_alpha+1)*eval_gamma(in_beta+1);

    jacobi=sqrt(dtemp_0*(dtemp_1/dtemp_2));
  }
  else if(in_mode==1)
  {
    double dtemp_0, dtemp_1, dtemp_2, dtemp_3, dtemp_4, dtemp_5;

    dtemp_0=pow(2.0,(-in_alpha-in_beta-1));
    dtemp_1=eval_gamma(in_alpha+in_beta+2);
    dtemp_2=eval_gamma(in_alpha+1)*eval_gamma(in_beta+1);
    dtemp_3=in_alpha+in_beta+3;
    dtemp_4=(in_alpha+1)*(in_beta+1);
    dtemp_5=(in_r*(in_alpha+in_beta+2)+(in_alpha-in_beta));

    jacobi=0.5*sqrt(dtemp_0*(dtemp_1/dtemp_2))*sqrt(dtemp_3/dtemp_4)*dtemp_5;
  }
  else
  {
    double dtemp_0, dtemp_1, dtemp_2, dtemp_3, dtemp_4, dtemp_5, dtemp_6, dtemp_7, dtemp_8, dtemp_9, dtemp_10, dtemp_11, dtemp_12, dtemp_13, dtemp_14;

    dtemp_0=in_mode*(in_mode+in_alpha+in_beta)*(in_mode+in_alpha)*(in_mode+in_beta);
    dtemp_1=((2*in_mode)+in_alpha+in_beta-1)*((2*in_mode)+in_alpha+in_beta+1);
    dtemp_3=(2*in_mode)+in_alpha+in_beta;

    dtemp_4=(in_mode-1)*((in_mode-1)+in_alpha+in_beta)*((in_mode-1)+in_alpha)*((in_mode-1)+in_beta);
    dtemp_5=((2*(in_mode-1))+in_alpha+in_beta-1)*((2*(in_mode-1))+in_alpha+in_beta+1);
    dtemp_6=(2*(in_mode-1))+in_alpha+in_beta;

    dtemp_7=-((in_alpha*in_alpha)-(in_beta*in_beta));
    dtemp_8=((2*(in_mode-1))+in_alpha+in_beta)*((2*(in_mode-1))+in_alpha+in_beta+2);

    dtemp_9=(2.0/dtemp_3)*sqrt(dtemp_0/dtemp_1);
    dtemp_10=(2.0/dtemp_6)*sqrt(dtemp_4/dtemp_5);
    dtemp_11=dtemp_7/dtemp_8;

    dtemp_12=in_r*eval_jacobi(in_r,in_alpha,in_beta,in_mode-1);
    dtemp_13=dtemp_10*eval_jacobi(in_r,in_alpha,in_beta,in_mode-2);
    dtemp_14=dtemp_11*eval_jacobi(in_r,in_alpha,in_beta,in_mode-1);

    jacobi=(1.0/dtemp_9)*(dtemp_12-dtemp_13-dtemp_14);
  }

  return jacobi;
}

// helper method to evaluate the gradient of a normalized jacobi polynomial
double eval_grad_jacobi(double in_r, int in_alpha, int in_beta, int in_mode)
{
  double grad_jacobi;

  if(in_mode==0)
  {
    grad_jacobi=0.0;
  }
  else
  {
    grad_jacobi=sqrt(1.0*in_mode*(in_mode+in_alpha+in_beta+1))*eval_jacobi(in_r,in_alpha+1,in_beta+1,in_mode-1);
  }

  return grad_jacobi;
}

double eval_dubiner_basis_2d(double in_r, double in_s, int in_mode, int in_basis_order)
{
  double dubiner_basis_2d;

  int n_dof=((in_basis_order+1)*(in_basis_order+2))/2;

  if(in_mode<n_dof)
  {
    int i,j,k;
    int mode;
    double jacobi_0, jacobi_1;
    Array<double> ab;

    ab=rs_to_ab(in_r,in_s);

    mode = 0;
    for (k=0;k<in_basis_order+1;k++)
    {
      for (j=0;j<k+1;j++)
      {
        i = k-j;
        if(mode==in_mode) // found the correct mode
        {
          jacobi_0=eval_jacobi(ab(0),0,0,i);
          jacobi_1=eval_jacobi(ab(1),(2*i)+1,0,j);
          dubiner_basis_2d=sqrt(2.0)*jacobi_0*jacobi_1*pow(1.0-ab(1),i);
        }
        mode++;
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating Dubiner basis ...." << endl;
  }

  return dubiner_basis_2d;
}

double eval_dr_dubiner_basis_2d(double in_r, double in_s, int in_mode, int in_basis_order)
{
  double dr_dubiner_basis_2d;

  int n_dof=((in_basis_order+1)*(in_basis_order+2))/2;

  if(in_mode<n_dof)
  {
    int i,j,k;
    int mode;
    double jacobi_0, jacobi_1;
    Array<double> ab;

    ab=rs_to_ab(in_r,in_s);

    mode = 0;
    for (k=0;k<in_basis_order+1;k++)
    {
      for (j=0;j<k+1;j++)
      {
        i = k-j;
        if(mode==in_mode) // found the correct mode
        {

          jacobi_0=eval_grad_jacobi(ab(0),0,0,i);
          jacobi_1=eval_jacobi(ab(1),(2*i)+1,0,j);

          if(i==0) // to avoid singularity
          {
            //dr_dubiner_basis_2d=sqrt(2.0)*jacobi_0*jacobi_1;
            dr_dubiner_basis_2d=0.;
          }
          else
          {
            dr_dubiner_basis_2d=2.0*sqrt(2.0)*jacobi_0*jacobi_1*pow(1.0-ab(1),i-1);
          }
        }
        mode++;
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating basis ...." << endl;
  }

  return dr_dubiner_basis_2d;
}

// helper method to evaluate d/ds of scalar dubiner basis

double eval_ds_dubiner_basis_2d(double in_r, double in_s, int in_mode, int in_basis_order)
{
  double ds_dubiner_basis_2d;

  int n_dof=((in_basis_order+1)*(in_basis_order+2))/2;

  if(in_mode<n_dof)
  {
    int i,j,k;
    int mode;
    double jacobi_0, jacobi_1, jacobi_2, jacobi_3, jacobi_4;
    Array<double> ab;

    ab=rs_to_ab(in_r,in_s);


    mode = 0;
    for (k=0;k<in_basis_order+1;k++)
    {
      for (j=0;j<k+1;j++)
      {
        i = k-j;
        if(mode==in_mode) // find the correct mode
            {
          jacobi_0=eval_grad_jacobi(ab(0),0,0,i);
          jacobi_1=eval_jacobi(ab(1),(2*i)+1,0,j);

          jacobi_2=eval_jacobi(ab(0),0,0,i);
          jacobi_3=eval_grad_jacobi(ab(1),(2*i)+1,0,j)*pow(1.0-ab(1),i);
          jacobi_4=eval_jacobi(ab(1),(2*i)+1,0,j)*i*pow(1.0-ab(1),i-1);

          if(i==0) // to avoid singularity
          {
            ds_dubiner_basis_2d=sqrt(2.0)*(jacobi_2*jacobi_3);
          }
          else
          {
            ds_dubiner_basis_2d=sqrt(2.0)*((jacobi_0*jacobi_1*pow(1.0-ab(1),i-1)*(1.0+ab(0)))+(jacobi_2*(jacobi_3-jacobi_4)));
          }
            }
        mode++;
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating basis ...." << endl;
  }

  return ds_dubiner_basis_2d;
}


double eval_dubiner_basis_3d(double in_r, double in_s, double in_t, int in_mode, int in_basis_order)
{
  double sdubiner_basis_3d;

  int n_dof=((in_basis_order+1)*(in_basis_order+2)*(in_basis_order+3))/6;

  if(in_mode<n_dof)
  {
    int i,j,k,m,n;
    int mode;
    double jacobi_0, jacobi_1,jacobi_2;
    Array<double> abc;

    abc=rst_to_abc(in_r,in_s,in_t);

    mode = 0;

    for(m=0;m<in_basis_order+1;m++)
    {
      for(n=0;n<m+1;n++)
      {
        for(k=0;k<n+1;k++)
        {
          j= n-k;
          i = m-j-k;
          if(mode==in_mode) // found the correct mode
              {
            jacobi_0=eval_jacobi(abc(0),0,0,i);
            jacobi_1=eval_jacobi(abc(1),(2*i)+1,0,j);
            jacobi_2=eval_jacobi(abc(2),(2*i)+(2*j)+2,0,k);
            sdubiner_basis_3d=2.0*sqrt(2.0)*jacobi_0*jacobi_1*jacobi_2*pow(1.0-abc(1),i)*pow(1-abc(2),i+j);
              }
          mode++;
        }
      }
    }

  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating basis ...." << endl;
  }

  return sdubiner_basis_3d;
}

// helper method to evaluate gradient of scalar dubiner basis

double eval_grad_dubiner_basis_3d(double in_r, double in_s, double in_t, int in_mode, int in_basis_order, int component)
{
  double dr_sdubiner,ds_sdubiner,dt_sdubiner;
  double temp,fa,gb,hc,dfa,dgb,dhc;

  int n_dof=((in_basis_order+1)*(in_basis_order+2)*(in_basis_order+3))/6;

  if(in_mode<n_dof)
  {
    int i,j,k,m,n;
    int mode;
    Array<double> abc;

    abc=rst_to_abc(in_r,in_s,in_t);

    mode = 0;

    for(m=0;m<in_basis_order+1;m++)
    {
      for(n=0;n<m+1;n++)
      {
        for(k=0;k<n+1;k++)
        {
          j= n-k;
          i = m-j-k;
          if(mode==in_mode) // found the correct mode
          {


            fa = eval_jacobi(abc(0),0,0,i);
            gb = eval_jacobi(abc(1),2*i+1,0,j);
            hc = eval_jacobi(abc(2),2*(i+j)+2,0,k);

            dfa = eval_grad_jacobi(abc(0),0,0,i);
            dgb = eval_grad_jacobi(abc(1),2*i+1,0,j);
            dhc = eval_grad_jacobi(abc(2),2*(i+j)+2,0,k);

            dr_sdubiner = dfa*gb*hc;

            if (i>0)
            {
              dr_sdubiner = dr_sdubiner*pow( 0.5*(1.-abc(1)), i-1);
            }
            if (i+j>0)
            {
              dr_sdubiner = dr_sdubiner*pow( 0.5*(1.-abc(2)), i+j-1);
            }

            if (component == 0)
            {
              return (dr_sdubiner*pow(2,2*i+j+1.5));
            }

            // ------------------

            ds_sdubiner = (0.5*(1.+abc(0)))*dr_sdubiner;

            temp = dgb*pow(0.5*(1.-abc(1)),i);

            if (i>0)
            {
              temp = temp+(-0.5*i)*gb*pow((0.5*(1.-abc(1))),i-1);
            }
            if (i+j>0)
            {
              temp = temp*pow(0.5*(1-abc(2)),i+j-1);
            }
            temp = fa*temp*hc;

            ds_sdubiner = ds_sdubiner + temp;

            if (component == 1)
            {
              return (ds_sdubiner*pow(2,2*i+j+1.5));

            }

            dt_sdubiner = 0.5*(1.+abc(0))*dr_sdubiner + 0.5*(1.+abc(1))*temp;
            temp = dhc*pow(0.5*(1.-abc(2)),i+j);

            if (i+j > 0)
            {
              temp = temp - 0.5*(i+j)*(hc*pow(0.5*(1-abc(2)),i+j-1));
            }

            temp = fa*(gb*temp);

            temp = temp*pow(0.5*(1.-abc(1)),i);

            dt_sdubiner = dt_sdubiner + temp;

            if (component==2)
            {
              return (dt_sdubiner*pow(2,2*i+j+1.5));
            }
          }
          mode++;
        }
      }
    }
  }
  FatalError("ERROR: Invalid mode when evaluating basis ....");
  return 0.0;
}

bool is_perfect_square(int in_a)
{
  int number = round(sqrt(1.0*in_a));
  return (in_a == number*number);
}

bool is_perfect_cube(int in_a)
{
  int number = round(pow(1.0*in_a,1./3.));
  return (in_a == number*number*number);
}

double compute_eta(int vcjh_scheme, int order)
{
  double eta;
  // Check for P=0 compatibility
  if(order == 0 && vcjh_scheme != 1)
    FatalError("ERROR: P=0 only compatible with DG. Set VCJH scheme type to 1!")

    if(vcjh_scheme==1) // dg
    {
      eta=0.0;
    }
    else if(vcjh_scheme==2) // sd
    {
      eta=(1.0*(order))/(1.0*(order+1));
    }
    else if(vcjh_scheme==3) // hu
    {
      eta=(1.0*(order+1))/(1.0*order);
    }
    else if (vcjh_scheme==4)
    {
      double c_1d;
      if (order==2)
        c_1d = 0.206;
      else if (order==3)
        c_1d = 3.80e-3;
      else if (order==4)
        c_1d = 4.67e-5;
      else if (order==5)
        c_1d = 4.28e-7;
      else
        FatalError("C_plus scheme not implemented for this order");

      double ap = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
      eta = c_1d*(2*order+1)/2*(factorial(order)*ap)*(factorial(order)*ap);

    }
    else
    {
      cout << "ERROR: Invalid VCJH scheme ... " << endl;
    }

  return eta;
}

int compare_ints(const void * a, const void *b)
{
  return ( *(int*)a-*(int*)b );
}


// Method that searches a value in a sorted Array without repeated entries and returns position in Array
int index_locate_int(int value, int* Array, int size)
{
  int ju,jm,jl;
  int ascnd;

  jl = 0;
  ju = size-1;

  if (Array[ju] <= Array[0] && ju!=0)
  {
    cout << "ERROR, Array not sorted, exiting" << endl;
    cout << "size= " << size << endl;
    cout << "Array[0] = " << Array[0] << endl;
    cout << "Array[size-1] = " << Array[ju] << endl;
    exit(1);
  }

  while(ju-jl > 1)
  {
    jm = (ju+jl) >> 1;
    if (value>=Array[jm])
    {
      jl=jm;
    }
    else
    {
      ju=jm;
    }
  }

  if (value == Array[0])
  {
    return 0;
  }
  else if (value == Array[size-1])
  {
    return size-1;
  }
  else if (value == Array[jl])
  {
    return jl;
  }
  else
  {
    return -1;
  }
}

void eval_isentropic_vortex(Array<double>& pos, double time, double& rho, double& vx, double& vy, double& vz, double& p, int n_dims)
{
  Array<double> relative_pos(n_dims);

  double gamma=1.4;
  /*
                double ev_x_pos_ic;
                double ev_y_pos_ic;
                double ev_mach_ic;
                double ev_eps_ic;
                double ev_theta_ic;
                double ev_rho_inf_ic;
                double ev_magv_inf_ic;
                double ev_rad_ic;
                double ev_p_inf_ic;

                double f;
   */

  /*
                ev_x_pos_ic=0.0;
                ev_y_pos_ic=0.0;
                ev_mach_ic=0.4;
                ev_eps_ic=5.0;
                ev_theta_ic=pi/2.0;
                ev_rho_inf_ic=1.0;
                ev_magv_inf_ic=1.0;
                ev_rad_ic=1;
                ev_p_inf_ic=1.0/(gamma*ev_mach_ic*ev_mach_ic);
                ev_p_inf_ic=1.0/(gamma*ev_mach_ic*ev_mach_ic);

        relative_pos(0) = pos(0) - ev_magv_inf_ic*cos(ev_theta_ic)*time;
        relative_pos(1) = pos(1) - ev_magv_inf_ic*sin(ev_theta_ic)*time;

                f=(1.0-((relative_pos(0)-ev_x_pos_ic)*(relative_pos(0)-ev_x_pos_ic))-((relative_pos(1)-ev_y_pos_ic)*(relative_pos(1)-ev_y_pos_ic)))/(ev_rad_ic*ev_rad_ic);

                rho=ev_rho_inf_ic*pow((1.0-(((ev_eps_ic*ev_eps_ic*(gamma-1.0)*ev_mach_ic*ev_mach_ic)/(8.0*pi*pi))*exp(f))),1.0/(gamma-1.0));
                vx=ev_magv_inf_ic*(cos(ev_theta_ic)-(((ev_eps_ic*(relative_pos(1)-ev_y_pos_ic))/(2.0*pi*ev_rad_ic))*exp(f/2.0)));
                vy=ev_magv_inf_ic*(sin(ev_theta_ic)+(((ev_eps_ic*(relative_pos(0)-ev_x_pos_ic))/(2.0*pi*ev_rad_ic))*exp(f/2.0)));
                vz=0.0;
                p=ev_p_inf_ic*pow((1.0-(((ev_eps_ic*ev_eps_ic*(gamma-1.0)*ev_mach_ic*ev_mach_ic)/(8.0*pi*pi))*exp(f))),gamma/(gamma-1.0));
   */

  double ev_eps_ic= 5.0;


  double x = pos(0) - time;
  double y = pos(1) - time;

  double f=1.0-(x*x+y*y);

  rho=pow(1.0-ev_eps_ic*ev_eps_ic*(gamma-1.0)/(8.0*gamma*pi*pi)*exp(f),1.0/(gamma-1.0));
  vx=1.-ev_eps_ic*y/(2.0*pi)*exp(f/2.0);
  vy=1.+ev_eps_ic*x/(2.0*pi)*exp(f/2.0);
  p = pow(rho,gamma);

}


void eval_sine_wave_single(Array<double>& pos, Array<double>& wave_speed, double diff_coeff, double time, double& rho, Array<double>& grad_rho, int n_dims)
{

  Array<double> relative_pos(n_dims);
  relative_pos(0) = pos(0) - wave_speed(0)*time;
  relative_pos(1) = pos(1) - wave_speed(1)*time;
  if (n_dims==3)
  {
    relative_pos(2) = pos(2) - wave_speed(2)*time;
  }

  double angle;
  if (n_dims==2)
    angle = relative_pos(0)+relative_pos(1);
  else if (n_dims==3)
    angle = relative_pos(0)+relative_pos(1)+relative_pos(2);

  rho = exp(-((double) n_dims)*diff_coeff*pi*pi*time)*sin(pi*angle);

  grad_rho(0) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*cos(pi*angle);
  grad_rho(1) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*cos(pi*angle);

  if(n_dims==3)
    grad_rho(2) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*cos(pi*angle);
}


void eval_sine_wave_group(Array<double>& pos, Array<double>& wave_speed, double diff_coeff, double time, double& rho, Array<double>& grad_rho, int n_dims)
{

  Array<double> relative_pos(n_dims);
  relative_pos(0) = pos(0) - wave_speed(0)*time;
  relative_pos(1) = pos(1) - wave_speed(1)*time;
  if (n_dims==3)
    relative_pos(2) = pos(2) - wave_speed(2)*time;

  if (n_dims==2)
    rho = exp(-((double) n_dims)*diff_coeff*pi*pi*time)*sin(pi*relative_pos(0))*sin(pi*relative_pos(1));
  if (n_dims==3)
    rho = exp(-((double) n_dims)*diff_coeff*pi*pi*time)*sin(pi*relative_pos(0))*sin(pi*relative_pos(1))*sin(pi*relative_pos(2));

  if(n_dims==2)
  {
    grad_rho(0) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*cos(pi*relative_pos(0))*sin(pi*relative_pos(1));
    grad_rho(1) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*sin(pi*relative_pos(0))*cos(pi*relative_pos(1));
  }
  if(n_dims==3)
  {
    grad_rho(0) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*cos(pi*relative_pos(0))*sin(pi*relative_pos(1))*sin(pi*relative_pos(2));
    grad_rho(1) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*sin(pi*relative_pos(0))*cos(pi*relative_pos(1))*sin(pi*relative_pos(2));
    grad_rho(2) = pi*exp(-((double) n_dims)*diff_coeff*pi*pi*time)*sin(pi*relative_pos(0))*sin(pi*relative_pos(1))*cos(pi*relative_pos(2));
  }
}


void eval_sphere_wave(Array<double>& pos, Array<double>& wave_speed, double time, double& rho, int n_dims)
{

  Array<double> relative_pos(n_dims);
  relative_pos(0) = pos(0) - wave_speed(0)*time;
  relative_pos(1) = pos(1) - wave_speed(1)*time;
  relative_pos(2) = pos(2) - wave_speed(2)*time;

  rho = exp(-0.5*(  relative_pos(0)*relative_pos(0) +
      relative_pos(1)*relative_pos(1) +
      relative_pos(2)*relative_pos(2) ));
}

int factorial(int in_n)
{

  int result;

  if (in_n ==0)
  {
    return 1;
  }
  else
  {
    result = 1;
    for(int i=1;i<=in_n;i++)
    {
      result *= i;
    }
    return result;
  }
}

void eval_couette_flow(Array<double>& pos, double in_gamma, double in_R_ref, double in_u_wall, double in_T_wall, double in_p_bound, double in_prandtl, double time, double& ene, Array<double>& grad_ene, int n_dims)
{
  double x,y,z;

  double cp;
  double h_channel, T_fact;
  double rho, mom_x, mom_y, mom_z, vx, vy, vz, Ts, ps;
  double gam, R_ref, u_wall, T_wall, p_bound, prandtl, ka, kb;
  double rho_dx, rho_dy, rho_dz;
  double mom_x_dx, mom_x_dy, mom_x_dz;
  double mom_y_dx, mom_y_dy, mom_y_dz;
  double mom_z_dx, mom_z_dy, mom_z_dz;
  double ene_dx, ene_dy, ene_dz;

  gam = in_gamma;
  R_ref = in_R_ref;
  u_wall = in_u_wall;
  T_wall = in_T_wall;
  p_bound = in_p_bound;
  prandtl = in_prandtl;

  T_fact = 1.0;
  h_channel = 1.0;

  x = pos(0);
  y = pos(1);
  if(n_dims==3)
    z = pos(2);

  cp = (gam*R_ref)/(gam-1.0);

  vx = u_wall*(y/h_channel);
  vy = 0.0;
  if(n_dims==3)
    vz = 0.0;

  ka = (T_fact*T_wall - T_wall);
  kb = 0.5*(prandtl/cp)*(u_wall*u_wall);

  ps = p_bound;
  Ts = T_wall + (y/h_channel)*ka + kb*(y/h_channel)*(1.0 - (y/h_channel));

  rho = ps/(R_ref*Ts);

  mom_x 	= rho*vx;
  mom_y 	= rho*vy;
  if(n_dims==3)
    mom_z = rho*vz;

  if(n_dims==2)
    ene 	= (ps/(gam-1.0))+0.5*rho*((vx*vx)+(vy*vy));

  if(n_dims==3)
    ene 	= (ps/(gam-1.0))+0.5*rho*((vx*vx)+(vy*vy)+(vz*vz));

  rho_dx = 0.0;
  rho_dy = -(ps/R_ref)*((ka/h_channel) - ((kb*y)/(h_channel*h_channel)) + (kb/h_channel)*(1.0 - (y/h_channel)))/
      pow( T_wall + ka*(y/h_channel) + kb*(y/h_channel)*(1.0 - (y/h_channel)) ,2.0);
  if(n_dims==3)
    rho_dz = 0.0;

  mom_x_dx = 0.0;
  mom_x_dy = rho_dy*vx + rho*(u_wall/h_channel);
  if(n_dims==3)
    mom_x_dz = 0.0;

  mom_y_dx = 0.0;
  mom_y_dy = 0.0;
  if(n_dims==3)
    mom_y_dz = 0.0;

  ene_dx = 0.0;
  ene_dy = 0.5*rho_dy*(vx*vx) + mom_x*(u_wall/h_channel);
  if(n_dims==3)
    ene_dz = 0.0;

  grad_ene(0) = ene_dx;
  grad_ene(1) = ene_dy;
  if(n_dims==3)
    grad_ene(2) = ene_dz;
}

// Set initial momentum as up to 4th order polynomial in each direction
// TODO: allow mixed terms e.g. xy, yz^2
void eval_poly_ic(Array<double>& pos, double rho, Array<double>& ics, int n_dims)
{
  // HACK: do not use profile outside the vertical bounds of the inlet in periodic hill case
  if(pos(1)<1.0)
  {
    ics(1) = 0.0;
    ics(2) = 0.0;
    ics(3) = 0.0;
  }
  else {
    // Take N user-specified coefficients {a,b,c,...,n} to construct a polynomial of the form
    // u = a + bx + cx^2 + ... + nx^N (1D)
    // In 2D and 3D, add extra coeffs for mixed terms xy, xyz, x^2y etc.
    Array <double> c(13);

    for(int i=0;i<13;++i)
      c(i) = run_input.x_coeffs(i);
    ics(1) = c(0)+c(1)*pos(0)+c(2)*pow(pos(0),2)+c(3)*pow(pos(0),3)+c(4)*pow(pos(0),4)+c(5)*pos(1)+c(6)*pow(pos(1),2)+c(7)*pow(pos(1),3)+c(8)*pow(pos(1),4);

    if(n_dims==3)
      ics(1) += c(9)*pos(2)+c(10)*pow(pos(2),2)+c(11)*pow(pos(2),3)+c(12)*pow(pos(2),4);

    for(int i=0;i<13;++i)
      c(i) = run_input.y_coeffs(i);
    ics(2) = c(0)+c(1)*pos(0)+c(2)*pow(pos(0),2)+c(3)*pow(pos(0),3)+c(4)*pow(pos(0),4)+c(5)*pos(1)+c(6)*pow(pos(1),2)+c(7)*pow(pos(1),3)+c(8)*pow(pos(1),4);

    if(n_dims==3)
    {
      ics(2) += c(9)*pos(2)+c(10)*pow(pos(2),2)+c(11)*pow(pos(2),3)+c(12)*pow(pos(2),4);

      for(int i=0;i<13;++i)
        c(i) = run_input.z_coeffs(i);
      ics(3) = c(0)+c(1)*pos(0)+c(2)*pow(pos(0),2)+c(3)*pow(pos(0),3)+c(4)*pow(pos(0),4)+c(5)*pos(1)+c(6)*pow(pos(1),2)+c(7)*pow(pos(1),3)+c(8)*pow(pos(1),4)+c(9)*pos(2)+c(10)*pow(pos(2),2)+c(11)*pow(pos(2),3)+c(12)*pow(pos(2),4);
    }
  }
}

/*! Functions used in evaluation of shape functions and its 1st and 2nd derivatives
BEGIN:*/
Array<double> convol(Array<double> & polynomial1, Array<double> & polynomial2)
    {
  // Accepts only row vectors that represent polynomials
  // Get lengths
  int sizep1 = polynomial1.get_dim(1);
  int sizep2 = polynomial2.get_dim(1);

  // Allocate memory for result of multiplication of polynomials
  Array<double> polynomial3;
  polynomial3.setup(1,sizep1 + sizep2 - 1);
  polynomial3.initialize_to_zero();

  for (int i = 0; i < sizep1; i++)
  {
    for (int j = 0; j < sizep2; j++)
    {
      polynomial3(i + j) += polynomial2(j)*polynomial1(i);
    }
  }

  return polynomial3;
    }

Array<double> LagrangeP(int order, int node, Array<double> & subs)
    {
  //Function that finds the coefficients of the Lagrange polynomial
  /*
    % order: order of the polynomial
    % node: index of xi corresponding to point where polynomial equals 1
    % subs: polynomial that is substituted for variable in lagrange polynomial
   */
  double range[] = {-1.0,1.0}; // range over which nodes are located

  // xi: Array with location of points where function is zero

  Array<double> xi = createEquispacedArray(range[0], range[1], order+1);

  int constInSubs = subs.get_dim(1); // location of the constant term in polynomial subs
  // this is just the last term of the polynomial
  // Constructing the polynomial

  Array<double> num;
  num(0) = 1; // initalize this Array to 1
  // Do the same for the denominator

  Array<double> den;
  den(0) = 1; // initalize this Array to 1

  // declare temporary variables
  Array<double> term;
  Array<double> tempConstant;

  for (int i = 0; i < order+1 ; i++)
  {
    if (i != node-1)
    {

      term = subs;
      term(constInSubs-1) -= xi(i); // adds constant from subs to that in lagrange polynomial calculation
      num = convol(num,term); // update the numerator; multiply polynomials

      tempConstant(0) = xi(node-1) - xi(i);
      den = convol(den,tempConstant); // update the denominator; this is a simple multiplication
    }
  }
  // Get reciprocal of denominator
  den(0) = 1/(den(0));

  return convol(num,den); // return the numerator times the reciprocal of the denominator

    }


Array<double> shapePoly4Tri(int in_index, int nNodesSide)
    {
  /*
    % returns the polynomial function T_I(r) in the polynomial format
    % Array values are coefficients of monomials of increasing order
    % in_index : index of node in triangle
    % nNodesSide: number of nodes in side
    For specifics, refer to Hughes, pp 166
   */

  Array<double> T_I;// special lagrange polynomial corresponding to a node in the triangle
  // this is the result

  if(in_index == 1)
  {
    T_I(0) = 1; // return constant 1
    return T_I;
  }
  else
  {
    int order = in_index - 1; // as described in Hughes pp 167

    double range[] = {-1.0,1.0}; // range over which nodes are located

    // xi: Array with location of points where function is zero
    Array<double> xi = createEquispacedArray(range[0], range[1], nNodesSide);

    double r_I = xi(in_index-1); // get location of node in_index in the range

    // Create polynomial to substitute to create polynomial related to triangles
    Array<double> subs(1,2);
    // Specify coefficient of r
    subs(0) = 2./(r_I + 1.);
    subs(1) = (1. - r_I)/(1. + r_I);

    T_I = LagrangeP(order,in_index,subs); //note that order = in_index - 1

    return T_I;

  }
    }


Array<double> createEquispacedArray(double a, double b, int nPoints)
    {
  Array<double> xi(1,nPoints);
  for ( int i = 0; i < nPoints ; i++)
  {
    xi(i) = (i)/(double(nPoints)-1)*(b - a) + a;
  }
  return xi;
    }


Array<double> addPoly(Array<double> & p1, Array<double> & p2)
    {
  // Returns a 3D Array; each layer represents a multiplication of polynomials
  Array<double> p3; // return polynomial

  // If any of the two is zero, return the other
  if (iszero(p1))   return p2;
  if (iszero(p2))  return p1;

  // Get dimensions from each polynomial
  int lengthp1 = p1.get_dim(1);
  int heightp1 = p1.get_dim(0);
  int depthp1 = p1.get_dim(2);
  //cout<<"length p1: "<<lengthp1<<"   height p1: "<<heightp1;
  //cout<<"   depth p1: "<<depthp1<<endl;
  int lengthp2 = p2.get_dim(1);
  int heightp2 = p2.get_dim(0);
  int depthp2 = p2.get_dim(2);
  //cout<<"length p2: "<<lengthp2<<"   height p2: "<<heightp2;
  //cout<<"   depth p2: "<<depthp2<<endl;

  int lengthp3 = max(lengthp1,lengthp2);
  int heightp3 = max(heightp1,heightp2);
  int depthp3 = depthp1 + depthp2;

  p3.setup(heightp3,lengthp3,depthp3);
  p3.initialize_to_zero();

  // Copy values from p1

  for(int k = 0; k < depthp1; k++)
  {
    for (int i = 0; i < heightp1; i++)
    {
      for (int j = 0; j < lengthp1; j++)
      {
        p3(i, lengthp3 - j - 1, k) = p1(i, lengthp1 - j - 1, k);
      }
    }
  }

  // Copy values from p2
  for(int k = 0; k < depthp2; k++)
  {
    for (int i = 0; i < heightp2; i++)
    {
      for (int j = 0; j < lengthp2; j++)
      {
        p3(i, lengthp3 - j - 1, depthp1 + k) = p2(i,lengthp2 - j - 1, k);
      }
    }
  }
  //cout<<endl<<"Depth of p3: "<<p3.get_dim(2)<<endl;
  //p2.print();
  return p3;

    }


template <typename T>
Array<T> multPoly(Array<T> & p1, Array<T> & p2)
{
  Array<T> p3; // return polynomial
  if (iszero(p1) || iszero(p2))
  {
    p3(0) = 0;
    return p3;
  }

  else
  {
    int lengthp1 = p1.get_dim(1);
    int heightp1 = p1.get_dim(0);
    //cout<<"length p1: "<<lengthp1<<"   height p1: "<<heightp1<<endl;
    int lengthp2 = p2.get_dim(1);
    int heightp2 = p2.get_dim(0);
    //cout<<"length p2: "<<lengthp2<<"   height p2: "<<heightp2<<endl;

    // Calculate dimensions of p3: needs to accomodate sizes of p1 and p2
    int lengthp3 = max(lengthp1,lengthp2); // return largest of the two values
    int heightp3 = heightp1 + heightp2; // add heights to accomodate

    p3.setup(heightp3,lengthp3);
    p3.initialize_to_zero();

    for (int i = 0; i < heightp1; i++)
    {
      for (int j = 0; j < lengthp1; j++)
      {
        p3(i, lengthp3 - j - 1) = p1(i, lengthp1 - j - 1);
      }
    }


    for (int i = 0; i < heightp2; i++)
    {
      for (int j = 0; j < lengthp2; j++)
      {
        p3(heightp1 + i ,lengthp3 - j - 1) = p2(i, lengthp2 - j - 1);
      }
    }

  }

  return p3;
}

template <typename T>
bool iszero(Array<T> & poly)
{
  // check if all contents of poly are zero
  int numDims = 4;
  int totalElements = 1;
  for (int i = 0; i < numDims; i++)
  {
    totalElements *= poly.get_dim(i);
  }
  //    cout<<"total elements: "<<totalElements<<endl;

  int check = true;

  for(int i = 0; i < totalElements; i++)
  {
    check = (*poly.get_ptr_cpu(i)*(*poly.get_ptr_cpu(i)) < 1e-12);// is it zero?
    if (!check) // if it is not zero, return false
    {
      //            cout<<check<<endl;
      return check;
    }
  }
  return check;


}

Array<double> nodeFunctionTri(int in_index, int in_n_spts, Array<int> & index_location_Array)
    {
  Array<double> N_a; // Global node-specific shape function to be returned

  // Calculate number of nodes on each side
  int nNodesSide;
  nNodesSide =  calcNumSides(in_n_spts) ;

  // Get specific r,s,t index based on global index
  int II = int(index_location_Array(0,in_index));
  int JJ = int(index_location_Array(1,in_index));
  int KK = int(index_location_Array(2,in_index));

  //cout<< " II = "<<II<<" ; JJ = "<<JJ<<" ; KK = "<<KK<<endl;

  // Create polynomial functions specific to r,s,t nodes
  Array<double> T_Ir, T_Js, T_Kt, temp;

  T_Ir = shapePoly4Tri(II, nNodesSide);
  T_Js = shapePoly4Tri(JJ, nNodesSide);
  T_Kt = shapePoly4Tri(KK, nNodesSide);

  // Multiply polynomials (order of multiplication does matter in this case, as the differentiation
  // with respect to t --third row-- is different to that with respec to r or s)
  temp = multPoly(T_Ir,T_Js);
  N_a = multPoly(temp,T_Kt);

  return N_a;


    }


Array<int> linkTriangleNodes(int in_n_spts)
    {
  // first row in index_location_Array contains indeces of r arranged in ascending global node number;
  // second row contains indeces of s arranged in ascending global node number;
  // third row contains indeces of t arranged in ascending global node number;
  // refer to Hughes pp 169 to see link between r/s indeces ordering and global indeces ordering

  // Calculate number of nodes on each side
  int nNodesSide;
  nNodesSide =  calcNumSides(in_n_spts) ;
  Array<int> index_location_Array; // Global node-specific shape function to be returned

  // Initialize Arrays that will contain indices corresponding to node numbers
  // Used temporarily to make code a bit clearer
  Array<int> rind(1,in_n_spts); // stores r indeces; location in Array is global node number
  Array<int> sind(1,in_n_spts); // stores s indeces; location in Array is global node number
  Array<int> tind(1,in_n_spts); // stores t indeces; location in Array is global node number
  Array<int> temp; // temporary variable used while multiplying Arrays

  // Initialize counters
  int nNodesLeft = in_n_spts; // stores number of nodes left to assign
  int lastInd = 0; // stores last index updated
  int cycle = 0; // stores number of cycles done
  int nSide; // counts number of nodes in side of next cycle; recalculated at every cycle
  int nNodesLayer; // counts number of nodes in outer layer just processed

  // Initialize constant
  int order = nNodesSide + 2; //used in the numbering

  while(nNodesLeft != 0)
  {
    // Update values of nSide and nNodesLayer as cycling progresses
    nSide = calcNumSides(nNodesLeft);

    if (nNodesLeft >= 3)
    {
      rind(lastInd) = cycle + 1;
      sind(lastInd) = cycle + 1;
      tind(lastInd) = cycle + nSide;

      rind(lastInd + 1) = cycle + nSide;
      sind(lastInd + 1) = cycle + 1;
      tind(lastInd + 1) = cycle + 1;

      rind(lastInd + 2) = cycle + 1;
      sind(lastInd + 2) = cycle + nSide;
      tind(lastInd + 2) = cycle + 1;

      if(nNodesLeft == 3) break; //skip while loop and process Arrays
    }
    else // this means there is 1 node left
    {
      rind(lastInd) = cycle + 1;
      sind(lastInd) = cycle + 1;
      tind(lastInd) = cycle + 1;

      break; //skip while loop and process Arrays
    }

    // go along the outer borders of triangle in this cycle
    for (int i = 2; i <= nSide-1; i++)
    {
      // going along the bottom side
      rind(lastInd + i + 1) = cycle + i;
      sind(lastInd + i + 1) = cycle + 1;
      tind(lastInd + i + 1) = order - (cycle + i) - (cycle + 1); // t = order - r - s

      // going along the hypotenuse
      rind(lastInd + i + nSide - 1) = nSide - i + 1 + cycle;
      tind(lastInd + i + nSide - 1) = cycle + 1;
      sind(lastInd + i + nSide - 1) = order - (nSide - i + 1 + cycle) - (cycle + 1) ;

      // going along the left side
      rind(lastInd + i + 2*(nSide-1) - 1) = cycle + 1;
      sind(lastInd + i + 2*(nSide-1) - 1) = nSide - i + 1 + cycle;
      tind(lastInd + i + 2*(nSide-1) - 1) = order - (nSide - i + 1 + cycle) - (cycle + 1) ;

    }

    // recalculate values
    nNodesLayer = 3*(nSide - 1);
    lastInd += nNodesLayer;
    nNodesLeft -= nNodesLayer;
    cycle++;

  }
  // Process Arrays: assemble them by stacking them: use multPoly function
  // Order matters: r indeces go first, s indeces second, t indeces third
  temp = multPoly(rind,sind);

  index_location_Array = multPoly(temp,tind);

  return index_location_Array;

    }


Array<double> diffPoly(Array<double> & p, Array<int> & term2Diff)
    {
  /*
    Returns 3D Array; differentiates polynomial p with respect to dimensions specified by term2Diff
    term2Diff: n x 1 Array of rows to differentiate (negative integer i differentiates row abs(i) and multiplies that row by -1 ); rows enumerated starting at 1
   */

  int numTerms = term2Diff.get_dim(1); // find number of variables with respect to which to differentiate
  //cout<<"numTerms = "<<numTerms<<endl;
  int depthp = p.get_dim(2); // find number of layers of polynomial p (remember layers represent addition; rows multiplication; columns coefficients of polynomials)

  Array<double> finalp(1); // this is the polynomial that will be returned
  finalp(0) = 0;  // initialize the polynomial

  // Declare variables used in the for loop
  int row, coeff;
  for(int l = 0; l < numTerms; l++)
  {
    // Create copy of p(:,:,1)
    Array<double> diffp;
    diffp = p;

    // Specify row to manipulate
    row = term2Diff(l);

    // Switch used in triangles; when row is negative, row number abs(row) is differentiated and multiplied by -1

    if(row < 0)
    {
      coeff = -1;
      row = -(row+1); // make value positive to use it later as an index
    }
    else
    {
      coeff = 1;
      row = row-1;
    }

    // Start differentiation of rows

    for( int k = 0; k < depthp; k++)
    {
      int lengthrow = diffp.get_dim(1); // obtain number of elements in the row
      diffp(row,0,k) = 0; // coefficients shift due to change in power of term

      for( int j = 1; j < lengthrow; j++)
      {
        //Grab column in row to manipulate
        int power = lengthrow - j; //power corresponding to coefficient being manipulated
        diffp(row,j,k) = coeff*p(row,j-1,k)*power;
      }
    }

    finalp = addPoly(finalp,diffp);

  }
  return finalp;
    }


double evalPoly(Array<double> p, Array<double> coords)
{
  //need to check that length of coords is equal to height of p
  if(p.get_dim(0) != coords.get_dim(1))
  {
    FatalError("@evalPoly: number of coordinates at which to evaluate polynomial is not equal to number of polynomial variables");
  }

  int lengthp, heightp, depthp;
  lengthp = p.get_dim(1);
  heightp = p.get_dim(0);
  depthp = p.get_dim(2);

  double total = 0;
  double val, rowval,power;

  for(int k = 0; k < depthp; k++)
  {
    val = 1;


    for(int i = 0; i < heightp; i++)
    {
      rowval = 0;

      for(int j = 0; j < lengthp; j++)
      {
        power = lengthp - j - 1;
        rowval +=  p(i,j,k)*pow(coords(i),power);
      }

      val *= rowval;
      //cout<<"k = "<<k<<" ;  rowval = "<<rowval<<" ;  val = "<<val<<" ;  total = "<<total<<endl;
    }

    total += val;
  }

  return total;
}


void eval_dn_nodal_s_basis(Array<double> &dd_nodal_s_basis,
    Array<double> in_loc, int in_n_spts, int n_deriv)
{
  /*
    Function that returns the values of the nth derivatives of the shape function
    of nodes (rows in dd_nodal_s_basis) with respect to r and s (dr^n, ds^2, drds are in 1st, 2nd, 3rd
    columns respectively) given:
    in_loc: 2x1 Array of coordinates where derivatives are evaluated
    in_n_spts: number of nodes used to create triangle function ( n*(n+1)/2 where n is nodes on each
    side of the unit right triangle)
    n_deriv order of derivatives to take

    NOTE: number of nodes must be equal to number of rows in dd_nodal_s_basis
   */



  if(dd_nodal_s_basis.get_dim(0) != in_n_spts)
  {
    FatalError("@eval_dd_nodal_s_basis_new: number of nodes must be equal to number of rows in dd_nodal_s_basis");
  }

  if(dd_nodal_s_basis.get_dim(1) != 0.5*n_deriv*(n_deriv + 1))
  {
    if((n_deriv == 1 && dd_nodal_s_basis.get_dim(1) == 2) || (n_deriv ==0 && dd_nodal_s_basis.get_dim(1) == 1))
    {

    }
    else
    {
      FatalError("@eval_dd_nodal_s_basis_new: not enough columns in dd_nodal_s_basis to store all derivatives");
    }
  }


  // Obtain linked list of node enumeration for triangle given the number of nodes in_n_spts
  Array<int> nodeList;
  nodeList = linkTriangleNodes(in_n_spts);

  //cout<<"Linked list: "<<endl;
  //nodeList.print();

  // Start loop to find derivatives at each node
  // Shape function N_a of node a
  Array<double> N_a;
  // Differentiation of N_a
  Array<double> diff_N_a;
  // Array that contains variables with respect to which shape function is differentiated
  Array<int> diff_coord(1,2);
  // Array that contains values of r,s,t to be plugged in when evaluating differentiated polynomial
  Array<double> coords(1,3);
  coords(0) = in_loc(0);
  coords(1) = in_loc(1);
  coords(2) = -1 - in_loc(0) - in_loc(1); // value of t given values of r, s : t = -1 -r -s
  diff_coord(1) = -3; //differentiate 3rd row always and multiply by -1 because
  // t = -1 - s - r so
  // d [f(r)*g(s)*h(t)]/dr = d[f(r)]/dr*g(s)*h(t) +
  //                          f(r)*g(s)*(-1*d[h(t)]/dt)
  // and
  // d [f(r)*g(s)*h(t)]/ds = f(r)*d[g(s)]/ds*h(t) +
  //                          f(r)*g(s)*(-1*d[h(t)]/dt)
  // where d is partial diff

  int num_deriv_iterations = n_deriv*(n_deriv+1)/2 + (n_deriv <= 1?1:0);

  for(int a = 0; a < in_n_spts; a++) //in_n_spts
  {
    // Find shape function specific to node a
    N_a = nodeFunctionTri(a, in_n_spts,nodeList);

    // Loop through all possible combinations of derivatives of order n_deriv
    for(int i = 0; i < num_deriv_iterations; i++)
    {

      // Store value of 0th derivative in diff_N_a
      diff_N_a = N_a;

      // Differentiate with respect to r, (n_deriv - i) times
      // Specify row of polynomial in order to differentiate with respect to r
      diff_coord(0) = 1;
      for(int dr_count = 0; dr_count < n_deriv - i ; dr_count ++)
      {

        // Differentiate shape function with respect to one variable (r is 1, s is 2)
        diff_N_a = diffPoly(diff_N_a,diff_coord);
      }

      // Differentiate with respect to s, (i) times
      // Specify row of polynomial in order to differentiate with respect to r
      diff_coord(0) = 2;
      for(int ds_count = 0; ds_count < i ; ds_count ++)
      {

        // Differentiate shape function with respect to one variable (r is 1, s is 2)
        diff_N_a = diffPoly(diff_N_a,diff_coord);
      }

      // Evaluate differentiated shape function and store in Array dd_nodal_s_basis

      dd_nodal_s_basis(a, (i==num_deriv_iterations-1 ? (i==0?0:1):(i==0? i:(i+1)))) = evalPoly(diff_N_a,coords);
    }


  }
}
/*! END */
//----------------------------------------------------------------------------
// Linear equation solution by Gauss-Jordan elimination.
// a(1:n,1:n) is the coefficients input matrix.
// b(1:n) is the input matrix containing the right-hand side vector.
// On output, a(1:n,1:n) is replaced by its matrix inverse,
// and b(1:n) is replaced by the corresponding solution vector.
//
// From Numerical Recipes (http://www.nr.com/)
//----------------------------------------------------------------------------
void gaussj(int n, Array<double>& A, Array<double>& b)
{
  Array<int> indxc(n),indxr(n),ipiv(n);
  int icol,irow,i,j,k,l,ll;
  double big,dum,pivinv;

  icol=0;irow=0;
  for(i=0;i<n;i++){
    ipiv(i)=0;indxr(i)=0;indxc(i)=0;
  }
  //
  for(i=0;i<n;i++){
    big=0.0;
    for(j=0;j<n;j++){
      if(ipiv(j)!=1){
        for(k=0;k<n;k++){
          if(ipiv(k)==0){
            if(abs(A(k,j))>=big){
              big=abs(A(k,j));
              irow=k; icol=j;
            }
          }
        }
      }
    }
    ipiv(icol)=ipiv(icol)+1;
    if(irow!=icol){
      for(l=0;l<n;l++){
        dum=A(l,irow);
        A(l,irow)=A(l,icol);
        A(l,icol)=dum;
      }
      dum=b(irow);
      b(irow)=b(icol);
      b(icol)=dum;
    }
    indxr(i)=irow;
    indxc(i)=icol;
    if(A(icol,icol)==0.0){
      cout<<"Error: Singular matrix in gaussj"<<endl;
      exit(1);
    }
    pivinv=1.0/A(icol,icol);
    A(icol,icol)=1.0;
    for(l=0;l<n;l++){
      A(l,icol)=A(l,icol)*pivinv;
    }
    b(icol)=b(icol)*pivinv;
    for(ll=0;ll<n;ll++){
      if(ll!=icol){
        dum=A(icol,ll);
        A(icol,ll)=0.0;
        for(l=0;l<n;l++){
          A(l,ll)=A(l,ll)-A(l,icol)*dum;
        }
        b(ll)=b(ll)-b(icol)*dum;
      }
    }
  }
  //
  for(l=n-1;l>=0;l--){
    if(indxr(l)!=indxc(l)){
      for(k=0;k<n;k++){
        dum=A(indxr(l),k);
        A(indxr(l),k)=A(indxc(l),k);
        A(indxc(l),k)=dum;
      }
    }
  }
}

double flt_res(int N, Array<double>& wf, Array<double>& B, double k_0, double k_c, int ctype)
{
  int i;
  double norm, xm;
  Array<double> flt(N);
  double flt_res=0.0;

  norm = 0.0;
  for (i=0;i<N;i++)
  {
    flt(i) = wf(i)*exp(-6.0*pow(k_c*B(i),2));
    norm += flt(i);
  }
  for (i=0;i<N;i++)
    flt(i) /= norm;

  if(ctype==0) // Constrain 2nd moment
  {
    xm = 0.0;
    norm = 0.0;
    for (i=0;i<N;i++)
      xm += flt(i)*B(i);
    for (i=0;i<N;i++)
      norm += flt(i)*pow((B(i)-xm),2);
    flt_res = sqrt(12.0*norm) - 1.0/k_0;
  }
  else // Constrain cutoff frequency
  {
    flt_res = -exp((pow((-1.0*pi),2))/24.0);
    for (i=0;i<N;i++)
      flt_res += flt(i)*cos(B(i)*k_0*pi);
  }
  //cout<<"flt_res: "<<flt_res<<endl;
  return flt_res;
}

// Set an Array to zero
void zero_Array(Array <double>& in_Array)
{
  int dim_1_0 = in_Array.get_dim(0);
  int dim_1_1 = in_Array.get_dim(1);
  int dim_1_2 = in_Array.get_dim(2);
  int dim_1_3 = in_Array.get_dim(3);

  for (int i=0;i<dim_1_0;++i) {
    for (int j=0;j<dim_1_1;++j) {
      for (int k=0;k<dim_1_2;++k) {
        for (int l=0;l<dim_1_3;++l) {
          in_Array(i,j,k,l) = 0.0;
        }
      }
    }
  }
}

// Add Arrays M1 and M2
Array <double> add_Arrays(Array <double>& M1, Array <double>& M2)
{
  // Get dimensions of Arrays
  int dim_1_0 = M1.get_dim(0);
  int dim_1_1 = M1.get_dim(1);
  int dim_1_2 = M1.get_dim(2);
  int dim_1_3 = M1.get_dim(3);

  int dim_2_0 = M2.get_dim(0);
  int dim_2_1 = M2.get_dim(1);
  int dim_2_2 = M2.get_dim(2);
  int dim_2_3 = M2.get_dim(3);
  Array <double> sum;
  if(dim_1_0==dim_2_0 and dim_1_1==dim_2_1 and dim_1_2==dim_2_2 and dim_1_3==dim_2_3) {
    sum.setup(dim_1_0,dim_1_1,dim_1_2,dim_1_3);
    for (int i=0;i<dim_1_0;++i) {
      for (int j=0;j<dim_1_1;++j) {
        for (int k=0;k<dim_1_2;++k) {
          for (int l=0;l<dim_1_3;++l) {
            sum(i,j,k,l) = M1(i,j,k,l) + M2(i,j,k,l);
          }
        }
      }
    }
  }
  else
  {
    FatalError("Array dimensions are not compatible in sum function");
  }
  return sum;
}

// Multiply M1(L*M) by M2(M*N)
Array <double> mult_Arrays(Array <double>& M1, Array <double>& M2)
    {
  // Get dimensions of Arrays
  int dim_1_0 = M1.get_dim(0);
  int dim_1_1 = M1.get_dim(1);
  int dim_1_2 = M1.get_dim(2);
  int dim_1_3 = M1.get_dim(3);

  int dim_2_0 = M2.get_dim(0);
  int dim_2_1 = M2.get_dim(1);
  int dim_2_2 = M2.get_dim(2);
  int dim_2_3 = M2.get_dim(3);

  // Only 2D Arrays
  if(dim_1_2==1 and dim_1_3==1 and dim_2_2==1 and dim_2_3==1) {
    // Ensure consistent inner dimensions
    if(dim_1_1==dim_2_0) {
      Array <double> product(dim_1_0,dim_2_1);

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
          dim_1_0,dim_2_1,dim_1_1,1.0,M1.get_ptr_cpu(),dim_1_0,M2.get_ptr_cpu(),
          dim_2_0,0.0,product.get_ptr_cpu(),dim_1_0);
#else

      for (int i=0;i<dim_1_0;++i) {
        for (int j=0;j<dim_2_1;++j) {
          product(i,j) = 0.0;
          for (int k=0;k<dim_1_1;++k) {
            product(i,j) += M1(i,k)*M2(k,j);
          }
        }
      }

#endif

      return product;
    }
    else {
      cout << "ERROR: Array dimensions are not compatible in multiplication function" << endl;
      exit(1);
    }
  }
  else {
    cout << "ERROR: Array multiplication function can only multiply 2-dimensional Arrays together" << endl;
    exit(1);
  }
    }

// method to get transpose of a square Array
Array <double> transpose_Array(Array <double>& in_Array)
    {
  // Get dimensions of Arrays
  int dim_0 = in_Array.get_dim(0);
  int dim_1 = in_Array.get_dim(1);
  int dim_2 = in_Array.get_dim(2);
  int dim_3 = in_Array.get_dim(3);

  // Only 2D square Arrays
  if(dim_2==1 and dim_3==1 and dim_0==dim_1) {
    int i,j;
    Array <double> transpose(dim_1,dim_0);

    for(i=0;i<dim_0;i++) {
      for(j=0;j<dim_1;j++) {
        transpose(j,i)=in_Array(i,j);
      }
    }
    return transpose;
  }
  else {
    cout << "ERROR: Array transpose function only accepts a 2-dimensional square Array" << endl;
    exit(1);
  }
    }

// method to get inverse of a square matrix

Array <double> inv_Array(Array <double>& in_Array)
    {
  // Get dimensions of Array
  int dim_0 = in_Array.get_dim(0);
  int dim_1 = in_Array.get_dim(1);
  int dim_2 = in_Array.get_dim(2);
  int dim_3 = in_Array.get_dim(3);

  if(dim_2==1 and dim_3==1) {
    if(dim_0==dim_1)
    {
      // Gaussian elimination with full pivoting
      // not to be used where speed is paramount

      int i,j,k;
      int pivot_i, pivot_j;
      int itemp_0;
      double mag;
      double max;
      double dtemp_0;
      double first;
      Array <double> atemp_0(dim_0);
      Array <double> identity(dim_0,dim_0);
      Array <double> input(dim_0,dim_0);
      Array <double> inverse(dim_0,dim_0);
      Array <double> inverse_out(dim_0,dim_0);
      Array<int> swap_0(dim_0);
      Array<int> swap_1(dim_0);

      // setup input Array
      for(i=0;i<dim_0;i++)
        for(j=0;j<dim_0;j++)
          input(i,j) = in_Array(i,j);

      // setup swap Arrays
      for(i=0;i<dim_0;i++) {
        swap_0(i)=i;
        swap_1(i)=i;
      }

      // setup identity Array
      for(i=0;i<dim_0;i++) {
        for(j=0;j<dim_0;j++) {
          identity(i,j)=0.0;
        }
        identity(i,i)=1.0;
      }

      // make triangular
      for(k=0;k<dim_0-1;k++) {
        max=0;

        // find pivot
        for(i=k;i<dim_0;i++) {
          for(j=k;j<dim_0;j++) {
            mag=input(i,j)*input(i,j);
            if(mag>max) {
              pivot_i=i;
              pivot_j=j;
              max=mag;
            }
          }
        }

        // swap the swap Arrays
        itemp_0=swap_0(k);
        swap_0(k)=swap_0(pivot_i);
        swap_0(pivot_i)=itemp_0;
        itemp_0=swap_1(k);
        swap_1(k)=swap_1(pivot_j);
        swap_1(pivot_j)=itemp_0;

        // swap the columns
        for(i=0;i<dim_0;i++) {
          atemp_0(i)=input(i,pivot_j);
          input(i,pivot_j)=input(i,k);
          input(i,k)=atemp_0(i);
        }

        // swap the rows
        for(j=0;j<dim_0;j++) {
          atemp_0(j)=input(pivot_i,j);
          input(pivot_i,j)=input(k,j);
          input(k,j)=atemp_0(j);
          atemp_0(j)=identity(pivot_i,j);
          identity(pivot_i,j)=identity(k,j);
          identity(k,j)=atemp_0(j);
        }

        // subtraction
        for(i=k+1;i<dim_0;i++) {
          first=input(i,k);
          for(j=0;j<dim_0;j++) {
            if(j>=k) {
              input(i,j)=input(i,j)-((first/input(k,k))*input(k,j));
            }
            identity(i,j)=identity(i,j)-((first/input(k,k))*identity(k,j));
          }
        }

        //exact zero
        for(j=0;j<k+1;j++) {
          for(i=j+1;i<dim_0;i++) {
            input(i,j)=0.0;
          }
        }
      }

      // back substitute
      for(i=dim_0-1;i>=0;i=i-1) {
        for(j=0;j<dim_0;j++) {
          dtemp_0=0.0;
          for(k=i+1;k<dim_0;k++) {
            dtemp_0=dtemp_0+(input(i,k)*inverse(k,j));
          }
          inverse(i,j)=(identity(i,j)-dtemp_0)/input(i,i);
        }
      }

      // swap solution rows
      for(i=0;i<dim_0;i++) {
        for(j=0;j<dim_0;j++) {
          inverse_out(swap_1(i),j)=inverse(i,j);
        }
      }

      return inverse_out;
    }
    else {
      cout << "ERROR: Can only obtain inverse of a square Array" << endl;
      exit(1);
    }
  }
  else {
    cout << "ERROR: Array you are trying to invert has > 2 dimensions" << endl;
    exit(1);
  }
    }

// integrand used to find Local Fourier Spectral filters in triangles
double filter_integrand_tris(double r, double s) {
  Array<double> rvect(2);
  rvect(0) = r;
  rvect(1) = s;

  // define properties of the kernel function
  double h = run_input.filter_width; // measure of the wavenumbers left unfiltered

  double distance = LOCAL_ELE_OF_INTEREST->reference_element_norm(rvect, LOCAL_X0);

  if (distance < 1e-10) distance = 1e-10;

  double kernel_eval = h * j1( h * distance)/distance; // this function is tophat in spectral domain
  //eval_dubiner_basis_2d(r,s,i,in_order)
  double basis_eval = eval_dubiner_basis_2d(r,s,LOCAL_BASIS_INDEX,LOCAL_ORDER);

  return kernel_eval*basis_eval;
}


// integrand used to find Local Fourier Spectral filters in tetrahedral elements
double filter_integrand_tets(double r, double s, double t) {
  Array<double> rvect(3);
  rvect(0) = r;
  rvect(1) = s;
  rvect(2) = t;

  // define properties of the kernel function
  double h = run_input.filter_width; // measure of the wavenumbers left unfiltered

  double distance = LOCAL_ELE_OF_INTEREST->reference_element_norm(rvect, LOCAL_X0);

  Array<double> vec1; vec1(0) = 0; vec1(1) = 0; vec1(2) = 0;
  Array<double> vec2; vec2(0) = 1; vec2(1) = 1; vec2(2) = 1;

  if (distance < 1e-10) {
    distance = 1e-10;
  }

  double kernel_eval = h * besselThreeHalves(h * distance)/distance; // this function is tophat in spectral domain

  double basis_eval = eval_dubiner_basis_3d(r,s,t,LOCAL_BASIS_INDEX,LOCAL_ORDER);

  return kernel_eval * basis_eval;
}


double ylower(double /*x*/) { return -1;}
double yupper(double x) {return -x;}
// populate the Local Fourier Spectral filter matrix in triangles
void fill_stabilization_interior_filter_tris(Array<double>& filter_matrix, int order,
    Array<double>& loc_upts, eles_tris *element) {
  // store the local globa variables
  LOCAL_ORDER = order;
  LOCAL_ELE_OF_INTEREST = element;
  LOCAL_X0.setup(2);
  int n = filter_matrix.get_dim(0); // assume it's a square matrix already

  double abserr = 0, relerr = 1e-10;

  double result, errest, flag;
  int nofun;
  for (int i = 0; i < n; i++) {

    LOCAL_X0(0) = loc_upts(0,i);
    LOCAL_X0(1) = loc_upts(1,i);

    for (int j = 0; j < n; j++) {

      LOCAL_BASIS_INDEX = j; // update basis number

      quad2(filter_integrand_tris, -1, 1,
          ylower,yupper, abserr, relerr,
          result, errest, nofun,flag);

      filter_matrix(i,j) = result;
    }
  }

}


// limits of integration for tetrahedron
double ylower_tet(double /*x*/) { return -1;}
double yupper_tet(double z) {return -z;}
double zlower_tet(double /*x*/, double /*y*/) { return -1;}
double zupper_tet(double x, double y) {return -1 - x - y;}
// populate the Local Fourier Spectral filter matrix in 2D for triangles
void fill_stabilization_interior_filter_tets(Array<double>& filter_matrix, int order,
    Array<double>& loc_upts, eles *element) {
  // store the local globa variables
  LOCAL_ORDER = order;
  LOCAL_ELE_OF_INTEREST = element;
  LOCAL_X0.setup(3); // store location of current solution point
  // of interest globally

  int n = filter_matrix.get_dim(0); // assume it's a square matrix already

  double abserr = 0, relerr = 1e-10;

  double result, errest, flag;
  int nofun;
  for (int i = 0; i < n; i++) {

    LOCAL_X0(0) = loc_upts(0,i);
    LOCAL_X0(1) = loc_upts(1,i);
    LOCAL_X0(2) = loc_upts(2,i);



    for (int j = 0; j < n; j++) {

      LOCAL_BASIS_INDEX = j; // update basis number

      quad3(filter_integrand_tets, -1, 1,
          ylower_tet, yupper_tet,
          zlower_tet, zupper_tet,
          abserr, relerr,
          result, errest, nofun,flag);

      filter_matrix(i,j) = result;

    }
  }

}




// evaluates the polynomial coeffs(n)*x^n + coeffs(n-1)*x^{n-1) + ... + coeffs(1)*x
double filter_poly(std::vector<double>& coeffs, double x) {
  int n_coeffs = coeffs.size(); // number of coefficients

  double result = 0;
  for (int i = 1; i <= n_coeffs; i++)
    result += coeffs[i-1]*pow(x, n_coeffs - i + 1);

  return result;
}

// computes the entry of the boundary filter matrix
double boundary_filter_radial_function(Array<double>& loc_fpts,
    Array<double>& loc_upts,
    std::vector<double>& coeffs,
    eles *element,
    int row, int col) {


  int n_dims = loc_fpts.get_dim(0); // number of dimensions
  int nf = loc_fpts.get_dim(1); // number of flux points
  Array<double> temp_upt(n_dims), temp_fpt(n_dims); // place holders for locations of solution and flux points

  // copy location of current solution and flux points of interest
  for (int ll = 0; ll < n_dims; ll++) {
    temp_upt(ll) = loc_upts(ll, row);
    temp_fpt(ll) = loc_fpts(ll, col);
  }

  double curr_distance = filter_poly(coeffs,
      element->reference_element_norm(temp_upt,temp_fpt));
  if (curr_distance < 1e-12) {
    return 1.;
  }

  double sum = 0;
  for (int k = 0; k < nf; k++) {
    if (k != col) {

      // copy location of current solution and flux points of interest
      for (int ll = 0; ll < n_dims; ll++) {
        temp_upt(ll) = loc_upts(ll, row);
        temp_fpt(ll) = loc_fpts(ll, k);
      }

      sum += 1./filter_poly(coeffs, element->reference_element_norm(temp_upt,temp_fpt));
    }
  }



  //  cout << "i = " << row << "; j = " << col << endl;
  //  cout << "sum = " << sum << "; poly(xij*) = " << filter_poly(coeffs,
  //                                                                    element->reference_element_norm(temp_upt,temp_fpt)) << endl;
  //  cout << "rad_func = " << element->reference_element_norm(temp_upt,temp_fpt) << endl;

  return 1./(1. + sum * curr_distance);

}

// cost function to be minimized in order to find coefficients that allow boundary filter matrix
// to satisfy the constraints of: conservation and plane preservation
double filter_cost_function(Array<double>& loc_fpts, Array<double>& loc_upts,
    std::vector<double>& coeffs, eles *element, int row) {
  int nf = loc_fpts.get_dim(1); // number of flux points
  int n_dims = loc_fpts.get_dim(0); // number of dimensions
  Array<double> row_temp(nf);

  for (int j = 0; j < nf; j++) {
    row_temp(j) = boundary_filter_radial_function(loc_fpts, loc_upts, coeffs, element,
        row, j);
  }

  // assemble the sum of the squares of the error
  double result = 0;
  for (int dim = 0; dim < n_dims; dim++) {
    double sum = 0;

    for (int k = 0; k < nf; k++)
      sum += row_temp(k)*loc_fpts(dim, k);

    result += pow(sum - loc_upts(dim, row), 2);
  }

  return result;
}

/*
 * Additional local global variables to be used for function optimization
 */
static Array<double> LOCAL_LOC_FPTS;
static Array<double> LOCAL_LOC_UPTS;
static eles *LOCAL_ELEMENT;
static int LOCAL_ROW;

double filter_function_simple(std::vector<double>& x) {
  return filter_cost_function(LOCAL_LOC_FPTS, LOCAL_LOC_UPTS, x, LOCAL_ELEMENT, LOCAL_ROW);
}

// wrapper for the simplex_min_method function
// finds the values of coefficients that minimize the filter_function
void minimum_search(Array<double>& loc_fpts, Array<double>& loc_upts,
    std::vector<double>& init, eles *element, int row) {
  //  int n_coeffs = init.size();
  std::vector<double> solution; // vector that will hold the solution

  // create lambda function to be passed on to the optimizer
  LOCAL_LOC_FPTS = loc_fpts;
  LOCAL_LOC_UPTS = loc_upts;
  LOCAL_ELEMENT = element;
  LOCAL_ROW = row;
  //  auto filter_function_simple =
  //      [&loc_fpts, &loc_upts, element, row] (std::vector<double>& x) {
  //      return filter_function(loc_fpts, loc_upts, x, element, row);
  //    };

  solution = simplex_min_method(filter_function_simple, init);

  //  for (int i = 0; i < n_coeffs; i++) {
  //      cout << "init[" << i <<"] = " << init[i] << endl;
  //      cout << "solution[" << i <<"] = " << solution[i] << endl;
  //    }

  init = solution;

}


// populate the filter matrix that communicates information from boundaries to interior of element
void fill_stabilization_boundary_filter(Array<double>& filter_matrix, Array<double>& loc_fpts,
    Array<double>& loc_upts, eles *element) {
  int n = loc_upts.get_dim(1); // number of solution points
  int nf = loc_fpts.get_dim(1); // number of flux points
  int n_dims = loc_upts.get_dim(0); // number of dimensions
  cout << "at fill_stabilization_boundary_filter" << endl;
  cout << n << " " << nf << endl;

  filter_matrix.setup(n,nf);

  for (int i = 0; i < n; i++) {

    std::vector<double> coeffs(n_dims, 1.); // coefficients that need to be found for each row of the filter matrix

    minimum_search(loc_fpts, loc_upts, coeffs, element, i);
    //      /*! Temporary change; call filter_function to check implementation
    //       */

    cout << "i = " << i << " filter_function = " <<
        filter_cost_function(loc_fpts, loc_upts, coeffs, element, i) << "; ";
    for (int j = 0; j < n_dims; j++) {
      cout <<coeffs[j]<<", ";
    }
    cout << endl;


    for (int j = 0; j < nf; j++) {
      filter_matrix(i,j) = boundary_filter_radial_function(loc_fpts,
          loc_upts, coeffs, element, i, j);
    }
  }
}

// use cblas_dgemm, dgemm, or cublasDgemm
// to perform C = (alpha*A*B) + (beta*C)
void dgemm_wrapper(int Arows, int Bcols, int Acols,
    double alpha,
    double *A_matrix, int Astride,
    double *B_matrix, int Bstride,
    double beta,
    double *C_matrix, int Cstride) {

#ifdef _CPU
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      Arows, Bcols, Acols,
      alpha, A_matrix, Astride,
      B_matrix, Bstride,
      beta, C_matrix, Cstride);

#elif defined _NO_BLAS
  dgemm(Arows,Bcols,Acols,
      alpha, beta, A_matrix,
      B_matrix,
      C_matrix);
#endif
#endif

#ifdef _GPU

  cublasDgemm('N','N',Arows,Bcols,Acols,
      alpha, A_matrix, Astride,
      B_matrix, Bstride,
      beta, C_matrix, Cstride);
#endif
}

/*! Checks if the file fileName exists
 * Input: fileName : name of the file whose existence is checked
 */
bool fileExists(const std::string& fileName) {
  struct stat buffer;
  return (stat (fileName.c_str(), &buffer) == 0);
}

/*! Counts the number of missing files
 * Input: fileNames : names of the files whose non-existence is counted
 *        numFiles: number of files to count, must match the number of elements in fileNames
 */
int countNumberMissingFiles(const std::string fileNames [], int numFiles) {
  int numFilesMissing = 0; // counter of the number of matrix files missing
  for (int i = 0; i < numFiles; i++) {
    std::string fileName = fileNames[i];
    if (!fileExists(fileName)) {
      numFilesMissing++;
    }
  }

  return numFilesMissing;
}


/*! Transforms string to double
 * Input: s : string to be transformed
 */
std::string num2str( const double num) {
  return static_cast<ostringstream*>( &(ostringstream() << num) )->str();
}

/*! Integrand in the integral used for calculating the Bessel function
 * of the first kind
 */
double besselIntegrand(double t) {
  return pow(1 - pow(t,2), -0.5 + LOCAL_BESSEL_NU) * cos(t * LOCAL_BESSEL_X);
}

/*! Calculates the Bessel function of the first kind: J_{\nu}(x)
 * where \nu is real and greater than -1/2
 */
double besselj(double nu, double x) {
  double integral, errest, flag;
  int nofun;
  LOCAL_BESSEL_NU = nu;
  LOCAL_BESSEL_X = x;
  quanc8(besselIntegrand, 0., 1.,
      0., 1e-10,
      integral, errest, nofun, flag);
  return pow(2, 1-nu) * pow(x, nu)
      /(sqrt(pi) * tgamma(0.5 + nu))
      * integral;
}

/*! Calculates the Bessel function of the first kind J_{3/2}(x)
 */
double besselThreeHalves(double x) {
  return 2 * (sin(x) - x*cos(x))/ (sqrt(2*pi) * pow(x, 1.5));
}


/*! Read an array in binary format */
void fromBinary(double* number, std::ifstream& file) {
  // read the rest of the array
  file.read((char*) number, sizeof(double));
}

void toBinary(double* number, std::ofstream& file) {
  // write the rest of the array
  file.write((char*) number, sizeof(double));
}


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

/*! Routing to compute y = alpha*x + y for vectors x and y - similar to BLAS's daxpy */
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
