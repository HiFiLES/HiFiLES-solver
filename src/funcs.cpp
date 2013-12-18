/*!
 * \file funcs.cpp
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

#include <iomanip>
#include <iostream>
#include <cmath>

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

#include "../include/funcs.h"
#include "../include/array.h"
#include "../include/cubature_1d.h"
#include "../include/global.h"

using namespace std;

// #### global functions ####

// evaluate lagrange basis

double eval_lagrange(double in_r, int in_mode, array<double>& in_loc_pts)
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

double eval_d_lagrange(double in_r, int in_mode, array<double>& in_loc_pts)
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

double eval_dd_lagrange(double in_r, int in_mode, array<double>& in_loc_pts)
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


void get_opp_3_tri(array<double>& opp_3, array<double>& loc_upts_tri, array<double>& loc_1d_fpts, array<double>& vandermonde_tri, array<double>& inv_vandermonde_tri, int n_upts_per_tri, int order, double c_tri, int vcjh_scheme_tri)
{

  array<double> Filt(n_upts_per_tri,n_upts_per_tri);
  array<double> opp_3_dg(n_upts_per_tri, 3*(order+1));
  array<double> m_temp;

  compute_filt_matrix_tri(Filt,vandermonde_tri,inv_vandermonde_tri,n_upts_per_tri,order,c_tri,vcjh_scheme_tri,loc_upts_tri);

  get_opp_3_dg(opp_3_dg, loc_upts_tri, loc_1d_fpts, n_upts_per_tri, order);
  m_temp = mult_arrays(Filt,opp_3_dg);
  opp_3 = array<double> (m_temp);
}

void get_opp_3_dg(array<double>& opp_3_dg, array<double>& loc_upts_tri, array<double>& loc_1d_fpts, int n_upts_per_tri, int order)
{

  int i,j,k;
  array<double> loc(2);

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

// Compute a modal filter matrix, given Vandermonde matrix and inverse
void compute_modal_filter(array <double>& filter_upts, array<double>& vandermonde, array<double>& inv_vandermonde, int N)
{
	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	int i,j;
	array <double> modal(N,N), mtemp(N,N);

	zero_array(modal);
	zero_array(filter_upts);

	// Modal coefficients
	double eta,Cp;
	Cp=-100.0; // Dubiner SVV filter strength coeff.
	double alpha = Cp/N; // Full form: alpha = Cp*(N+!)*dt/delta

	for(i=0;i<N;i++)
	{
		//modal(i,i)=1.0;	// Sharp modal cutoff filter
		eta = i/(N+1.0);
		modal(i,i)=exp(alpha*pow(eta,2*(N-1))); // Dubiner SVV 2D exp filter. MUST be even power
	}
	// Sharp modal cutoff filter
	//modal(N-1,N-1)=0.0;

	cout<<"modal coeffs:"<<endl;
	modal.print();

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,vandermonde.get_ptr_cpu(),N,modal.get_ptr_cpu(),N,0.0,mtemp.get_ptr_cpu(),N);

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,mtemp.get_ptr_cpu(),N,inv_vandermonde.get_ptr_cpu(),N,0.0,filter_upts.get_ptr_cpu(),N);

	#else // inefficient matrix multiplication

	// TODO: finish coding
	int i,j;
	array<double> mtemp(N,N);

	for(i=0;i<N-1;i++)
		filter_upts(i,i) = 1.0;

	mtemp = mult_arrays(inv_vandermonde,filter_upts);
	filter_upts = mult_arrays(mtemp,vandermonde);

	#endif
}

void compute_filt_matrix_tri(array<double>& Filt, array<double>& vandermonde_tri, array<double>& inv_vandermonde_tri, int n_upts_tri, int order, double c_tri, int vcjh_scheme_tri, array<double>& loc_upts_tri)
{

  // -----------------
  // VCJH Filter
  // -----------------
  double ap;
  double c_plus;
  double c_plus_1d, c_sd_1d, c_hu_1d;

  array<double> c_coeff(order+1);
  array<double> mtemp_0, mtemp_1, mtemp_2;
  array<double> K(n_upts_tri,n_upts_tri);
  array<double> Identity(n_upts_tri,n_upts_tri);
  array<double> Filt_dubiner(n_upts_tri,n_upts_tri);
  array<double> Dr(n_upts_tri,n_upts_tri);
  array<double> Ds(n_upts_tri,n_upts_tri);
  array<double> tempr(n_upts_tri,n_upts_tri);
  array<double> temps(n_upts_tri,n_upts_tri);
  array<double> D_high_order_trans(n_upts_tri,n_upts_tri);
  array<double> vandermonde_tri_trans(n_upts_tri,n_upts_tri);

  array<array <double> > D_high_order;
  array<array <double> > D_T_D;
  
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
  
  cout << "c_tri " << c_tri << endl;
  
  run_input.c_tri = c_tri;

	// Evaluate the derivative normalized of Dubiner basis at position in_loc
	for (int i=0;i<n_upts_tri;i++) {
    for (int j=0;j<n_upts_tri;j++) {
      tempr(i,j) = eval_dr_dubiner_basis_2d(loc_upts_tri(0,i),loc_upts_tri(1,i),j,order);
      temps(i,j) = eval_ds_dubiner_basis_2d(loc_upts_tri(0,i),loc_upts_tri(1,i),j,order);
    }
  }

  //Convert to nodal derivatives
  Dr = mult_arrays(tempr,inv_vandermonde_tri);
  Ds = mult_arrays(temps,inv_vandermonde_tri);

	//Create identity matrix
  zero_array(Identity);

  for (int i=0;i<n_upts_tri;++i)
    Identity(i,i) = 1.;

	// Set array with binomial coefficients multiplied by value of c
	for(int k=0; k<(order+1);k++) {
		c_coeff(k) = (1./n_upts_tri)*(factorial(order)/( factorial(k)*factorial(order-k) ));
    //cout << "k=" << k << "coeff= " << c_coeff(k) << endl;
  }

  // Initialize K to zero
  zero_array(K);

  // Compute D_transpose*D
  D_high_order.setup(order+1);
  D_T_D.setup(order+1);

  for (int k=0;k<(order+1);k++)
  {
    int m = order-k;
    D_high_order(k) = array<double> (Identity);
    for (int k2=0;k2<k;k2++)
      D_high_order(k) = mult_arrays(D_high_order(k),Ds);
    for (int m2=0;m2<m;m2++)
      D_high_order(k) = mult_arrays(D_high_order(k),Dr);
    //cout << "k=" << k << endl;
    //cout<<"D_high_order(k)"<<endl;
    //D_high_order(k).print();
    //cout << endl;

    D_high_order_trans = transpose_array(D_high_order(k));
    D_T_D(k) = mult_arrays(D_high_order_trans,D_high_order(k));

    //mtemp_2 = transpose_array(vandermonde_tri);
    //mtemp_2 = mult_arrays(mtemp_2,D_high_order(k));
    //mtemp_2 = mult_arrays(mtemp_2,vandermonde_tri);
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
  vandermonde_tri_trans = transpose_array(vandermonde_tri);
  mtemp_0 = mult_arrays(vandermonde_tri,vandermonde_tri_trans);

  //filter
  mtemp_1 = array<double>(mtemp_0);
  mtemp_1 = mult_arrays(mtemp_1,K);

  for (int i=0;i<n_upts_tri;i++)
    for (int j=0;j<n_upts_tri;j++)
      mtemp_1(i,j) += Identity(i,j);

  Filt = inv_array(mtemp_1);
  Filt_dubiner = mult_arrays(inv_vandermonde_tri,Filt);
  Filt_dubiner = mult_arrays(Filt_dubiner,vandermonde_tri);

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

  Filt = mult_arrays(vandermonde_tri,Filt_dubiner);
  Filt = mult_arrays(Filt,inv_vandermonde_tri);

  cout << "Filt_dubiner_diag" << endl;
  Filt_dubiner.print();

  cout << "Filt_diag" << endl;
  Filt.print();
  */

}


double eval_div_dg_tri(array<double> &in_loc , int in_edge, int in_edge_fpt, int in_order, array<double> &in_loc_fpts_1d)
{
  int n_upts_tri = (in_order+1)*(in_order+2)/2;

  double r,s,t;
  double integral, edge_length, gdotn_at_cubpt;
  double div_vcjh_basis;

  array<double> mtemp_0((in_order+1),(in_order+1));
  array<double> gdotn((in_order+1),1);
  array<double> coeff_gdotn((in_order+1),1);
  array<double> coeff_divg(n_upts_tri,1);

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

  mtemp_0 = inv_array(mtemp_0);
  coeff_gdotn = mult_arrays(mtemp_0,gdotn);

  // 2. Perform the edge integrals to obtain coefficients sigma_i
  for (int i=0;i<n_upts_tri;i++)
  {
	   cubature_1d cub1d(20);  // TODO: CHECK STRENGTH
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

// get intel mkl csr 4 array format
void array_to_mklcsr(array<double>& in_array, array<double>& out_data, array<int>& out_cols, array<int>& out_b, array<int>& out_e)
{
	int i,j;

	double tol=1e-24;
	int nnz=0;
	int pos=0;
	int new_row=0;

	array<double> temp_data;
	array<int> temp_cols, temp_b, temp_e;

	for(j=0;j<in_array.get_dim(0);j++)
	{
		for(i=0;i<in_array.get_dim(1);i++)
		{
			if((in_array(j,i)*in_array(j,i))>tol)
			{
				nnz++;
			}
		}
	}

	temp_data.setup(nnz);
	temp_cols.setup(nnz);
	temp_b.setup(in_array.get_dim(0));
	temp_e.setup(in_array.get_dim(0));

	pos=0;

	for(j=0;j<in_array.get_dim(0);j++)
	{
		for(i=0;i<in_array.get_dim(1);i++)
		{
			if((in_array(j,i)*in_array(j,i))>tol)
			{
				temp_data(pos)=in_array(j,i);
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

void array_to_ellpack(array<double>& in_array, array<double>& out_data, array<int>& out_cols, int& nnz_per_row)
{

  double zero_tol = 1.0e-12;

  int n_rows = in_array.get_dim(0);
  int n_cols = in_array.get_dim(1);
  nnz_per_row = 0;
  int temp;

  for (int i=0;i<n_rows;i++)
  {
    temp = 0;
    for (int j=0;j<n_cols;j++)
    {
      //cout << "in_array=" << in_array(i,j) << endl;
      if (abs(in_array(i,j)) >= zero_tol)
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
      if ( abs(in_array(i,j)) >= zero_tol)
      {
        index=i+count*n_rows;
        out_data(index) = in_array(i,j);
        out_cols(index) = j;
        count++;
      }
    }
  }


}


array<double> rs_to_ab(double in_r, double in_s)
{
	array<double> ab(2);

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
void array_to_cusparse_csr(array<double>& in_array, cusparseHybMat_t &hyb_array, cusparseHandle_t& handle)
{
  int n_rows = in_array.get_dim(0);
  int n_cols = in_array.get_dim(1);

  cout << "Converting to hybrid format" << endl;

  array<int> nnz_per_row(n_rows);
  int nnzTotalDevHostPtr;

  //cusparseCreateHybMat(&hyb_array);
  cusparseMatDescr_t mat_description;
  cusparseStatus_t status;

  status = cusparseCreateMatDescr(&mat_description);
  if (status != CUSPARSE_STATUS_SUCCESS){
    cout << "error create Mat Desc" << endl;
    exit(1);
  }

  cusparseSetMatType(mat_description,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(mat_description,CUSPARSE_INDEX_BASE_ZERO);

  cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, n_rows, n_cols,mat_description,in_array.get_ptr_gpu(),n_rows,nnz_per_row.get_ptr_cpu(),&nnzTotalDevHostPtr);

  const double* gpu_ptr = in_array.get_ptr_gpu();

  cusparseDdense2csr(handle,n_rows,n_cols,mat_description,gpu_ptr,n_rows,nnz_per_row.get_ptr_cpu(),hyb_array,NULL,CUSPARSE_HYB_PARTITION_AUTO);
}
*/
#endif

array<double> rst_to_abc(double in_r, double in_s, double in_t) // CHECK (PASSING)
{
	array<double> abc(3);

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
		array<double> ab;

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
		array<double> ab;

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
		array<double> ab;

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
		array<double> abc;

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
		array<double> abc;

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
	else
	{
		cout << "ERROR: Invalid mode when evaluating basis ...." << endl;
	}

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

double compute_eta(int vcjh_scheme, double order)
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


// Method that searches a value in a sorted array without repeated entries and returns position in array
int index_locate_int(int value, int* array, int size)
{
	int ju,jm,jl;
	int ascnd;

	jl = 0;
	ju = size-1;

	if (array[ju] <= array[0] && ju!=0)
	{
		cout << "ERROR, array not sorted, exiting" << endl;
    cout << "size= " << size << endl;
    cout << "array[0] = " << array[0] << endl;
    cout << "array[size-1] = " << array[ju] << endl;
		exit(1);
	}

	while(ju-jl > 1)
	{
		jm = (ju+jl) >> 1;
		if (value>=array[jm])
		{
			jl=jm;
		}
		else
		{
			ju=jm;
		}
	}

	if (value == array[0])
	{
		return 0;
	}
	else if (value == array[size-1])
	{
		return size-1;
	}
	else if (value == array[jl])
	{
		return jl;
	}
	else
	{
		return -1;
	}
}

void eval_isentropic_vortex(array<double>& pos, double time, double& rho, double& vx, double& vy, double& vz, double& p, int n_dims)
{
  array<double> relative_pos(n_dims);

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


void eval_sine_wave_single(array<double>& pos, array<double>& wave_speed, double diff_coeff, double time, double& rho, array<double>& grad_rho, int n_dims)
{

  array<double> relative_pos(n_dims);
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


void eval_sine_wave_group(array<double>& pos, array<double>& wave_speed, double diff_coeff, double time, double& rho, array<double>& grad_rho, int n_dims)
{

  array<double> relative_pos(n_dims);
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


void eval_sphere_wave(array<double>& pos, array<double>& wave_speed, double time, double& rho, int n_dims)
{

  array<double> relative_pos(n_dims);
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

void eval_couette_flow(array<double>& pos, double in_gamma, double in_R_ref, double in_u_wall, double in_T_wall, double in_p_bound, double in_prandtl, double time, double& ene, array<double>& grad_ene, int n_dims)
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
void eval_poly_ic(array<double>& pos, double rho, array<double>& ics, int n_dims)
{
	// Take N user-specified coefficients {a,b,c,...,n} to construct a polynomial of the form
	// u = a + bx + cx^2 + ... + nx^N (1D)
	// In 2D and 3D, add extra coeffs for mixed terms xy, xyz, x^2y etc.
	array <double> c(13);

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

/*! Functions used in evaluation of shape functions and its 1st and 2nd derivatives
BEGIN:*/
array<double> convol(array<double> & polynomial1, array<double> & polynomial2)
{
    // Accepts only row vectors that represent polynomials
    // Get lengths
    int sizep1 = polynomial1.get_dim(1);
    int sizep2 = polynomial2.get_dim(1);

    // Allocate memory for result of multiplication of polynomials
    array<double> polynomial3;
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

array<double> LagrangeP(int order, int node, array<double> & subs)
{
    //Function that finds the coefficients of the Lagrange polynomial
    /*
    % order: order of the polynomial
    % node: index of xi corresponding to point where polynomial equals 1
    % subs: polynomial that is substituted for variable in lagrange polynomial
    */
    double range[] = {-1.0,1.0}; // range over which nodes are located

// xi: array with location of points where function is zero

    array<double> xi = createEquispacedArray(range[0], range[1], order+1);

    int constInSubs = subs.get_dim(1); // location of the constant term in polynomial subs
    // this is just the last term of the polynomial
// Constructing the polynomial

    array<double> num;
    num(0) = 1; // initalize this array to 1
// Do the same for the denominator

    array<double> den;
    den(0) = 1; // initalize this array to 1

// declare temporary variables
    array<double> term;
    array<double> tempConstant;

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


array<double> shapePoly4Tri(int I, int nNodesSide)
{
    /*
    % returns the polynomial function T_I(r) in the polynomial format
    % Array values are coefficients of monomials of increasing order
    % I : index of node in triangle
    % nNodesSide: number of nodes in side
    For specifics, refer to Hughes, pp 166
    */

    array<double> T_I;// special lagrange polynomial corresponding to a node in the triangle
    // this is the result

    if(I == 1)
    {
        T_I(0) = 1; // return constant 1
        return T_I;
    }
    else
    {
        int order = I - 1; // as described in Hughes pp 167

        double range[] = {-1.0,1.0}; // range over which nodes are located

// xi: array with location of points where function is zero
        array<double> xi = createEquispacedArray(range[0], range[1], nNodesSide);

        double r_I = xi(I-1); // get location of node I in the range

        // Create polynomial to substitute to create polynomial related to triangles
        array<double> subs(1,2);
        // Specify coefficient of r
        subs(0) = 2./(r_I + 1.);
        subs(1) = (1. - r_I)/(1. + r_I);

        T_I = LagrangeP(order,I,subs); //note that order = I - 1

        return T_I;

    }
}


array<double> createEquispacedArray(double a, double b, int nPoints)
{
    array<double> xi(1,nPoints);
    for ( int i = 0; i < nPoints ; i++)
    {
        xi(i) = (i)/(double(nPoints)-1)*(b - a) + a;
    }
    return xi;
}


array<double> addPoly(array<double> & p1, array<double> & p2)
{
    // Returns a 3D array; each layer represents a multiplication of polynomials
    array<double> p3; // return polynomial

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
array<T> multPoly(array<T> & p1, array<T> & p2)
{
    array<T> p3; // return polynomial
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
bool iszero(array<T> & poly)
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

array<double> nodeFunctionTri(int in_index, int in_n_spts, array<int> & index_location_array)
{
    array<double> N_a; // Global node-specific shape function to be returned

    // Calculate number of nodes on each side
    int nNodesSide;
    nNodesSide =  calcNumSides(in_n_spts) ;

    // Get specific r,s,t index based on global index
    int I = int(index_location_array(0,in_index));
    int J = int(index_location_array(1,in_index));
    int K = int(index_location_array(2,in_index));

    //cout<< " I = "<<I<<" ; J = "<<J<<" ; K = "<<K<<endl;

    // Create polynomial functions specific to r,s,t nodes
    array<double> T_Ir, T_Js, T_Kt, temp;

    T_Ir = shapePoly4Tri(I, nNodesSide);
    T_Js = shapePoly4Tri(J, nNodesSide);
    T_Kt = shapePoly4Tri(K, nNodesSide);

    // Multiply polynomials (order of multiplication does matter in this case, as the differentiation
    // with respect to t --third row-- is different to that with respec to r or s)
    temp = multPoly(T_Ir,T_Js);
    N_a = multPoly(temp,T_Kt);

    return N_a;


}


array<int> linkTriangleNodes(int in_n_spts)
{
    // first row in index_location_array contains indeces of r arranged in ascending global node number;
// second row contains indeces of s arranged in ascending global node number;
// third row contains indeces of t arranged in ascending global node number;
// refer to Hughes pp 169 to see link between r/s indeces ordering and global indeces ordering

    // Calculate number of nodes on each side
    int nNodesSide;
    nNodesSide =  calcNumSides(in_n_spts) ;
    array<int> index_location_array; // Global node-specific shape function to be returned

// Initialize arrays that will contain indices corresponding to node numbers
// Used temporarily to make code a bit clearer
    array<int> rind(1,in_n_spts); // stores r indeces; location in array is global node number
    array<int> sind(1,in_n_spts); // stores s indeces; location in array is global node number
    array<int> tind(1,in_n_spts); // stores t indeces; location in array is global node number
    array<int> temp; // temporary variable used while multiplying arrays

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

            if(nNodesLeft == 3) break; //skip while loop and process arrays
        }
        else // this means there is 1 node left
        {
            rind(lastInd) = cycle + 1;
            sind(lastInd) = cycle + 1;
            tind(lastInd) = cycle + 1;

            break; //skip while loop and process arrays
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
// Process arrays: assemble them by stacking them: use multPoly function
// Order matters: r indeces go first, s indeces second, t indeces third
    temp = multPoly(rind,sind);

    index_location_array = multPoly(temp,tind);

    return index_location_array;

}


array<double> diffPoly(array<double> & p, array<int> & term2Diff)
{
    /*
    Returns 3D array; differentiates polynomial p with respect to dimensions specified by term2Diff
    term2Diff: n x 1 array of rows to differentiate (negative integer i differentiates row abs(i) and multiplies that row by -1 ); rows enumerated starting at 1
    */

    int numTerms = term2Diff.get_dim(1); // find number of variables with respect to which to differentiate
    //cout<<"numTerms = "<<numTerms<<endl;
    int depthp = p.get_dim(2); // find number of layers of polynomial p (remember layers represent addition; rows multiplication; columns coefficients of polynomials)

    array<double> finalp(1); // this is the polynomial that will be returned
    finalp(0) = 0;  // initialize the polynomial

    // Declare variables used in the for loop
    int row, coeff;
    for(int l = 0; l < numTerms; l++)
    {
        // Create copy of p(:,:,1)
        array<double> diffp;
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


double evalPoly(array<double> p, array<double> coords)
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


void eval_dn_nodal_s_basis(array<double> &dd_nodal_s_basis,
                           array<double> in_loc, int in_n_spts, int n_deriv)
{
    /*
    Function that returns the values of the nth derivatives of the shape function
    of nodes (rows in dd_nodal_s_basis) with respect to r and s (dr^n, ds^2, drds are in 1st, 2nd, 3rd
    columns respectively) given:
    in_loc: 2x1 array of coordinates where derivatives are evaluated
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
    array<int> nodeList;
    nodeList = linkTriangleNodes(in_n_spts);

    //cout<<"Linked list: "<<endl;
    //nodeList.print();

    // Start loop to find derivatives at each node
    // Shape function N_a of node a
    array<double> N_a;
    // Differentiation of N_a
    array<double> diff_N_a;
    // Array that contains variables with respect to which shape function is differentiated
    array<int> diff_coord(1,2);
    // array that contains values of r,s,t to be plugged in when evaluating differentiated polynomial
    array<double> coords(1,3);
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

// Evaluate differentiated shape function and store in array d_nodal_s_basis

            dd_nodal_s_basis(a, (i==num_deriv_iterations-1 ? (i==0?0:1):(i==0? i:(i+1)))) = evalPoly(diff_N_a,coords);
        }


    }
}

//----------------------------------------------------------------------------
// Linear equation solution by Gauss-Jordan elimination.
// a(1:n,1:n) is the coefficients input matrix. 
// b(1:n) is the input matrix containing the right-hand side vector.
// On output, a(1:n,1:n) is replaced by its matrix inverse,
// and b(1:n) is replaced by the corresponding solution vector.
//
// From Numerical Recipes (http://www.nr.com/)
//----------------------------------------------------------------------------
void gaussj(int n, array<double>& A, array<double>& b)
{
	array<int> indxc(n),indxr(n),ipiv(n);
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

double flt_res(int N, array<double>& wf, array<double>& B, double k_0, double k_c, int ctype)
{
	int i;
	double norm, xm;
	array<double> flt(N);
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

// Set an array to zero
void zero_array(array <double>& in_array)
{
  int dim_1_0 = in_array.get_dim(0);
  int dim_1_1 = in_array.get_dim(1);
  int dim_1_2 = in_array.get_dim(2);
  int dim_1_3 = in_array.get_dim(3);

	for (int i=0;i<dim_1_0;++i) {
		for (int j=0;j<dim_1_1;++j) {
			for (int k=0;k<dim_1_2;++k) {
				for (int l=0;l<dim_1_3;++l) {
					in_array(i,j,k,l) = 0.0;
				}
			}
		}
	}
}

// Add arrays M1 and M2
array <double> add_arrays(array <double>& M1, array <double>& M2)
{
	// Get dimensions of arrays
  int dim_1_0 = M1.get_dim(0);
  int dim_1_1 = M1.get_dim(1);
  int dim_1_2 = M1.get_dim(2);
  int dim_1_3 = M1.get_dim(3);

  int dim_2_0 = M2.get_dim(0);
  int dim_2_1 = M2.get_dim(1);
  int dim_2_2 = M2.get_dim(2);
  int dim_2_3 = M2.get_dim(3);

	if(dim_1_0==dim_2_0 and dim_1_1==dim_2_1 and dim_1_2==dim_2_2 and dim_1_3==dim_2_3) {
		array <double> sum(dim_1_0,dim_1_1,dim_1_2,dim_1_3);
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
	else {
		cout << "ERROR: array dimensions are not compatible in sum function" << endl;
		exit(1);
	}
}

// Multiply M1(L*M) by M2(M*N)
array <double> mult_arrays(array <double>& M1, array <double>& M2)
{
	// Get dimensions of arrays
  int dim_1_0 = M1.get_dim(0);
  int dim_1_1 = M1.get_dim(1);
  int dim_1_2 = M1.get_dim(2);
  int dim_1_3 = M1.get_dim(3);

  int dim_2_0 = M2.get_dim(0);
  int dim_2_1 = M2.get_dim(1);
  int dim_2_2 = M2.get_dim(2);
  int dim_2_3 = M2.get_dim(3);

	// Only 2D arrays
	if(dim_1_2==1 and dim_1_3==1 and dim_2_2==1 and dim_2_3==1) {
		// Ensure consistent inner dimensions
		if(dim_1_1==dim_2_0) {
			array <double> product(dim_1_0,dim_2_1);

			#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  	
			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dim_1_0,dim_2_1,dim_1_1,1.0,M1.get_ptr_cpu(),dim_1_0,M2.get_ptr_cpu(),dim_2_0,0.0,product.get_ptr_cpu(),dim_1_0);

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
			cout << "ERROR: array dimensions are not compatible in multiplication function" << endl;
			exit(1);
		}
	}
	else {
		cout << "ERROR: Array multiplication function can only multiply 2-dimensional arrays together" << endl;
		exit(1);
	}
}

// method to get transpose of a square array
array <double> transpose_array(array <double>& in_array)
{
	// Get dimensions of arrays
  int dim_0 = in_array.get_dim(0);
  int dim_1 = in_array.get_dim(1);
  int dim_2 = in_array.get_dim(2);
  int dim_3 = in_array.get_dim(3);

	// Only 2D square arrays
	if(dim_2==1 and dim_3==1 and dim_0==dim_1) {
		int i,j;
		array <double> transpose(dim_1,dim_0);

		for(i=0;i<dim_0;i++) {
			for(j=0;j<dim_1;j++) {
				transpose(j,i)=in_array(i,j);
			}
		}
		return transpose;
	}
	else {
		cout << "ERROR: Array transpose function only accepts a 2-dimensional square array" << endl;
		exit(1);
	}
}

// method to get inverse of a square matrix

array <double> inv_array(array <double>& in_array)
{
  // Get dimensions of array
  int dim_0 = in_array.get_dim(0);
  int dim_1 = in_array.get_dim(1);
  int dim_2 = in_array.get_dim(2);
  int dim_3 = in_array.get_dim(3);

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
			array <double> atemp_0(dim_0);
			array <double> identity(dim_0,dim_0);
			array <double> input(dim_0,dim_0);
			array <double> inverse(dim_0,dim_0);
			array <double> inverse_out(dim_0,dim_0);
			array<int> swap_0(dim_0);
			array<int> swap_1(dim_0);

			// setup input array
			for(i=0;i<dim_0;i++)
				for(j=0;j<dim_0;j++)
					input(i,j) = in_array(i,j);

	 		// setup swap arrays
			for(i=0;i<dim_0;i++) {
				swap_0(i)=i;
				swap_1(i)=i;
			}

			// setup identity array		
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
		
				// swap the swap arrays
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
			cout << "ERROR: Can only obtain inverse of a square array" << endl;
			exit(1);
		}
	}
	else {
		cout << "ERROR: Array you are trying to invert has > 2 dimensions" << endl;
		exit(1);
	}
}

/*! END */

