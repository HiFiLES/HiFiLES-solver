/*!
 * \file matrix.cpp
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

#include <iostream>
#include <iomanip>
#include "../include/array.h"
#include "../include/matrix.h"

using namespace std;

// #### constructors ####

// default constructor

matrix::matrix()
{	
	dim_0=1;
	dim_1=1;
	
	data = new double[dim_0*dim_1];	
}

// constructor 1

matrix::matrix(int in_dim_0, int in_dim_1)
{	
	dim_0=in_dim_0;
	dim_1=in_dim_1;
	
	data = new double[dim_0*dim_1];		
}

// copy constructor

matrix::matrix(const matrix& in_matrix)
{	
	int i;
	
	dim_0=in_matrix.dim_0;
	dim_1=in_matrix.dim_1;
	
	data = new double[dim_0*dim_1];
	
	for(i=0;i<dim_0*dim_1;i++)
	{
		data[i]=in_matrix.data[i];
	}		
}	

// assignment

matrix& matrix::operator=(const matrix& in_matrix)
{	
	int i;

	if(this == &in_matrix)
	{
		return (*this);
	}
	else
	{
		delete[] data;
	
		dim_0=in_matrix.dim_0;
		dim_1=in_matrix.dim_1;
		
		data = new double[dim_0*dim_1];
	
		for(i=0;i<dim_0*dim_1;i++)
		{
			data[i]=in_matrix.data[i];		
		}		
				
		return (*this);				
	}	
}

// destructor

matrix::~matrix()
{	
	delete[] data;
}

// #### methods ####

// method to setup

void matrix::setup(int in_dim_0, int in_dim_1)
{
	int i;
		
	delete[] data;
	
	dim_0=in_dim_0;
	dim_1=in_dim_1;
	
	data = new double[dim_0*dim_1];
	for (i=0;i<dim_0*dim_1;i++)
	{
		data[i] = 0.;
	}
}

// method to access

const double& matrix::operator() (int in_pos_0, int in_pos_1) const
{
	return data[in_pos_0+(dim_0*in_pos_1)];	
}

// method to access/set

double& matrix::operator() (int in_pos_0, int in_pos_1)
{
	return data[in_pos_0+(dim_0*in_pos_1)];	
}

// method to multiply

matrix matrix::operator*(const matrix& in_matrix) const
{
	if(dim_1==in_matrix.dim_0)
	{
		int i,j,k;
	
		matrix atemp_0(dim_0,in_matrix.dim_1);
	
		double dtemp_0;
			
		// not to be used where speed is paramount
			
		for(i=0;i<dim_0;i++)
		{
			for(j=0;j<in_matrix.dim_1;j++)
			{
				dtemp_0=0.0;
					
				for(k=0;k<dim_1;k++)
				{
					dtemp_0=dtemp_0+(*this)(i,k)*in_matrix(k,j);
				}
			
				atemp_0(i,j)=dtemp_0;
			}
		}
			
		return atemp_0;	
	}
	else
	{
		cout << "ERROR: Invalid array dimensions for multiplication ...." << endl;
	}
}

// method to add

matrix matrix::operator+(const matrix& in_matrix) const
{
	if(dim_0==in_matrix.dim_0 && dim_1 ==in_matrix.dim_1)
	{
		int i,j,k;
		matrix atemp_0(dim_0,in_matrix.dim_1);
	
		// not to be used where speed is paramount
			
		for(i=0;i<dim_0;i++)
		{
			for(j=0;j<dim_1;j++)
			{
        atemp_0(i,j) = (*this)(i,j)+in_matrix(i,j);
			}
		}
			
		return atemp_0;	
	}
	else
	{
		cout << "ERROR: Invalid array dimensions for addition...." << endl;
	}
}




// method to get transpose

matrix matrix::get_trans(void)
{
	int i,j;
		
	matrix atemp_0(dim_1,dim_0);
		
	// not to be used where speed is paramount
		
	for(i=0;i<dim_0;i++)
	{
		for(j=0;j<dim_1;j++)
		{
			atemp_0(j,i)=(*this)(i,j);
		}
	}
		
	return atemp_0;
}

// method to get inverse

matrix matrix::get_inv(void) // TODO: Tidy up matrix inverse routine
{
	if(dim_0==dim_1)
	{
		// gausian elimination with full pivoting
		// not to be used where speed is paramount
			
		int i,j,k;
	
		double mag;
		double max;
	
		int pivot_i, pivot_j;
	
		double dtemp_0;
		
		int itemp_0;
	
		double first;
		
		matrix atemp_0(dim_0);
		matrix input, identity(dim_0,dim_0), inverse(dim_0,dim_0), inverse_out(dim_0,dim_0);
				
		array<int> swap_0(dim_0);
		array<int> swap_1(dim_0);
	
	 	// setup swap arrays
	 	
		for(i=0;i<dim_0;i++)
		{
			swap_0(i)=i;
			swap_1(i)=i;
		}
	
		// setup identity array
		
		for(i=0;i<dim_0;i++)
		{
			for(j=0;j<dim_0;j++)
			{
				if(i==j)
				{
					identity(i,j)=1.0;
				}
				else
				{
					identity(i,j)=0.0;
				}
			}
		}
	
		// setup input array
		
		input=(*this);
	
		// make triangular
	
		for(k=0;k<dim_0-1;k++)
		{
			max=0;
		
			// find pivot
			
			for(i=k;i<dim_0;i++)
			{
				for(j=k;j<dim_0;j++)
				{
					mag=input(i,j)*input(i,j);
			
					if(mag>max)
					{
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
		
			for(i=0;i<dim_0;i++) 
			{
				atemp_0(i)=input(i,pivot_j);
				input(i,pivot_j)=input(i,k);
				input(i,k)=atemp_0(i);
			}	
			
			// swap the rows
		
			for(j=0;j<dim_0;j++)
			{
				atemp_0(j)=input(pivot_i,j);
				input(pivot_i,j)=input(k,j);
				input(k,j)=atemp_0(j);
			
				atemp_0(j)=identity(pivot_i,j); 
				identity(pivot_i,j)=identity(k,j); 
				identity(k,j)=atemp_0(j);
			}
		
			// subtraction
		
			for(i=k+1;i<dim_0;i++)
			{
				first=input(i,k);
			
				for(j=0;j<dim_0;j++)
				{
					if(j>=k)
					{
						input(i,j)=input(i,j)-((first/input(k,k))*input(k,j));
					}
				
					identity(i,j)=identity(i,j)-((first/input(k,k))*identity(k,j));
				}
			}
		
			//exact zero
			
			for(j=0;j<k+1;j++)
			{
				for(i=j+1;i<dim_0;i++)
				{
					input(i,j)=0.0;
				}
			}
		}
	
		// back substitute
	
		for(i=dim_0-1;i>=0;i=i-1)
		{
			for(j=0;j<dim_0;j++)
			{
				dtemp_0=0.0;
		
				for(k=i+1;k<dim_0;k++)
				{
					dtemp_0=dtemp_0+(input(i,k)*inverse(k,j));
				}
		
				inverse(i,j)=(identity(i,j)-dtemp_0)/input(i,i);			
			}
		}
	
		// swap solution rows

		for(i=0;i<dim_0;i++)
		{
			for(j=0;j<dim_0;j++)
			{	
		
				inverse_out(swap_1(i),j)=inverse(i,j);
			}
		}
	
		return inverse_out;
	}
	else
	{
		cout << "ERROR: Can only obtain inverse of a square matrix ...." << endl;
	}
}

// method to get direct sum

matrix matrix::get_directsum(const matrix& in_matrix)
{
	int i,j;
	
	matrix directsum(dim_0+in_matrix.dim_0,dim_1+in_matrix.dim_1);
	
	for(i=0;i<dim_0+in_matrix.dim_0;i++)
	{
		for(j=0;j<dim_1+in_matrix.dim_1;j++)
		{
			if(i<dim_0&&j<dim_1)
			{
				directsum(i,j)=(*this)(i,j);
			}
			else if(i>=dim_0&&j>=dim_1)
			{
				directsum(i,j)=in_matrix(i-dim_0,j-dim_1);
			}
			else
			{
				directsum(i,j)=0.0;
			}
		}
	}
	
	return directsum;
}

// method to get dim_0

const int& matrix::get_dim_0(void) const
{
	return dim_0;
}
	
// method to get dim_1

const int& matrix::get_dim_1(void) const
{
	return dim_1;	
}

// method to print

void matrix::print(void)
{
	int i,j;
		
	for(i=0;i<dim_0;i++)
	{
		for(j=0;j<dim_1;j++)
		{		
			if((*this)(i,j)*(*this)(i,j)<1e-16) // print 0 if less than tolerance
			{
				cout << " " << "0.000000" << " ";
			}
			else
			{
				cout << " " << setprecision(6) << (*this)(i,j) << " ";
			}
		}
			
		cout << endl;
	}	
}
