/*!
 * \file matrix.h
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

#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>

class matrix
{	
	public:
	
	// #### constructors ####	
		
	// default constructor
	matrix();
	
	// constructor 1
	matrix(int in_dim_0, int in_dim_1=1);
	
	// copy constructor
	matrix(const matrix& in_matrix);
	
	// assignment
	matrix& operator=(const matrix& in_matrix);
	
	// destructor
	~matrix();
	
	// #### methods ####
	
	// method to setup 
	void setup(int in_dim_0, int in_dim_1=1);
	
	// method to access
	const double& operator() (int in_pos_0, int in_pos_1=0) const;
	
	// method to access/set
	double& operator() (int in_pos_0, int in_pos_1=0);
	
	// method to multiply
	matrix operator*(const matrix& in_matrix) const;

  matrix operator+(const matrix& in_matrix) const;

	// method to get transpose
	matrix get_trans(void);
	
	// method to get inverse
	matrix get_inv(void);
	
	// method to get direct sum
	matrix get_directsum(const matrix& in_matrix);
	
	// method to get dim_0
	const int& get_dim_0(void) const;
	
	// method to get dim_1
	const int& get_dim_1(void) const;
	
	// method to print
	void print(void);
	
	protected:
		
	int dim_0;
	int dim_1;
	
	double* data;
};
