/*!
 * \file cubature_hexa.h
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

#include <string>
#include "array.h"

class cubature_hexa
{
	public:

	// #### constructors ####

	// default constructor
	cubature_hexa();

	// constructor 1
	cubature_hexa(int in_rule); // set by order
	
	// copy constructor
	cubature_hexa(const cubature_hexa& in_cubature);
	
	// assignment
	cubature_hexa& operator=(const cubature_hexa& in_cubature);
	
	// destructor
	~cubature_hexa();

	// #### methods ####
	
	// method to initialize a cubature rule
	void set_rule(int in_rule);
	
	// method to get number of cubature points
	double get_n_pts(void);

	// method to get r location of cubature point
	double get_r(int in_pos);

	// method to get s location of cubature point
	double get_s(int in_pos);

	// method to get s location of cubature point
	double get_t(int in_pos);

	// method to get weight location of cubature point
	double get_weight(int in_pos);

	// #### members ####

	// cubature rule
	int rule;
	
	// number of cubature points
	int n_pts;

	// location of cubature points
	array<double> locs;
	
	// weight of cubature points
	array<double> weights;
};
