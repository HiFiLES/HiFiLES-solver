/*!
 * \file cubature_tri.h
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

class cubature_tri
{
	public:

	// #### constructors ####

	// default constructor
	cubature_tri();

	// constructor 1
	cubature_tri(int in_order); // set by order
	
	// copy constructor
	cubature_tri(const cubature_tri& in_cubature_tri);
	
	// assignment
	cubature_tri& operator=(const cubature_tri& in_cubature_tri);
	
	// destructor
	~cubature_tri();

	// #### methods ####
	
	// method to initialize a cubature_tri rule
	void set_order(int in_order);
	
	// method to get number of cubature_tri points
	double get_n_pts(void);

	// method to get r location of cubature_tri point
	double get_r(int in_pos);

	// method to get s location of cubature_tri point
	double get_s(int in_pos);

	// method to get weight location of cubature_tri point
	double get_weight(int in_pos);

	// #### members ####

	// cubature_tri order
	int order;
	
	// number of cubature_tri points
	int n_pts;

	// location of cubature_tri points
	array<double> locs;
	
	// weight of cubature_tri points
	array<double> weights;
};
