/*!
 * \file cubature_1d.h
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

class cubature_1d
{
public:

  // #### constructors ####

  // default constructor
  cubature_1d();

  // constructor 1
  cubature_1d(int in_order); // set by order

  // copy constructor
  cubature_1d(const cubature_1d& in_cubature_1d);

  // assignment
  cubature_1d& operator=(const cubature_1d& in_cubature_1d);

  // destructor
  ~cubature_1d();

  // #### methods ####

  // method to get number of cubature_1d points
  int get_n_pts(void);

  // method to get r location of cubature_1d point
  double get_r(int in_pos);

  // method to get weight location of cubature_1d point
  double get_weight(int in_pos);

  // #### members ####

  // cubature_1d order
  int order;

  // number of cubature_1d points
  int n_pts;

  // location of cubature_1d points
  array<double> locs;

  // weight of cubature_1d points
  array<double> weights;
};
