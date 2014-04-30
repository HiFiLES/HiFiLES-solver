/*!
 * \file cubature_tet.h
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

class cubature_tet
{
public:

  // #### constructors ####

  // default constructor
  cubature_tet();

  // constructor 1
  cubature_tet(int in_rule); // set by order

  // copy constructor
  cubature_tet(const cubature_tet& in_cubature);

  // assignment
  cubature_tet& operator=(const cubature_tet& in_cubature);

  // destructor
  ~cubature_tet();

  // #### methods ####

  // method to get number of cubature points
  int get_n_pts(void);

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
  
  // cubature data file
  ifstream datfile;
};
