/*!
 * \file eles.cpp
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

#ifndef INC_SOURCE_H
#define INC_SOURCE_H

#include "array.h"

/*! calculate source term for Spalart-Allmaras turbulence model in 2D */
void calc_source_SA_2d(array<double>& in_u, array<double>& in_grad_u, double& d, double& out_source);

/*! calculate source term for Spalart-Allmaras turbulence model in 3D */
void calc_source_SA_3d(array<double>& in_u, array<double>& in_grad_u, double& d, double& out_source);

#endif
