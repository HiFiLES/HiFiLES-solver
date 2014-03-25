/*!
 * \file flux.h
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

#include "array.h"

/*! calculate inviscid flux in 2D */
void calc_invf_2d(array<double>& in_u, array<double>& out_f);

/*! calculate inviscid flux in 3D */
void calc_invf_3d(array<double>& in_u, array<double>& out_f);

/*! calculate viscous flux in 2D */
void calc_visf_2d(array<double>& in_u, array<double>& in_grad_u, array<double>& out_f);

/*! calculate viscous flux in 3D */
void calc_visf_3d(array<double>& in_u, array<double>& in_grad_u, array<double>& out_f);

/*!
 * \brief calculate & add addtional ALE flux term in 2D
 * \param[in] in_u - Solution vector
 * \param[in] in_w - Grid velocity
 * \param[in,out] out_f - Modified flux vector
 */
void calc_alef_2d(array<double>& in_u, array<double>& in_v, array<double>& out_f);

/*!
 * \brief calculate & add addtional ALE flux term in 3D
 * \param[in] in_u - Solution vector
 * \param[in] in_w - Grid velocity
 * \param[in,out] out_f - Modified flux vector
 */
void calc_alef_3d(array<double>& in_u, array<double>& in_v, array<double>& out_f);
