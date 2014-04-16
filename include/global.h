/*!
 * \file global.h
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

#include "input.h"

/*! input 'run_input' has global scope */
extern input run_input;

/*! double 'pi' has global scope */
extern double pi;

/*! environment variable specifying location of HiFiLES repository */
extern const char* HIFILES_DIR;