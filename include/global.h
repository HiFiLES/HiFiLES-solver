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

/** enumeration for cell type */
enum CTYPE {
    TRI   = 0,
    QUAD  = 1,
    TET   = 2,
    WEDGE = 3,
    BRICK = 4
};

/** enumeration for mesh motion type */
enum {MOTION_DISABLED, MOTION_ENABLED};
