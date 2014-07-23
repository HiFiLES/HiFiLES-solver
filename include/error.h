/*!
 * \file error.h
 * \brief _____________________________
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL) directed
 *                                by Prof. Jameson. (Aero/Astro Department. Stanford University.
 * \version 1.0.0
 *
 * HiFiLES (High Fidelity Large Eddy Simulation).
 * Copyright (C) 2013 Aerospace Computing Laboratory.
 */

#pragma once

#include <stdio.h>
/********************************************************
 * Prints the error message, the stack trace, and exits
 * ******************************************************/
#define FatalError(s) {                                             \
  printf("Fatal error '%s' at %s:%d\n",s,__FILE__,__LINE__);        \
  exit(1); }
