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
#include <execinfo.h>
#include <unistd.h>
#include <iostream>
/********************************************************
 * Prints the error message, the stack trace, and exits
 * ******************************************************/
#define FatalError(s) {                                           \
  void* array[10];                                                  \
  size_t size;                                                      \
  size = backtrace(array,10);                                       \
  std::cout << "Fatal error at " << __FILE__ <<": " << __LINE__ \
  << ": " << s << std::endl;      \
  backtrace_symbols_fd(array, size, STDERR_FILENO);                 \
  exit(1); }

#define _(x) {   \
  std::cout << "At " << __FILE__ << ": " << __LINE__ \
  << ": " << #x << ": " << x << std::endl; }

//#define _MPI

//#define _CPU
