/*!
 * \file Array.h
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 * \version 0.1.0
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 * Copyright (C) 2014 Aerospace Computing Laboratory (ACL).
 *
 * HiFiLES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HiFiLES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HiFiLES.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "error.h"
#include "funcs.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <typeinfo>
#include <vector>

#ifdef _GPU
#include "cuda.h"
#include "cuda_runtime_api.h"
#endif

template <typename T>
class Array
{
public:

  // #### constructors ####

  // default constructor

  Array();

  // constructor 1

  Array(int in_dim_0, int in_dim_1=1, int in_dim_2=1, int in_dim_3=1);

  // copy constructor

  Array(const Array<T>& in_Array);

  // assignment

  Array<T>& operator=(const Array<T>& in_Array);

  // destructor

  ~Array();

  // #### methods ####

#ifdef _GPU
  void check_cuda_error(const char *message, const char *filename, const int lineno);
#endif

  // setup

  void setup(int in_dim_0, int in_dim_1=1, int in_dim_2=1, int in_dim_3=1);

// access data
  T& operator() (int in_pos_0, int in_pos_1 = 0, int in_pos_2 = 0, int in_pos_3 = 0);
  T& operator() (int in_pos_0, int in_pos_1 = 0, int in_pos_2 = 0, int in_pos_3 = 0) const;

  // return pointer

  T* get_ptr_cpu(void);
  T* get_ptr_gpu(void);

  // return pointer

  T* get_ptr_cpu(int in_pos_0, int in_pos_1=0, int in_pos_2=0, int in_pos_3=0);
  T* get_ptr_gpu(int in_pos_0, int in_pos_1=0, int in_pos_2=0, int in_pos_3=0);

  // return dimension

  int get_dim(int in_dim);

  // return number of elements
  int size() const;

  // method to get maximum value of Array

  T get_max(void);

  // method to get minimum value of Array

  T get_min(void);

  // print
  void print(void);

  template <typename R>
  friend std::ostream& operator<<(std::ostream& out, Array<R>& array);

  template <typename R>
  friend std::istream& operator>>(std::istream& in, Array<R>& array);

  template <typename R>
  friend void toBinary(Array<R>* array, std::ofstream& file);

  template <typename R>
  friend void fromBinary(Array<R>* array, std::ifstream& file);

  // move data from cpu to gpu

  void mv_cpu_gpu(void);

  // copy data from cpu to gpu

  void cp_cpu_gpu(void);

  // move data from gpu to cpu

  void mv_gpu_cpu(void);

  // copy data from gpu to cpu

  void cp_gpu_cpu(void);

  // remove data from cpu

  void rm_cpu(void);

  /*! Initialize Array to zero - Valid for numeric data types (int, float, double) */
  void initialize_to_zero();

  /*! Initialize Array to given value */
  void fill(const T val);

  /*! Write array contents to a file */
  void writeToFile(std::string fileName, bool inBinary = true);

  /*! Read array contents froma a file */
  void initFromFile(std::string fileName);

  /*! Write array in binary format */
  template <typename R>
  friend void toBinary(Array<R>* array, std::ofstream& file);

  /*! Read an array in binary format */
  template <typename R>
  friend void fromBinary(Array<R>* array, std::ifstream& file);

  /*! Normalize array rows */
  void normalizeRows();

  /*! BLAS wrappers */

  /*! dgemm performs the operation: this = alpha * A * B + beta * this */
  /*! where A, B are 2D matrices, and alpha, beta are scalars */
  void dgemm(double alpha, Array<T>& A, Array<T>& B, double beta);

  /*! daxpy performs the operation: this = alpha * x + y */
  /*! where x and y are vectors of the same length and alpha is a scalar */
  void daxpy(double alpha, Array<T>& x);

protected:

  int dim_0;
  int dim_1;
  int dim_2;
  int dim_3;

  T* cpu_data;
  T* gpu_data;

  int cpu_flag;
  int gpu_flag;

};

/*! Write array in binary format */
template <typename T>
void toBinary(Array<T>* array, std::ofstream& file);

/*! Read an array in binary format */
template <typename T>
void fromBinary(Array<T>* array, std::ifstream& file);


void fromBinary(double* number, std::ifstream& file);
void toBinary(double* number, std::ofstream& file);
#include "../src/Array.cpp" // include class implementation because it is templated
