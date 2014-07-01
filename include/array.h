/*!
 * \file array.h
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include "error.h"

#ifdef _GPU
#include "cuda.h"
#include "cuda_runtime_api.h"
#endif


template <typename T>
class array
{
public:

  // #### constructors ####

  // default constructor

  array();

  // constructor 1

  array(int in_dim_0, int in_dim_1=1, int in_dim_2=1, int in_dim_3=1);

  // copy constructor

  array(const array<T>& in_array);

  // assignment

  array<T>& operator=(const array<T>& in_array);

  // destructor

  ~array();

  // #### methods ####

#ifdef _GPU
  void check_cuda_error(const char *message, const char *filename, const int lineno);
#endif

  // setup

  void setup(int in_dim_0, int in_dim_1=1, int in_dim_2=1, int in_dim_3=1);

  // access/set 1d

  T& operator() (int in_pos_0);

  // access/set 2d

  T& operator() (int in_pos_0, int in_pos_1);

  // access/set 3d

  T& operator() (int in_pos_0, int in_pos_1, int in_pos_2);

  // access/set 4d

  T& operator() (int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3);

  // return pointer

  T* get_ptr_cpu(void);
  T* get_ptr_gpu(void);

  // return pointer

  T* get_ptr_cpu(int in_pos_0, int in_pos_1=0, int in_pos_2=0, int in_pos_3=0);
  T* get_ptr_gpu(int in_pos_0, int in_pos_1=0, int in_pos_2=0, int in_pos_3=0);

  // return dimension

  int get_dim(int in_dim);

  // method to get maximum value of array

  T get_max(void);

  // method to get minimum value of array

  T get_min(void);

  // print

  void print(void);

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

  /*! Initialize array to zero - Valid for numeric data types (int, float, double) */
  void initialize_to_zero();

  /*! Initialize array to given value */
  void initialize_to_value(const T val);

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

// definitions

#include <iostream>

using namespace std;

// #### constructors ####

// default constructor

template <typename T>
array<T>::array()
{
  dim_0=1;
  dim_1=1;
  dim_2=1;
  dim_3=1;

  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];

  cpu_flag=1;
  gpu_flag=0;
}

// constructor 1

template <typename T>
array<T>::array(int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3)
{
  dim_0=in_dim_0;
  dim_1=in_dim_1;
  dim_2=in_dim_2;
  dim_3=in_dim_3;

  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];


  cpu_flag=1;
  gpu_flag=0;
}

// copy constructor

template <typename T>
array<T>::array(const array<T>& in_array)
{
  int i;

  dim_0=in_array.dim_0;
  dim_1=in_array.dim_1;
  dim_2=in_array.dim_2;
  dim_3=in_array.dim_3;

  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];

  for(i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
    {
      cpu_data[i]=in_array.cpu_data[i];
    }
}

// assignment

template <typename T>
array<T>& array<T>::operator=(const array<T>& in_array)
{
  int i;

  if(this == &in_array)
    {
      return (*this);
    }
  else
    {
      delete[] cpu_data;

      dim_0=in_array.dim_0;
      dim_1=in_array.dim_1;
      dim_2=in_array.dim_2;
      dim_3=in_array.dim_3;

      cpu_data = new T[dim_0*dim_1*dim_2*dim_3];
      //NOTE: THIS COPIES POINTERS; NOT VALUES
      for(i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
        {
          cpu_data[i]=in_array.cpu_data[i];
        }

      cpu_flag=1;
      gpu_flag=0;

      return (*this);
    }
}

// destructor

template <typename T>
array<T>::~array()
{
  delete[] cpu_data;
  // do we need to deallocate gpu memory here as well?
}

// #### methods ####

// setup

template <typename T>
void array<T>::setup(int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3)
{
  delete[] cpu_data;

  dim_0=in_dim_0;
  dim_1=in_dim_1;
  dim_2=in_dim_2;
  dim_3=in_dim_3;

  cpu_data=new T[dim_0*dim_1*dim_2*dim_3];
  cpu_flag=1;
  gpu_flag=0;
}

template <typename T>
T& array<T>::operator()(int in_pos_0)
{
  return cpu_data[in_pos_0]; // column major with matrix indexing
}

template <typename T>
T& array<T>::operator()(int in_pos_0, int in_pos_1)
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)]; // column major with matrix indexing
}

template <typename T>
T& array<T>::operator()(int in_pos_0, int in_pos_1, int in_pos_2)
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)]; // column major with matrix indexing
}

template <typename T>
T& array<T>::operator()(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3)]; // column major with matrix indexing
}

// return pointer

template <typename T>
T* array<T>::get_ptr_cpu(void)
{
  if(cpu_flag==1)
    return cpu_data;
  else
    FatalError("CPU array does not exist");
}


// return pointer

template <typename T>
T* array<T>::get_ptr_gpu(void)
{
  if(gpu_flag==1)
    return gpu_data;
  else
    {
      cout << "dim_0=" << dim_0 << " dim_1=" << dim_1 << " dim_2=" << dim_2 << endl;
      FatalError("GPU array does not exist");
    }
}



// return pointer

template <typename T>
T* array<T>::get_ptr_cpu(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
{
  if(cpu_flag==1)
    return cpu_data+in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3); // column major with matrix indexing
  else
    FatalError("Cpu data does not exist");
}


template <typename T>
T* array<T>::get_ptr_gpu(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
{
  if(gpu_flag==1)
    return gpu_data+in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3); // column major with matrix indexing
  else
    FatalError("GPU data does not exist, get ptr");
}


// obtain dimension

template <typename T>
int array<T>::get_dim(int in_dim)
{
  if(in_dim==0)
    {
      return dim_0;
    }
  else if(in_dim==1)
    {
      return dim_1;
    }
  else if(in_dim==2)
    {
      return dim_2;
    }
  else if(in_dim==3)
    {
      return dim_3;
    }
  else
    {
      cout << "ERROR: Invalid dimension ... " << endl;
      return 0;
    }
}


// method to calculate maximum value of array
// Template specialization
template <typename T>
T array<T>::get_max(void)
{
  int i;
  T max = 0;

  for(i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
    {
      if( ((*this).get_ptr_cpu())[i] > max)
        max = ((*this).get_ptr_cpu())[i];
    }
  return max;
}

// method to calculate minimum value of array
// Template specialization
template <typename T>
T array<T>::get_min(void)
{
  int i;
  T min = 1e16;

  for(i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
    {
      if( ((*this).get_ptr_cpu())[i] < min)
        min = ((*this).get_ptr_cpu())[i];
    }
  return min;
}
// print

template <typename T>
void array<T>::print(void)
{
  if(dim_3==1)
    {
      int i,j,k;
      bool threeD = (dim_2==1?false:true);
      for (k = 0; k< dim_2; k++)
        {
          if (threeD)
            cout<<endl<<"ans(:,:,"<<k+1<<") = "<<endl;
          for(i=0; i<dim_0; i++)
            {
              for(j=0; j<dim_1; j++)
                {

                  if((*this)(i,j,k)*(*this)(i,j,k)<1e-12)
                    {
                      cout << " 0 ";
                    }
                  else
                    {
                      cout << " " << (*this)(i,j,k) << " ";
                    }
                }

              cout << endl;
            }
          if (threeD)
            cout<<endl;
        }
    }
  else
    {
      cout << "ERROR: Can only print an array of dimension three or less ...." << endl;
    }
}

#ifdef _GPU
template <typename T>
void array<T>::check_cuda_error(const char *message, const char *filename, const int lineno)
{
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
    {
      printf("CUDA error after %s at %s:%d: %s\n", message, filename, lineno, cudaGetErrorString(error));
      exit(-1);
    }
}
#endif

// move data from cpu to gpu

template <typename T>
void array<T>::mv_cpu_gpu(void)
{
#ifdef _GPU

  if (cpu_flag==0)
    FatalError("CPU data does not exist");

  check_cuda_error("Before",__FILE__,__LINE__);

  // free gpu pointer first?
  cudaMalloc((void**) &gpu_data,dim_0*dim_1*dim_2*dim_3*sizeof(T));
  cudaMemcpy(gpu_data,cpu_data,dim_0*dim_1*dim_2*dim_3*sizeof(T),cudaMemcpyHostToDevice);

  delete[] cpu_data;
  cpu_data = new T[1];

  cpu_flag=0;
  gpu_flag=1;

  check_cuda_error("After Memcpy, asking for too much memory?",__FILE__,__LINE__);

#endif
}

// move data from gpu to cpu

template <typename T>
void array<T>::mv_gpu_cpu(void)
{
#ifdef _GPU

  check_cuda_error("mv_gpu_cpu before",__FILE__, __LINE__);
  delete[] cpu_data;
  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];

  cudaMemcpy(cpu_data,gpu_data,dim_0*dim_1*dim_2*dim_3*sizeof(T),cudaMemcpyDeviceToHost);
  cudaFree(gpu_data);
  // assign gpu pointer unit size afterwards?

  cpu_flag=1;
  gpu_flag=0;

  check_cuda_error("mv_gpu_cpu after",__FILE__, __LINE__);
#endif
}

// copy data from gpu to cpu

template <typename T>
void array<T>::cp_gpu_cpu(void)
{
#ifdef _GPU

  //delete[] cpu_data;
  //cpu_data = new T[dim_0*dim_1*dim_2*dim_3];

  if (gpu_flag==0)
    FatalError("GPU data does not exist");

  if (cpu_flag==0)
    {
      cpu_data = new T[dim_0*dim_1*dim_2*dim_3];
      cpu_flag=1;
    }

  check_cuda_error("cp_gpu_cpu before",__FILE__, __LINE__);
  cudaMemcpy(cpu_data,gpu_data,dim_0*dim_1*dim_2*dim_3*sizeof(T),cudaMemcpyDeviceToHost);
  check_cuda_error("cp_gpu_cpu after",__FILE__, __LINE__);

#endif
}

// copy data from cpu to gpu

template <typename T>
void array<T>::cp_cpu_gpu(void)
{
#ifdef _GPU

  if (cpu_flag==0)
    FatalError("Cpu data does not exist");

  check_cuda_error("cp_cpu_gpu before",__FILE__, __LINE__);
  if (gpu_flag==0)
    {
      cudaMalloc((void**) &gpu_data,dim_0*dim_1*dim_2*dim_3*sizeof(T));
      gpu_flag=1;
    }
  cudaMemcpy(gpu_data,cpu_data,dim_0*dim_1*dim_2*dim_3*sizeof(T),cudaMemcpyHostToDevice);

  check_cuda_error("cp_cpu_gpu after",__FILE__, __LINE__);

#endif
}

// remove data from cpu

template <typename T>
void array<T>::rm_cpu(void)
{
#ifdef _GPU

  check_cuda_error("rm_cpu before",__FILE__, __LINE__);
  delete[] cpu_data;
  cpu_data = new T[1];

  cpu_flag=0;
  check_cuda_error("rm_cpu after",__FILE__, __LINE__);

#endif
}

// Initialize values to zero (for numeric data types)
template <typename T>
void array<T>::initialize_to_zero()
{

  for(int i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
    {
      cpu_data[i]=0;
    }

}

// Initialize array to given value
template <typename T>
void array<T>::initialize_to_value(const T val)
{
  for(int i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
  {
    cpu_data[i]=val;
  }
}
