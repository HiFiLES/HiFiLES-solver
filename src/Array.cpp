#ifndef ARRAY_CPP_
#define ARRAY_CPP_


#include "../include/Array.h"
#include <stdexcept>
// #### constructors ####

// default constructor

template <typename T>
Array<T>::Array()
{
  dim_0=1;
  dim_1=1;
  dim_2=1;
  dim_3=1;

  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];

  cpu_flag=1;
  gpu_flag=0;

  gpu_data = NULL;
}

// constructor 1

template <typename T>
Array<T>::Array(int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3)
{
  dim_0=in_dim_0;
  dim_1=in_dim_1;
  dim_2=in_dim_2;
  dim_3=in_dim_3;

  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];


  cpu_flag=1;
  gpu_flag=0;

  gpu_data = NULL;
}
//*/
// copy constructor

template <typename T>
Array<T>::Array(const Array<T>& in_Array)
{
  dim_0=in_Array.dim_0;
  dim_1=in_Array.dim_1;
  dim_2=in_Array.dim_2;
  dim_3=in_Array.dim_3;

  cpu_data = new T[in_Array.size()];

  for(int i = 0; i < in_Array.size(); i++)
    {
      (*this)(i)=in_Array(i);
    }

  cpu_flag=1;
  gpu_flag=0;

  gpu_data = NULL;
}

// assignment

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& in_Array)
{
  this->setup(in_Array.dim_0, in_Array.dim_1, in_Array.dim_2, in_Array.dim_3);

  for(int i = 0; i < in_Array.size(); i++)
    {
      (*this)(i)=in_Array(i);
    }

  return (*this);

}

// destructor

template <typename T>
Array<T>::~Array()
{
  delete[] cpu_data;
  // do we need to deallocate gpu memory here as well?
}

// #### methods ####

// setup

template <typename T>
void Array<T>::setup(int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3)
{
  delete[] cpu_data;

  dim_0=in_dim_0;
  dim_1=in_dim_1;
  dim_2=in_dim_2;
  dim_3=in_dim_3;

  cpu_data=new T[size()];
  cpu_flag=1;
  gpu_flag=0;
}

template <typename T>
T& Array<T>::operator()(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3)]; // column major with matrix indexing
}

template <typename T>
T& Array<T>::operator()(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3) const
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3)]; // column major with matrix indexing
}

// return pointer

template <typename T>
T* Array<T>::get_ptr_cpu(void)
{
  if(cpu_flag==1)
    return cpu_data;
  else
    FatalError("CPU Array does not exist");
}


// return pointer

template <typename T>
T* Array<T>::get_ptr_gpu(void)
{
  if(gpu_flag==1)
    return gpu_data;
  else
    {
      std::cout << "dim_0=" << dim_0 << " dim_1=" << dim_1 << " dim_2=" << dim_2 << std::endl;
      FatalError("GPU Array does not exist");
    }
}


// return pointer

template <typename T>
T* Array<T>::get_ptr_cpu(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
{
  if(cpu_flag==1)
    return cpu_data+in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3); // column major with matrix indexing
  else
    FatalError("Cpu data does not exist");
}


template <typename T>
T* Array<T>::get_ptr_gpu(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
{
  if(gpu_flag==1)
    return gpu_data+in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)+(dim_0*dim_1*dim_2*in_pos_3); // column major with matrix indexing
  else
    FatalError("GPU data does not exist, get ptr");
}


// obtain dimension
template <typename T>
int Array<T>::get_dim(int in_dim)
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
      std::cout << "ERROR: Invalid dimension ... " << std::endl;
      return 0;
    }
}

// get number of entries in array
template <typename T>
int Array<T>::size() const
{
  return dim_0 * dim_1 * dim_2 * dim_3;
}


// method to calculate maximum value of Array
// Template specialization
template <typename T>
T Array<T>::get_max(void)
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

// method to calculate minimum value of Array
// Template specialization
template <typename T>
T Array<T>::get_min(void)
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
void Array<T>::print(void)
{
  std::cout << *this;
}

#ifdef _GPU
template <typename T>
void Array<T>::check_cuda_error(const char *message, const char *filename, const int lineno)
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
void Array<T>::mv_cpu_gpu(void)
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
void Array<T>::mv_gpu_cpu(void)
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
void Array<T>::cp_gpu_cpu(void)
{
#ifdef _GPU

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
void Array<T>::cp_cpu_gpu(void)
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
void Array<T>::rm_cpu(void)
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
void Array<T>::initialize_to_zero()
{

  for(int i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
    {
      cpu_data[i]=0;
    }

}

// Initialize Array to given value
template <typename T>
void Array<T>::fill(const T val)
{
  for(int i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
  {
    cpu_data[i]=val;
  }
}

// returns an ostream object with the contents of the array
template <typename T>
std::ostream& operator<<(std::ostream& out, Array<T>& array) {

#ifdef _GPU
  // We need to pass the gpu data back to the cpu to output it
  if (array.gpu_flag == 1) {
      array.cp_gpu_cpu();
    }
#endif

  out << array.dim_0 << " " << array.dim_1 << " "
      << array.dim_2 << " " << array.dim_3
      << std::endl;
  for (int l = 0; l < array.dim_3; l++)
    {
      for (int k = 0; k< array.dim_2; k++)
        {
          for(int i = 0; i < array.dim_0; i++)
            {
              for(int j=0; j < array.dim_1; j++)
                {
                  out << std::left << std::setw(50) << std::setprecision(45)
                      << array(i,j,k,l);
                }
              out << std::endl;
            }
          out << std::endl;
        }
      out << std::endl;
    }

  return out;
}


/*! Write array contents to file
 * Input: fileName : name of the file to be written/overwritten
 * Input: inBinary : whether or not to write the file in binary
 */
template <typename T>
void Array<T>::writeToFile(std::string fileName, bool inBinary) {

#ifdef _GPU
  // We need to pass the gpu data back to the cpu to output it
  if (this->cpu_flag == 0) {
      if (this->gpu_flag == 1) {
          this->cp_gpu_cpu();
        } else {
          FatalError("At Array<T>::writeToFile: data not in GPU nor CPU");
        }
    }
#endif

  if (inBinary) {
      std::ofstream file(fileName.c_str(), std::ios::binary);
      toBinary(this, file);
      file.close();

    } else {
      std::ofstream file(fileName.c_str());
      file << *this;
      file.close();
    }

  std::cout << "Wrote array of dimensions "
            << this->dim_0 << " "
            << this->dim_1 << " "
            << this->dim_2 << " "
            << this->dim_3 << " to file " << fileName << std::endl;
}

/*! Read array contents from file
 * Input: fileName : name of the file to be read to initialize the array
 */
template <typename T>
void Array<T>::initFromFile(std::string fileName) {

  std::string fileNameBinary = fileName;
  if (fileExists(fileNameBinary)) {
      fileName = fileNameBinary;
      std::ifstream file(fileName.c_str(), std::ios::binary);
      fromBinary(this, file);
      file.close();
    } else {
      std::ifstream file(fileName.c_str());
      file >> *this;
      file.close();
    }

  std::cout << "Read array of dimensions "
            << this->dim_0 << " "
            << this->dim_1 << " "
            << this->dim_2 << " "
            << this->dim_3 << " from file " << fileName << std::endl;
}


/*! Read array contents from stream
 * Input: in : istream with the contents of the array
 * Output: array : will be modified to store the contents of the array
 * Output: istream : passess the istream back to the client for further processing if desired
 */
  template <typename R>
std::istream& operator>>(std::istream& in, Array<R>& array) {
  const int NUM_DIMS = 4; // maximum number of dimensions of an Array
  std::string line;
  Array<int> dims(NUM_DIMS); // will store the dimensions of the array

  for (int i = 0; i < NUM_DIMS; i++) {
      in >> dims(i); // ingest the dimensions
    }

  // pre-allocate memory for the array
  array.setup(dims(0), dims(1), dims(2), dims(3));

  // get all the contents from the file
  for (int l = 0; l < array.dim_3; l++)
    {
      for (int k = 0; k< array.dim_2; k++)
        {
          for(int i = 0; i < array.dim_0; i++)
            {
              for(int j=0; j < array.dim_1; j++)
                {
                  in >> array(i,j,k,l);
                }
            }
          getline(in, line); // get empty line
        }
      getline(in, line); // get empty line
    }
  return in;
}

/*! Normalizes the rows of the matrix
 * Input: matrix : matrix whose rows will be normalized by their sum
 */
template<typename R>
void Array<R>::normalizeRows() {

  bool copyBack = false;
  if (cpu_flag == 0) {
      this->mv_gpu_cpu();
      copyBack = true;
    }

  for (int i = 0; i < dim_0; i++) {
      double sum_row = 0;
      for (int j = 0; j < dim_1; j++) {
          sum_row += (*this)(i,j);
        }

      // apply the normalization
      for (int j = 0; j < dim_1; j++) {
          (*this)(i,j) /= sum_row;
        }
    }

  if (copyBack) {
      this->mv_cpu_gpu();
    }

}


/*! Write an item in binary format */
template <typename T>
void toBinary(Array<T> *array, std::ofstream& file) {
  // write the size of the array
  file.write((char*) &(array->dim_0), sizeof(int));
  file.write((char*) &(array->dim_1), sizeof(int));
  file.write((char*) &(array->dim_2), sizeof(int));
  file.write((char*) &(array->dim_3), sizeof(int));

  // write the rest of the array
  for (int i = 0; i < array->size(); i++) {
      toBinary(&(*array)(i), file);
    }
}

/*! Read an array in binary format */
template <typename T>
void fromBinary(Array<T>* array, std::ifstream& file) {
  // read the size of the array
  file.read((char*) &(array->dim_0), sizeof(int));
  file.read((char*) &(array->dim_1), sizeof(int));
  file.read((char*) &(array->dim_2), sizeof(int));
  file.read((char*) &(array->dim_3), sizeof(int));

  array->setup(array->dim_0, array->dim_1,
                array->dim_2, array->dim_3);

  // read the rest of the array
  for (int i = 0; i < array->size(); i++) {
      fromBinary(&(*array)(i), file);
    }
}


/*! BLAS wrappers */

/*! dgemm performs the operation: this = alpha * A * B + beta * this */
/*! where A, B are 2D matrices, and alpha, beta are scalars */
template <typename T>
void Array<T>::dgemm(double alpha, Array<T>& A, Array<T>& B, double beta)
{
  int Arows =  A.get_dim(0); int Acols = A.get_dim(1);

  int Brows = B.get_dim(0); int Bcols = B.get_dim(1);

  int Astride = Arows; int Bstride = Brows; int Cstride = Arows;

  // Perform a few checks before carrying out the calculations

#ifdef _CPU
  double* A_matrix = A.get_ptr_cpu();
  double* B_matrix = B.get_ptr_cpu();
  double* C_matrix = this->get_ptr_cpu();
#endif
#ifdef _GPU
  // Ensure the matrices are in the GPU; client should not worry about this
  if (this->gpu_flag == 0) {
    this->mv_cpu_gpu();
    }
  if (A.gpu_flag == 0) {
      A.mv_cpu_gpu();
    }
  if (B.gpu_flag == 0) {
      B.mv_cpu_gpu();
    }

  double* A_matrix = A.get_ptr_gpu();
  double* B_matrix = B.get_ptr_gpu();
  double* C_matrix = this->get_ptr_gpu();
#endif

  dgemm_wrapper(Arows, Bcols, Acols, alpha,
      A_matrix, Astride, B_matrix, Bstride,
      beta, C_matrix, Cstride);
}

/*! daxpy performs the operation: this = alpha * x + this */
/*! where x and y are vectors of the same length and alpha is a scalar */
template <typename T>
void Array<T>:: daxpy(double alpha, Array<T>& x) {

  int n = x.size(); // number of elements

  if (n != this->size()) {
    FatalError("at Array<T>:: daxpy: array dimensions do not match");
  }

#ifdef _CPU
  double *x_vector = x.get_ptr_cpu();
  double *y_vector = this->get_ptr_cpu();

  ::daxpy(n, alpha, x_vector, y_vector); // the double colons forces the compiler to use the daxpy function in global.cpp

#endif
#ifdef _GPU

  if (this->gpu_flag == 0) {
    this->mv_cpu_gpu();
    }
  if (x.gpu_flag == 0) {
      x.mv_cpu_gpu();
    }

  double *x_vector = x.get_ptr_gpu();
  double *y_vector = this->get_ptr_gpu();

      cublasDaxpy(n, alpha,
                  x_vector, 1 /*x vector stride*/,
                  y_vector, 1 /*y vector stride*/);
#endif
}

#endif
