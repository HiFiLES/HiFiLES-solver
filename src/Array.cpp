#ifndef ARRAY_CPP_
#define ARRAY_CPP_


//#include "../include/Array.h"
#include "../include/funcs.h"

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
}
//*/
// copy constructor

template <typename T>
Array<T>::Array(const Array<T>& in_Array)
{
  int i;

  dim_0=in_Array.dim_0;
  dim_1=in_Array.dim_1;
  dim_2=in_Array.dim_2;
  dim_3=in_Array.dim_3;

  cpu_data = new T[dim_0*dim_1*dim_2*dim_3];

  for(i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
    {
      cpu_data[i]=in_Array.cpu_data[i];
    }
}

// assignment

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& in_Array)
{
  int i;

  if(this == &in_Array)
    {
      return (*this);
    }
  else
    {
      delete[] cpu_data;

      dim_0=in_Array.dim_0;
      dim_1=in_Array.dim_1;
      dim_2=in_Array.dim_2;
      dim_3=in_Array.dim_3;

      cpu_data = new T[dim_0*dim_1*dim_2*dim_3];
      //NOTE: THIS COPIES POINTERS; NOT VALUES
      for(i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
        {
          cpu_data[i]=in_Array.cpu_data[i];
        }

      cpu_flag=1;
      gpu_flag=0;

      return (*this);
    }
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

  cpu_data=new T[dim_0*dim_1*dim_2*dim_3];
  cpu_flag=1;
  gpu_flag=0;
}

template <typename T>
T& Array<T>::operator()(int in_pos_0)
{
  return cpu_data[in_pos_0]; // column major with matrix indexing
}

template <typename T>
T& Array<T>::operator()(int in_pos_0, int in_pos_1)
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)]; // column major with matrix indexing
}

template <typename T>
T& Array<T>::operator()(int in_pos_0, int in_pos_1, int in_pos_2)
{
  return cpu_data[in_pos_0+(dim_0*in_pos_1)+(dim_0*dim_1*in_pos_2)]; // column major with matrix indexing
}

template <typename T>
T& Array<T>::operator()(int in_pos_0, int in_pos_1, int in_pos_2, int in_pos_3)
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
void Array<T>::initialize_to_value(const T val)
{
  for(int i=0; i<dim_0*dim_1*dim_2*dim_3; i++)
  {
    cpu_data[i]=val;
  }
}


// returns an ostream object with the contents of the array
template <typename T>
std::ostream& operator<<(std::ostream& out, Array<T>& array) {
  if(array.dim_3 == 1)
    {
      bool threeD = (array.dim_2 == 1 ? false:true );
      for (int k = 0; k< array.dim_2; k++)
        {
          if (threeD)
            out << std::endl << "ans(:,:," << k+1 << ") = " << std::endl;
          for(int i = 0; i < array.dim_0; i++)
            {
              for(int j=0; j < array.dim_1; j++)
                {

                  if(array(i,j,k) * array(i,j,k) < 1e-12)
                    {
                      out << std::left << std::setw(13) << std::setprecision(6)
                           << "0";
                    }
                  else
                    {
                      out << std::left << std::setw(13) << std::setprecision(6)
                           << array(i,j,k);
                    }
                }

              out << std::endl;
            }
          if (threeD)
            out << std::endl;
        }
    }
  else
    {
      out << "ERROR: Can only print an Array of dimension three or less ...." << std::endl;
    }
  return out;
}

/*! Write array contents to file
 * Input: fileName : name of the file to be written/overwritten
 * Input: overwriteEnabled : overwrites existing file if true; otherwise leaves existing file intact
 */
template <typename T>
void Array<T>::writeToFile(const std::string fileName, bool overwriteEnabled) {
  if (this->dim_2 > 1) FatalError("At Array<T>::writeToFile; cannot write matrix of dimensions greater than 2 to file");

  if (!overwriteEnabled) { // if we don't want to overwrite the file, check for its existence

      if ( fileExists(fileName) ) return;
    }
  std::ofstream file(fileName.c_str());
  file << *this;
  file.close();

}

/*! Read array contents from file
 * Input: fileName : name of the file to be written/overwritten
 * Input: overwriteEnabled : overwrites existing file if true; otherwise leaves existing file intact
 */
template <typename T>
void Array<T>::initFromFile(const std::string fileName) {

  std::ifstream file(fileName.c_str());
  file >> *this;
  file.close();

}


/*! Read array contents from stream
 * Input: in : istream with the contents of the array
 * Output: array : will be modified to store the contents of the array
 * Output: istream : passess the istream back to the client for further processing if desired
 */
template <typename R>
std::istream& operator>>(std::istream& in, Array<R>& array) {

  std::vector< std::vector<double> > temp_array;

  std::string line;
  while (getline(in, line)) {
      std::istringstream input(line);

      std::vector<double> temp_row;
      double temp_double;
      while (input >> temp_double) {
          temp_row.push_back(temp_double);
        }
      temp_array.push_back(temp_row);
    }

  // Resize the array and copy its contents
  int nRows = temp_array.size();
  int nCols = temp_array[0].size();

  array.setup(nRows, nCols);

  for (int j = 0; j < nCols; j++) {
      for (int i = 0; i < nRows; i++) {
          array(i,j) = temp_array[i][j];
        }
    }

  return in;
}

#endif

