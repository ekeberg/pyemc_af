#include <Python.h>
#include "afemc.h"

using namespace std;

#define cudaErrorCheck(ans) {_cudaErrorCheck((ans), __FILE__, __LINE__);}
inline void _cudaErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "cudaErrorCheck: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

float *int_to_float_pointer(const unsigned long long pointer_int)
{
  float *pointer = (float *)pointer_int;
  return pointer;
}

int *int_to_int_pointer(const unsigned long long pointer_int)
{
  int *pointer = (int *)pointer_int;
  return pointer;
}

__global__ void add_one_kernel(float *array, const int length)
{
  int index = blockDim.x*blockIdx.x + threadIdx.x;
  if (index < length) {
    array[index] += 1.;
  }
}

//void cuda_add_one(float *const array, const int length)
void cuda_add_one(float *const array, const int length)
{
  const int nthreads = 256;
  const int nblocks = (length+nthreads-1)/nthreads;
  add_one_kernel<<<nblocks, nthreads>>>(array, length);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

