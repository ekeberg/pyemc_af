#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

const int NTHREADS = 256;
const float RESPONSABILITY_THRESHOLD = 1e-10f;

#define cudaErrorCheck(ans) {_cudaErrorCheck((ans), __FILE__, __LINE__);}
inline void _cudaErrorCheck(cudaError_t code,
			    const char *file,
			    int line, bool abort=true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "cudaErrorCheck: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}


template<typename T>
__device__ void inblock_reduce(T * data){
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1){
    if (threadIdx.x < s){
      data[threadIdx.x] += data[threadIdx.x + s];
    }
    __syncthreads();
  }
}

#endif
