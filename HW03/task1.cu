#include <iostream>
#include <cuda_runtime.h>

// Kernel: each thread computes one factorial
__global__ void factorial(int *dA)
{
    // get thread number (0 to 7)
    int tid = threadIdx.x;

    int result = 1;

    // compute tid!
    for (int i = 1; i <= tid; i++)
    {
        result = result * i;
    }

    // store answer in device array
    dA[tid] = result;
}

int main()
{
    const int N = 8;

    // host array
    int hA[N];

    // device array
    int *dA;

    // allocate memory on GPU
    cudaMalloc(&dA, N * sizeof(int));

    // launch kernel: 1 block, 8 threads
    factorial<<<1, N>>>(dA);

    // copy results back to CPU
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    // print answers
    for (int i = 0; i < N; i++)
    {
        std::cout << hA[i] << std::endl;
    }

    // free GPU memory
    cudaFree(dA);

    return 0;
}
