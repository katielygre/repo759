#include <iostream>
#include <cuda_runtime.h>

// Kernel: each thread computes one factorial
__global__ void factorial(int *dA) {
    // get thread number (0 to 7)
    int a = threadIdx.x;

    int result = 1;

    // compute a!
    for (int i = 1; i <= a; i++) {
        result *= i;
    }

    // store answer in device array
    dA[a] = result;
}

int main(){
    const int N = 8;
    const size_t size = N * sizeof(int);

    // host array
    int hA[N];

    // device array
    int *dA;

    // allocate memory on GPU
    cudaMalloc((void**)&dA, size);

    // launch kernel: 1 block, 8 threads
    factorial<<<1, N>>>(dA);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // copy results back to CPU
    cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);

    // print answers
    for (int i = 0; i < N; i++) {
        std::cout << hA[i] << std::endl;
    }

    // free GPU memory
    cudaFree(dA);

    return 0;
}
