#include <iostream>
#include <cuda_runtime.h>

// Kernel: each thread computes one factorial
__global__ void factorial(int *dA) {
    // get thread number (0 to 7)
    int idx = threadIdx.x;
    int a = idx + 1; // compute a! where a = thread number

    int result = 1;

    // compute a!
    for (int i = 1; i <= a; i++) {
        result *= i;
    }

    // store answer in device array
    dA[idx] = result;
}

int main(){
    const int N = 8;
    const size_t size = N * sizeof(int);

    // host array
    int hA[N];

    // device array
    int *dA;

    // allocate memory on GPU
    cudaError_t err = cudaMalloc((void**)&dA, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // launch kernel: 1 block, 8 threads
    factorial<<<1, N>>>(dA);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dA);
        return -1;
    }

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // copy results back to CPU
    err = cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dA);
        return -1;
    }

    // print answers
    for (int i = 0; i < N; i++) {
        std::cout << hA[i] << std::endl;
    }

    // free GPU memory
    cudaFree(dA);

    return 0;
}
