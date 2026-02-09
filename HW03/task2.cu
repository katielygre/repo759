#include <iostream>
#include <cuda_runtime.h> // for CUDA runtime API
#include <random> // for random number generation

__global__ void randomArray(int *arr, int a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = a * threadIdx.x + blockIdx.x;
}

int main() {
    const int numBlocks = 2;
    const int numThreads = 8;
    const int numElements = numBlocks * numThreads;

    // Fixed seed
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> distribution(0, 10);

    int a = distribution(generator);

    // device array and allocate memory on GPU
    int *dA;
    size_t size = numElements * sizeof(int);
    cudaMalloc((void**)&dA, size);

    // host array
    int hA[numElements];

    randomArray<<<numBlocks, numThreads>>>(dA, a);
    cudaDeviceSynchronize();

    //copy results back to CPU
    cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements - 1; i++) {
    	std::printf("%d ", hA[i]);
  	}
  	std::printf("%d\n", hA[numElements - 1]);

    cudaFree(dA);
    return 0;
}
