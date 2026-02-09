#include <iostream>
#include <cuda_runtime.h> // for CUDA runtime API
#include <random> // for random number generation

__global__ void randomArray(int *arr, int a) {
    arr[blockIdx.x * blockDim.x + threadIdx.x] = a * threadIdx.x + blockIdx.x;
}

int main() {
    const int numBlocks = 2;
    const int numThreads = 8;
    const int numElements = numBlocks * numThreads;

    // Fixed seed
    mt19937 generator(759);
    uniform_real_distribution<float> distribution(0, 10);

    int a = distribution(generator);

    int *dA;
    size_t size = numElements * sizeof(int);
    cudaMalloc((void**)&dA, size);

    int hA[numElements];

    randomArray<<<numBlocks, numThreads>>>(dA, a);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements - 1; i++) {
    	std::printf("%d ", hA[i]);
  	}
  	std::printf("%d\n", hA[numElements - 1]);

    cudaFree(dA);
    return 0;
}
