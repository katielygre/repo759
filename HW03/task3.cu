#include <iostream>
#include <cuda_runtime.h> // for CUDA runtime API
#include <random> // for random number generation
#include "vscale.cuh"

int main(int argc, char *argv[]) {
    //check for args
    if (argc < 2) {
        std::cerr << "no argument" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    if (n <= 0) {
        std::cerr << "n must be a positive integer." << std::endl;
        return 1;
    }

    // random number gen
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> distributionA(-10, 10);
    std::uniform_real_distribution<float> distributionB(0, 1);

    //memory alloc
    float *a, *b;
    size_t size = sizeof(float) * n;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);


    // initalize a and b
    for (int i = 0; i< n; i++){
        a[i] = distributionA(generator);
        b[i] = distributionB(generator);
    }

    // calculate number of blocks needed
    int threadsPerBlock = 512;
    int numBlocks = (n + threadsPerBlock -1)/threadsPerBlock;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // prefetch
    int device = 0;
    cudaGetDevice(&device);

    // Define the location (the GPU)
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    // Use the 5-argument version: (ptr, size, location, flags, stream)
    cudaMemPrefetchAsync(a, sizeof(float) * n, loc, 0, nullptr);
    cudaMemPrefetchAsync(b, sizeof(float) * n, loc, 0, nullptr);

    cudaEventRecord(start);

    vscale<<<numBlocks, threadsPerBlock>>>(a, b, n);

    // Prefetch back to CPU for faster access by std::printf
    cudaMemPrefetchAsync(b, sizeof(float) * n, cudaCpuDeviceId, nullptr);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::printf("Kernel time (ms): %f\nFirst element: %f\nLast element: %f\n",
            ms, b[0], b[n - 1]);

    // free mem
    cudaFree(a);
    cudaFree(b);

}