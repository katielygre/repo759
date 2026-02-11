#include <iostream>
#include <cuda_runtime.h> // for CUDA runtime API
#include <random> // for random number generation

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
    size = sizeof(float) * n;
    cudaMalloc((void**)&a, size);
    cudaMalloc((void**)&b, size);

    // initalize a and b
    for (int i = 0; i< n; i++){
        a[i] = distributionA(generator);
        b[i] = distributionB(generator);
    }

}