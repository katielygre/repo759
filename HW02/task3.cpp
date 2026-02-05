#include <iostream> // for cout, cerr
#include <random> // for random number generation
#include <chrono> // for timing
#include <cstdlib> // for malloc, free, atol
#include "matmul.h"

using namespace std;
using namespace std::chrono;

int main() {
    // Set up
    const unsigned int n = 1000;
    const unsigned int size = n * n;

    // Generate random input matricoes
    std::vector<double> A_vec(size);
    std::vector<double> B_vec(size);

    std::mt19937 generator(759);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (unsigned int i = 0; i < size; i++) {
        A_vec[i] = distribution(generator);
        B_vec[i] = distribution(generator);
    }

    const double* A = A_vec.data();
    const double* B = B_vec.data();
    double* C = new double[size];

    std::cout << "Matrix rows: " << n << std::endl;


    auto benchmark = [&](void (*func)(const double*, const double*, double*, const unsigned int)) {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C, n);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " milliseconds" << std::endl;
        std::cout << "Last element of C: " << C[size - 1] << std::endl;
    };

    std::cout << "mmul1" << std::endl;
    benchmark(mmul1);
    std::cout << "mmul2" << std::endl;
    benchmark(mmul2);
    std::cout << "mmul3" << std::endl;
    benchmark(mmul3);
    std::cout << "mmul4" << std::endl;
    auto start4 = high_resolution_clock::now();
    mmul4(A_vec, B_vec, C, n);
    auto end4 = high_resolution_clock::now();

    duration<double, std::milli> elapsed4 = end4 - start4;
    cout << "Elapsed time: " << elapsed4.count() << " milliseconds" << endl;
    cout << "Last element of C: " << C[size - 1] << endl;

    delete[] C;
    return 0;
}