#include <iostream> // for cout, cerr
#include <random> // for random number generation
#include <chrono> // for timing
#include <cstdlib> // for malloc, free, atol
#include "convolution.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    // Check for argument
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <n> <m>" << endl;
        return 1;
    }

    // Convert arguments to size_t
    size_t n = atol(argv[1]);
    size_t m = atol(argv[2]);

    // i and ii) Create matrices
    float *image = new float[n * n];
    float *mask = new float[m * m];
    float *output = new float[n * n];

    // Fixed seed
    mt19937 generator(759);
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (size_t i = 0; i < n * n; i++) {
        image[i] = distribution(generator);
    }
    for (size_t i = 0; i < m * m; i++) {
        mask[i] = distribution(generator);
    }

    // iii) Convolve image with mask
    auto start = high_resolution_clock::now();

    convolve(image, output, n, mask, m);

    auto stop = high_resolution_clock::now();

    // iv) Print time taken
    auto elapsed = duration_cast<duration<double, std::milli>>(stop - start);
    cout << "Time taken: " << elapsed.count() << " milliseconds" << endl;

    // v) Print first element
    cout << "First element of output: " << output[0] << endl;

    // vi) Print last element
    cout << "Last element of output: " << output[n * n - 1] << endl;

    // vii) Deallocate memory
    delete[] image;
    delete[] mask;
    delete[] output;
    return 0;
}