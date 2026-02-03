#include <iostream> // for cout, cerr
#include <random> // for random number generation
#include <chrono> // for timing
#include <cstdlib> // for malloc, free, atol
#include "scan.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    // Check for argument
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <n>" << endl;
        return 1;
    }

    // Convert argument to size_t
    size_t n = atol(argv[1]);

    // i) Create arrays
    float *arr = (float*) malloc(n * sizeof(float));
    float *output = (float*) malloc(n * sizeof(float));

    // Fixed seed
    mt19937 generator(759);
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (size_t i = 0; i < n; i++) {
        arr[i] = distribution(generator);
    }

    // ii) Scan array
    auto start = high_resolution_clock::now();

    scan(arr, output, n);

    auto stop = high_resolution_clock::now();

    // iii) Print time taken
    auto elapsed = duration_cast<duration<double, std::milli>>(stop - start);

    cout << "Time taken for scan of size " << n << ": " << elapsed.count() << " milliseconds" << endl;

    // iv) Print first element
    cout << "First element of output: " << output[0] << endl;

    // v) Print last element
    cout << "Last element of output: " << output[n-1] << endl;

    // vi) Deallocate memory
    free(arr);
    free(output);
    return 0;
}
