#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    int k = (m - 1) / 2;
    int ni = static_cast<int>(n);

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    int r = static_cast<int>(x + i) - k;
                    int c = static_cast<int>(y + j) - k;

                    // Count how many coordinates are out of bounds
                    int out_count = (r < 0 || r >= ni) + (c < 0 || c >= ni);

                    float val;
                    if (out_count == 0) val = image[r * ni + c]; // Inside
                    else if (out_count == 1) val = 1.0f;          // Edge
                    else val = 0.0f;                             // Corner (out_count == 2)

                    sum += mask[i * m + j] * val;
                }
            }
            output[x * n + y] = sum;
        }
    }
}