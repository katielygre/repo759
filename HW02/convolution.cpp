#include convolution.h

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    int k = (m - 1) / 2;

        for (std::size_t x = 0; x < n; ++x) {
            for (std::size_t y = 0; y < n; ++y) {
                float sum = 0.0f;

                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < m; ++j) {
                        // Calculate the coordinates in the image space
                        int img_i = static_cast<int>(x + i) - k;
                        int img_j = static_cast<int>(y + j) - k;

                        float val;
                        bool i_out = (img_i < 0 || img_i >= static_cast<int>(n));
                        bool j_out = (img_j < 0 || img_j >= static_cast<int>(n));

                        if (!i_out && !j_out) {
                            // Inside bounds
                            val = image[img_i * n + img_j];
                        } else if (i_out && j_out) {
                            // Corner (both out)
                            val = 0.0f;
                        } else {
                            // Edge (only one out)
                            val = 1.0f;
                        }

                        sum += mask[i * m + j] * val;
                    }
                }
                output[x * n + y] = sum;
            }
        }
    }