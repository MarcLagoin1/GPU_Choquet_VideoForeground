#include "FuzzyChoquetIntegral.h"

__device__ int calculatePixelValue(double c1, double c2, double t) {
    // Put the similarity measures in an array
    double similarities[3] = {c1, c2, t};

    // Sort the array in ascending order (using bubble sort for simplicity)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2 - i; ++j) {
            if (similarities[j] > similarities[j + 1]) {
                double temp = similarities[j];
                similarities[j] = similarities[j + 1];
                similarities[j + 1] = temp;
            }
        }
    }

    // Define the weighting array
    double weights[3] = {0.1, 0.3, 0.6};

    // Calculate the dot product
    double dotProduct = 0.0;
    for (int i = 0; i < 3; ++i) {
        dotProduct += weights[i] * similarities[i];
    }

    // Apply the threshold
    if (dotProduct > 0.67) {
        return 0;// Pixel belongs to the background
    } else {
        return 1;// Pixel does not belong to the background
    }
}

__global__ void calculatePixelValues(Color *color, double *texture, png_bytep output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        double pixelValue = calculatePixelValue(color[y * width + x].g, color[y * width + x].gb,
                                                texture[y * width + x]);
        output[y * width * 4 + x * 4] = (png_byte)(pixelValue * 255.0);
        output[y * width * 4 + x * 4 + 1] = (png_byte)(pixelValue * 255.0);
        output[y * width * 4 + x * 4 + 2] = (png_byte)(pixelValue * 255.0);
        output[y * width * 4 + x * 4 + 3] = 255;
    }
}

void process(png_bytep fg, png_bytep bg, png_bytep output, int width, int height) {
    dim3 block_size(32, 32);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    png_bytep fg_device, bg_device;
    cudaMalloc(&fg_device, sizeof(png_byte) * width * height * 4);
    cudaMalloc(&bg_device, sizeof(png_byte) * width * height * 4);
    cudaMemcpy(fg_device, fg, sizeof(png_byte) * width * height * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(bg_device, bg, sizeof(png_byte) * width * height * 4, cudaMemcpyHostToDevice);

    Color *bg_color, *bg_similarity;
    cudaMalloc(&bg_color, width * height * sizeof(Color));
    cudaMalloc(&bg_similarity, width * height * sizeof(Color));

    uint8_t *fg_texture, *bg_texture;
    cudaMalloc(&fg_texture, width * height * sizeof(uint8_t));
    cudaMalloc(&bg_texture, width * height * sizeof(uint8_t));

    double *fg_similarity;
    cudaMalloc(&fg_similarity, width * height * sizeof(double));

    ColorFeaturesKernel<<<grid_size, block_size>>>(bg_color, bg_device, width, height);
    ColorSimilarityMeasures<<<grid_size, block_size>>>(bg_color, fg_device, bg_similarity, width, height);

    TextureFeaturesKernel<<<grid_size, block_size>>>(fg_device, fg_texture, width, height);
    TextureFeaturesKernel<<<grid_size, block_size>>>(bg_device, bg_texture, width, height);
    TextureSimilarityMeasuresKernel<<<grid_size, block_size>>>(fg_texture, bg_texture, fg_similarity, width, height);

    calculatePixelValues<<<grid_size, block_size>>>(bg_similarity, fg_similarity, fg_device, width, height);

    cudaMemcpy(output, fg_device, sizeof(png_byte) * width * height * 4, cudaMemcpyDeviceToHost);

    cudaFree(fg_device);
    cudaFree(bg_device);
    cudaFree(bg_color);
    cudaFree(fg_texture);
    cudaFree(bg_texture);
    cudaFree(fg_similarity);
    cudaFree(bg_similarity);
}
