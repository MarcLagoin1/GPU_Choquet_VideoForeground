//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#include "FeaturesExtraction.h"

__global__ void ColorFeaturesKernel(Color *bg_color, png_bytep bg_pointers, int width, int height) {
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        png_bytep px = &(bg_pointers[(i * width + j) * 4]);
        bg_color[i * width + j].g = px[1];
        bg_color[i * width + j].gb = (px[1] + px[2]) / 2.0;
    }
}

__global__ void
ColorSimilarityMeasures(Color *bg_color, png_bytep bg_pointers, Color *similarity, int width, int height) {
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        png_bytep px = &(bg_pointers[(i * width + j) * 4]);

        double min_g = min(bg_color[i * width + j].g, static_cast<double>(px[1]));
        double max_g = max(bg_color[i * width + j].g, static_cast<double>(px[1]));
        similarity[i * width + j].g = (max_g == 0) ? 0 : min_g / max_g;

        double min_gb = min(bg_color[i * width + j].gb, (px[1] + px[2]) / 2.0);
        double max_gb = max(bg_color[i * width + j].gb, static_cast<double>((px[1] + px[2])) / 2);
        similarity[i * width + j].gb = (max_gb == 0) ? 0 : min_gb / max_gb;
    }
}

__global__ void TextureFeaturesKernel(png_bytep bg_pointers, uint8_t *lbp, int width, int height) {
    // Calculate global indices
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory
    __shared__ uint8_t shared_bg[34][34];  // Increased size to handle halo pixels

    // Load data into shared memory
    if (i < height && j < width) {
        shared_bg[threadIdx.y + 1][threadIdx.x + 1] = bg_pointers[i * width + j];
    }

    // Load halo pixels
    if (threadIdx.x == 0 && j > 0) {
        shared_bg[threadIdx.y + 1][0] = bg_pointers[i * width + j - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && j < width - 1) {
        shared_bg[threadIdx.y + 1][blockDim.x + 1] = bg_pointers[i * width + j + 1];
    }
    if (threadIdx.y == 0 && i > 0) {
        shared_bg[0][threadIdx.x + 1] = bg_pointers[(i - 1) * width + j];
    }
    if (threadIdx.y == blockDim.y - 1 && i < height - 1) {
        shared_bg[blockDim.y + 1][threadIdx.x + 1] = bg_pointers[(i + 1) * width + j];
    }

    // Synchronize to make sure the data is loaded before proceeding
    __syncthreads();

    // Perform the calculation using the data in shared memory
    if (i >= 1 && i < height - 1 && j >= 1 && j < width - 1) {
        uint8_t lbp_value = 0;
        lbp_value |= (shared_bg[threadIdx.y][threadIdx.x] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 7;
        lbp_value |= (shared_bg[threadIdx.y][threadIdx.x + 1] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 6;
        lbp_value |= (shared_bg[threadIdx.y][threadIdx.x + 2] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 5;
        lbp_value |= (shared_bg[threadIdx.y + 1][threadIdx.x] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 4;
        lbp_value |= (shared_bg[threadIdx.y + 1][threadIdx.x + 2] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 3;
        lbp_value |= (shared_bg[threadIdx.y + 2][threadIdx.x] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 2;
        lbp_value |= (shared_bg[threadIdx.y + 2][threadIdx.x + 1] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]) << 1;
        lbp_value |= (shared_bg[threadIdx.y + 2][threadIdx.x + 2] > shared_bg[threadIdx.y + 1][threadIdx.x + 1]);
        lbp[i * width + j] = lbp_value;
    }
}

__global__ void TextureSimilarityMeasuresKernel(uint8_t *fg, uint8_t *bg, double *similarity, int width, int height) {
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory
    __shared__ uint8_t shared_fg[32][32];
    __shared__ uint8_t shared_bg[32][32];

    // Load data into shared memory
    if (i < height && j < width) {
        shared_fg[threadIdx.y][threadIdx.x] = fg[i * width + j];
        shared_bg[threadIdx.y][threadIdx.x] = bg[i * width + j];
    }

    // Synchronize to make sure the data is loaded before proceeding
    __syncthreads();

    // Perform the calculation using the data in shared memory
    if (i < height && j < width) {
        int xorValue = shared_fg[threadIdx.y][threadIdx.x] ^ shared_bg[threadIdx.y][threadIdx.x];
        int identicalBits = 8 - __popc(xorValue);
        similarity[i * width + j] = (double) identicalBits / 8.0;
    }
}