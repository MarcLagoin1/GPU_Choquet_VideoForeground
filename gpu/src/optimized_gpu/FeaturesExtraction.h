#ifndef GPU_FEATURESEXTRACTION_H
#define GPU_FEATURESEXTRACTION_H

#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <png.h>
#include <string>
#include <vector>

struct Color {
    double g, gb;
};

__global__ void ColorFeaturesKernel(Color *img_color, png_bytep bg_pointers, int width, int height);

__global__ void ColorSimilarityMeasures(Color *fg, png_bytep bg_pointers, Color *similarity, int width, int height);

__global__ void TextureFeaturesKernel(png_bytep bg_pointers, uint8_t *lbp, int width, int height);

__global__ void TextureSimilarityMeasuresKernel(uint8_t *fg, uint8_t *bg, double *similarity, int width, int height);

#endif//GPU_FEATURESEXTRACTION_H
