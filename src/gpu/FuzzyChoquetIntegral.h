//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#ifndef GPU_FUZZYCHOQUETINTEGRAL_H
#define GPU_FUZZYCHOQUETINTEGRAL_H

#include "FeaturesExtraction.h"

__device__ int calculatePixelValue(double c1, double c2, double t);

__global__ void calculatePixelValues(Color *color, double *texture, png_bytep output, int width, int height);

void process(png_bytep fg, png_bytep bg, png_bytep output, int width, int height);

#endif//GPU_FUZZYCHOQUETINTEGRAL_H
