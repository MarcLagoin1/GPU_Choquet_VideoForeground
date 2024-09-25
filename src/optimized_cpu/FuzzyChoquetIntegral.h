//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#ifndef OPTIMIZED_FUZZYCHOQUETINTEGRAL_H
#define OPTIMIZED_FUZZYCHOQUETINTEGRAL_H

#include "FeaturesExtraction.h"

void process(png_bytep fg, png_bytep bg, png_bytep output, int width, int height);

int calculatePixelValue(double c1, double c2, double t);


#endif//OPTIMIZED_FUZZYCHOQUETINTEGRAL_H
