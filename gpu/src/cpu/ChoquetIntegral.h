//
// Created by Pierre-Louis Delcroix on 05/06/2023.
//

#ifndef GPU_CHOQUETINTEGRAL_H
#define GPU_CHOQUETINTEGRAL_H

#include "FeaturesExtraction.h"
#include <algorithm>
#include <array>

class ChoquetIntegral {
public:
    static void
    process(const std::vector<std::vector<Color>> &color, std::vector<std::vector<double>> &texture, Image *output);

    static int calculatePixelValue(double c1, double c2, double t);
};

#endif//GPU_CHOQUETINTEGRAL_H
