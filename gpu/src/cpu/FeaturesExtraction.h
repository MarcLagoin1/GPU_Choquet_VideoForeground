//
// Created by Pierre-Louis Delcroix on 06/06/2023.
//

#ifndef GPU_FEATURESEXTRACTION_H
#define GPU_FEATURESEXTRACTION_H

#include "Image.h"
#include <cstdint>

struct Color {
    double g, gb;
};

class FeaturesExtraction {
public:
    static std::vector<std::vector<Color>> ColorFeatures(const Image *img);

    static std::vector<std::vector<Color>>
    ColorSimilarityMeasures(const std::vector<std::vector<Color>> &fg, const Image *bg);

    static std::vector<uint8_t> TextureFeatures(const Image *img);

    static std::vector<std::vector<double>>
    TextureSimilarityMeasures(const std::vector<uint8_t> &fg, const std::vector<uint8_t> &bg, int h, int w);

    static double min(double a, double b);

    static double max(double a, double b);
};


#endif//GPU_FEATURESEXTRACTION_H
