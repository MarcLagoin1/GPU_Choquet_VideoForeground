//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#ifndef OPTIMIZED_FEATURESEXTRACTION_H
#define OPTIMIZED_FEATURESEXTRACTION_H

#include "Image.h"
#include <cstdint>

struct Color {
    double g, gb;
};

std::vector<std::vector<Color>> ColorFeatures(png_bytep bg_pointers_1D, int width, int height);

std::vector<std::vector<Color>>
ColorSimilarityMeasures(const std::vector<std::vector<Color>> &fg, png_bytep bg_pointers_1D, int width, int height);

std::vector<uint8_t> TextureFeatures(png_bytep bg_pointers_1D, int width, int height);

std::vector<std::vector<double>>
TextureSimilarityMeasures(const std::vector<uint8_t> &fg, const std::vector<uint8_t> &bg, int h, int w);

#endif//OPTIMIZED_FEATURESEXTRACTION_H
