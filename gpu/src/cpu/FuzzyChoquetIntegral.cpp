//
// Created by Pierre-Louis Delcroix on 06/06/2023.
//

#include "FuzzyChoquetIntegral.h"
#include <iostream>

void FuzzyChoquetIntegral::process(const Image *fg, const Image *bg, Image *output) {
    auto bg_color = FeaturesExtraction::ColorFeatures(bg);
    auto bg_similarity = FeaturesExtraction::ColorSimilarityMeasures(bg_color, fg);

    auto fg_texture = FeaturesExtraction::TextureFeatures(fg);
    auto bg_texture = FeaturesExtraction::TextureFeatures(bg);
    auto fg_similarity = FeaturesExtraction::TextureSimilarityMeasures(fg_texture, bg_texture, bg->height, bg->width);

    ChoquetIntegral::process(bg_similarity, fg_similarity, output);
}