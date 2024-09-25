//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#include "FuzzyChoquetIntegral.h"
#include <array>
#include <algorithm>

void process(png_bytep fg, png_bytep bg, png_bytep output, int width, int height) {
    auto bg_color = ColorFeatures(bg, width, height);
    auto bg_similarity = ColorSimilarityMeasures(bg_color, fg, width, height);

    auto fg_texture = TextureFeatures(fg, width, height);
    auto bg_texture = TextureFeatures(bg, width, height);
    auto fg_similarity = TextureSimilarityMeasures(fg_texture, bg_texture, height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            auto pixelValue = calculatePixelValue(bg_similarity[i][j].g, bg_similarity[i][j].gb, fg_similarity[i][j]);
            output[i * width * 4 + j * 4 + 0] = pixelValue * 255;
            output[i * width * 4 + j * 4 + 1] = pixelValue * 255;
            output[i * width * 4 + j * 4 + 2] = pixelValue * 255;
            output[i * width * 4 + j * 4 + 3] = 255;
        }
    }
}

int calculatePixelValue(double c1, double c2, double t) {
    // Put the similarity measures in a vector
    std::vector<double> similarities = {c1, c2, t};

    // Sort the vector in ascending order
    std::sort(similarities.begin(), similarities.end());

    // Define the weighting vector
    std::array<double, 3> weights = {0.1, 0.3, 0.6};

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