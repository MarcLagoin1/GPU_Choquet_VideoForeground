//
// Created by Pierre-Louis Delcroix on 05/06/2023.
//

#include "ChoquetIntegral.h"

void ChoquetIntegral::process(const std::vector<std::vector<Color>> &color, std::vector<std::vector<double>> &texture,
                              Image *output) {
    for (int i = 0; i < output->height; ++i) {
        for (int j = 0; j < output->width; ++j) {
            double pixelValue = calculatePixelValue(color[i][j].g, color[i][j].gb, texture[i][j]);
            output->pixels_grayscale[i][j] = pixelValue * 255;
        }
    }
}

int ChoquetIntegral::calculatePixelValue(double c1, double c2, double t) {
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