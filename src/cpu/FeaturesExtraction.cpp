
//
// Created by Pierre-Louis Delcroix on 06/06/2023.
//

#include "FeaturesExtraction.h"

std::vector<std::vector<Color>> FeaturesExtraction::ColorFeatures(const Image *img) {
    std::vector<std::vector<Color>> img_color(img->height, std::vector<Color>(img->width));

    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            img_color[i][j].g = img->pixels[i][j].g;
            img_color[i][j].gb = (img->pixels[i][j].g + img->pixels[i][j].b) / 2;
        }
    }

    return img_color;
}

std::vector<std::vector<Color>>
FeaturesExtraction::ColorSimilarityMeasures(const std::vector<std::vector<Color>> &fg, const Image *bg) {
    std::vector<std::vector<Color>> similarity(bg->height, std::vector<Color>(bg->width));

    for (int i = 0; i < bg->height; i++) {
        for (int j = 0; j < bg->width; j++) {

            double min_g = min(fg[i][j].g, bg->pixels[i][j].g);
            double max_g = max(fg[i][j].g, bg->pixels[i][j].g);
            similarity[i][j].g = (min_g == 0 || max_g == 0) ? 0 : min_g / max_g;

            double min_gb = min(fg[i][j].gb, (bg->pixels[i][j].g + bg->pixels[i][j].b) / 2);
            double max_gb = max(fg[i][j].gb, (bg->pixels[i][j].g + bg->pixels[i][j].b) / 2);
            similarity[i][j].gb = (min_gb == 0 || max_gb == 0) ? 0 : min_gb / max_gb;
        }
    }

    return similarity;
}

std::vector<uint8_t> FeaturesExtraction::TextureFeatures(const Image *img) {
    std::vector<uint8_t> lbp(img->height * img->width);

    for (int i = 1; i < img->height - 1; i++) {
        for (int j = 1; j < img->width - 1; j++) {
            uint8_t lbp_value = 0;
            lbp_value |= (img->pixels_grayscale[i - 1][j - 1] > img->pixels_grayscale[i][j]) << 7;
            lbp_value |= (img->pixels_grayscale[i - 1][j] > img->pixels_grayscale[i][j]) << 6;
            lbp_value |= (img->pixels_grayscale[i - 1][j + 1] > img->pixels_grayscale[i][j]) << 5;
            lbp_value |= (img->pixels_grayscale[i][j - 1] > img->pixels_grayscale[i][j]) << 4;
            lbp_value |= (img->pixels_grayscale[i][j + 1] > img->pixels_grayscale[i][j]) << 3;
            lbp_value |= (img->pixels_grayscale[i + 1][j + 1] > img->pixels_grayscale[i][j]) << 2;
            lbp_value |= (img->pixels_grayscale[i + 1][j] > img->pixels_grayscale[i][j]) << 1;
            lbp_value |= (img->pixels_grayscale[i + 1][j - 1] > img->pixels_grayscale[i][j]) << 0;
            lbp[i * img->width + j] = lbp_value;
        }
    }
    return lbp;
}

std::vector<std::vector<double>>
FeaturesExtraction::TextureSimilarityMeasures(const std::vector<uint8_t> &fg, const std::vector<uint8_t> &bg, int h,
                                              int w) {
    std::vector<std::vector<double>> similarity(h, std::vector<double>(w));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int xorValue = fg[i * w + j] ^ bg[i * w + j];
            int identicalBits = 8 - __builtin_popcount(xorValue);
            similarity[i][j] = (double) identicalBits / 8;
        }
    }

    return similarity;
}

double FeaturesExtraction::min(double a, double b) {
    return a < b ? a : b;
}

double FeaturesExtraction::max(double a, double b) {
    return a > b ? a : b;
}