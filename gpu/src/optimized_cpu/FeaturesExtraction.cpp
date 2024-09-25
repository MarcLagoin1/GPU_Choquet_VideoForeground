
//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#include "FeaturesExtraction.h"

std::vector<std::vector<Color>> ColorFeatures(png_bytep bg_pointers_1D, int width, int height) {
    std::vector<std::vector<Color>> img_color(height, std::vector<Color>(width));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            png_bytep px = bg_pointers_1D + (i * width + j) * 4;
            img_color[i][j].g = px[1];
            img_color[i][j].gb = (px[1] + px[2]) / 2;
        }
    }

    return img_color;
}

std::vector<std::vector<Color>>
ColorSimilarityMeasures(const std::vector<std::vector<Color>> &fg, png_bytep bg_pointers_1D, int width, int height) {
    std::vector<std::vector<Color>> similarity(height, std::vector<Color>(width));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            png_bytep px = bg_pointers_1D + (i * width + j) * 4;

            double min_g = fg[i][j].g < px[1] ? fg[i][j].g : px[1];
            double max_g = fg[i][j].g > px[1] ? fg[i][j].g : px[1];
            similarity[i][j].g = (min_g == 0 || max_g == 0) ? 0 : min_g / max_g;

            double min_gb = fg[i][j].gb < (px[1] + px[2]) / 2 ? fg[i][j].gb : (px[1] + px[2]) / 2;
            double max_gb = fg[i][j].gb > (px[1] + px[2]) / 2 ? fg[i][j].gb : (px[1] + px[2]) / 2;
            similarity[i][j].gb = (min_gb == 0 || max_gb == 0) ? 0 : min_gb / max_gb;
        }
    }

    return similarity;
}

std::vector<uint8_t> TextureFeatures(png_bytep bg_pointers_1D, int width, int height) {
    std::vector<uint8_t> lbp(height * width);

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            png_bytep px = bg_pointers_1D + (i * width + j) * 4;
            uint8_t lbp_value = 0;
            lbp_value |= (bg_pointers_1D[(i - 1) * width * 4 + (j - 1) * 4 + 1] > px[1]) << 7;
            lbp_value |= (bg_pointers_1D[(i - 1) * width * 4 + j * 4 + 1] > px[1]) << 6;
            lbp_value |= (bg_pointers_1D[(i - 1) * width * 4 + (j + 1) * 4 + 1] > px[1]) << 5;
            lbp_value |= (px[(j - 1) * 4 + 1] > px[1]) << 4;
            lbp_value |= (px[(j + 1) * 4 + 1] > px[1]) << 3;
            lbp_value |= (bg_pointers_1D[(i + 1) * width * 4 + (j + 1) * 4 + 1] > px[1]) << 2;
            lbp_value |= (bg_pointers_1D[(i + 1) * width * 4 + j * 4 + 1] > px[1]) << 1;
            lbp_value |= (bg_pointers_1D[(i + 1) * width * 4 + (j - 1) * 4 + 1] > px[1]);
            lbp[i * width + j] = lbp_value;
        }
    }

    return lbp;
}

std::vector<std::vector<double>>
TextureSimilarityMeasures(const std::vector<uint8_t> &fg, const std::vector<uint8_t> &bg, int h, int w) {
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