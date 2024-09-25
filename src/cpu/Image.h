//
// Created by Pierre-Louis Delcroix on 05/06/2023.
//

#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

#include <cstdio>
#include <iostream>
#include <png.h>
#include <string>
#include <vector>

struct Pixel {
    int r, g, b, a;
};

class Image {
public:
    explicit Image(std::string path);

    Image(int width, int height);

    ~Image();

    void load();

    void save_to_vector();

    void save_to_row_pointers();

    void grayscale();

    void save(std::string path);

    const char *filename;
    std::vector<std::vector<Pixel>> pixels;
    std::vector<std::vector<double>> pixels_grayscale;

    int width, height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep *row_pointers = nullptr;
};


#endif//GPU_IMAGE_H
