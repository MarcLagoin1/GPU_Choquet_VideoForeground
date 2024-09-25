//
// Created by Pierre-Louis Delcroix on 09/06/2023.
//

#ifndef OPTIMIZED_IMAGE_H
#define OPTIMIZED_IMAGE_H

#include <cstdio>
#include <iostream>
#include <png.h>
#include <string>
#include <vector>

void load_image(std::string path, int *width, int *height, png_byte *color_type, png_byte *bit_depth,
                png_bytep **row_pointers);

void save_image(std::string path, int width, int height, png_bytep *row_pointers);

#endif//OPTIMIZED_IMAGE_H
