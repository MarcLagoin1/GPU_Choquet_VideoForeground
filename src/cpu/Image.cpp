//
// Created by Pierre-Louis Delcroix on 05/06/2023.
//

#include "Image.h"

Image::Image(std::string path) {
    filename = path.c_str();
    width = 0;
    height = 0;
    color_type = 0;
    bit_depth = 0;
    row_pointers = nullptr;

    pixels = std::vector<std::vector<Pixel>>();
    pixels_grayscale = std::vector<std::vector<double>>();

    load();
}

Image::Image(int width, int height) {
    filename = "";
    this->width = width;
    this->height = height;
    color_type = 0;
    bit_depth = 0;
    row_pointers = nullptr;

    pixels = std::vector<std::vector<Pixel>>(height, std::vector<Pixel>(width));
    pixels_grayscale = std::vector<std::vector<double>>(height, std::vector<double>(width, 0));
}

Image::~Image() {
    if (row_pointers) {
        for (int y = 0; y < height; y++) {
            free(row_pointers[y]);
        }
        free(row_pointers);
    }
}

void Image::load() {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << filename << " could not be opened for reading" << std::endl;
        return;
    }

    png_structp png = png_create_read_struct("1.6.39", nullptr, nullptr, nullptr);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    if (row_pointers) abort();

    row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte *) malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, nullptr);

    save_to_vector();
}

void Image::save(std::string path) {
    save_to_row_pointers();

    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) abort();

    png_structp png = png_create_write_struct("1.6.39", nullptr, nullptr, nullptr);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
            png,
            info,
            width, height,
            8,
            PNG_COLOR_TYPE_RGBA,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    if (!row_pointers) abort();

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}

void Image::save_to_vector() {
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        std::vector<Pixel> row_vector;

        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            Pixel pixel = {px[0], px[1], px[2], px[3]};
            row_vector.push_back(pixel);
        }

        pixels.push_back(row_vector);
    }
}

void Image::grayscale() {
    for (int y = 0; y < height; y++) {
        std::vector<double> row_vector;

        for (int x = 0; x < width; x++) {
            Pixel pixel = pixels[y][x];
            int gray = (pixel.r + pixel.g + pixel.b) / 3;
            row_vector.push_back(gray);
        }

        pixels_grayscale.push_back(row_vector);
    }
}

void Image::save_to_row_pointers() {
    row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);

    for (int y = 0; y < height; y++) {
        png_bytep row = (png_bytep) malloc(sizeof(png_byte) * width * 4);
        png_bytep px = row;

        for (int x = 0; x < width; x++) {
            int gray = pixels_grayscale[y][x];
            px[0] = gray;
            px[1] = gray;
            px[2] = gray;
            px[3] = 255;
            px += 4;
        }

        row_pointers[y] = row;
    }
}
