#include "Image.h"
#include <cuda_runtime.h>

void load_image(std::string path, int *width, int *height, png_byte *color_type, png_byte *bit_depth,
                png_bytep **row_pointers, bool video) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        std::cerr << path << " could not be opened for reading" << std::endl;
        return;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    *color_type = png_get_color_type(png, info);
    *bit_depth = png_get_bit_depth(png, info);

    if (*bit_depth == 16)
        png_set_strip_16(png);

    if (*color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if (*color_type == PNG_COLOR_TYPE_GRAY && *bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (*color_type == PNG_COLOR_TYPE_RGB ||
        *color_type == PNG_COLOR_TYPE_GRAY ||
        *color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (*color_type == PNG_COLOR_TYPE_GRAY ||
        *color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    if (!video && *row_pointers) abort();

    *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * *height);
    for (int y = 0; y < *height; y++) {
        (*row_pointers)[y] = (png_byte *) malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, *row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, nullptr);
}

void save_image(std::string path, int width, int height, png_bytep *row_pointers) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
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

    if (!row_pointers) abort();

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    free(row_pointers);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}