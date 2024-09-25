//
// Created by scott on 12/06/23.
//

#include "VideoProcessingExecutor.h"
#include "../utils.h"
#include <filesystem>

#include "Image.h"

#include <iostream>
#include <utility>

VideoProcessingExecutor::VideoProcessingExecutor(std::string videoPath, std::string outputDirectory) {
    std::string videoName = std::filesystem::path(videoPath).filename().string();
    videoName = videoName.substr(0, videoName.find_last_of('.'));
    std::string framesDirectory = "Frames_from_video_" + videoName;
    std::filesystem::create_directory(framesDirectory);
    videoToFrames(videoPath, framesDirectory);
    imagePaths = getFilesInDirectory(framesDirectory);
    std::sort(imagePaths.begin(), imagePaths.end(), naturalSortComparator);
    this->outputDirectory = std::move(outputDirectory);
    this->videoPath = std::move(videoPath);
}

VideoProcessingExecutor::VideoProcessingExecutor(std::string outputDirectory) {
    std::string framesDirectory = "dataset/frames/";
    imagePaths = getFilesInDirectory(framesDirectory);
    std::sort(imagePaths.begin(), imagePaths.end(), naturalSortComparator);
    this->outputDirectory = std::move(outputDirectory);
}

void VideoProcessingExecutor::process_single_frame(std::string backgroundPath, std::string fgPath) {
    int width, height;
    png_byte color_type, bit_depth;

    png_bytep *bg_pointers = nullptr;
    load_image(backgroundPath, &width, &height, &color_type, &bit_depth, &bg_pointers);

    png_bytep *fg_pointers = nullptr;
    load_image(fgPath, &width, &height, &color_type, &bit_depth, &fg_pointers);

    auto bg_pointers_1D = (png_bytep) malloc(sizeof(png_byte) * width * height * 4);
    auto fg_pointers_1D = (png_bytep) malloc(sizeof(png_byte) * width * height * 4);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width * 4; ++j) {
            bg_pointers_1D[i * width * 4 + j] = bg_pointers[i][j];
            fg_pointers_1D[i * width * 4 + j] = fg_pointers[i][j];
        }
    }

    auto output_pointers = (png_bytep) malloc(sizeof(png_bytep) * width * height * 4);

    process(fg_pointers_1D, bg_pointers_1D, output_pointers, width, height);

    auto *output_pointers_2D = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; ++i) {
        output_pointers_2D[i] = output_pointers + i * width * 4;
    }

    std::string output = "output.png";

    save_image(output, width, height, output_pointers_2D);

    std::cout << "Processed: " << fgPath << ", saved as: " << output << std::endl;
}

void VideoProcessingExecutor::processImages() {
    if (imagePaths.size() < 2) {
        std::cerr << "Insufficient number of images provided." << std::endl;
        return;
    }

    const std::string &backgroundPath = imagePaths.front();

    std::filesystem::create_directory(outputDirectory);

    for (auto it = imagePaths.begin() + 1; it != imagePaths.end(); ++it) {

        const std::string &fgPath = *it;
        int width, height;
        png_byte color_type, bit_depth;

        png_bytep *bg_pointers = nullptr;
        load_image(backgroundPath, &width, &height, &color_type, &bit_depth, &bg_pointers);

        png_bytep *fg_pointers = nullptr;
        load_image(fgPath, &width, &height, &color_type, &bit_depth, &fg_pointers);

        auto bg_pointers_1D = (png_bytep) malloc(sizeof(png_byte) * width * height * 4);
        auto fg_pointers_1D = (png_bytep) malloc(sizeof(png_byte) * width * height * 4);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width * 4; ++j) {
                bg_pointers_1D[i * width * 4 + j] = bg_pointers[i][j];
                fg_pointers_1D[i * width * 4 + j] = fg_pointers[i][j];
            }
        }

        auto output_pointers = (png_bytep) malloc(sizeof(png_bytep) * width * height * 4);

        process(fg_pointers_1D, bg_pointers_1D, output_pointers, width, height);

        auto *output_pointers_2D = (png_bytep *) malloc(sizeof(png_bytep) * height);
        for (int i = 0; i < height; ++i) {
            output_pointers_2D[i] = output_pointers + i * width * 4;
        }

        std::string outputFilePath = outputDirectory + "/" + std::filesystem::path(fgPath).filename().string();

        save_image(outputFilePath, width, height, output_pointers_2D);

        std::cout << "Processed: " << fgPath << ", saved as: " << outputFilePath << std::endl;
    }

    std::cout << "Processing completed." << std::endl;
}

void VideoProcessingExecutor::processVideo() {
    std::string videoName = std::filesystem::path(videoPath).filename().string();
    videoName = videoName.substr(0, videoName.find_last_of('.'));
    std::string framesDirectory = "Frames_from_video_" + videoName;

    std::filesystem::create_directory(framesDirectory);
    videoToFrames(videoPath, framesDirectory);

    this->processImages();
    framesToVideo(outputDirectory, "processed_" + videoName + ".avi");
}
