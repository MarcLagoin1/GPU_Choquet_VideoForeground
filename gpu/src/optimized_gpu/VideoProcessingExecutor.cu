//
// Created by scott on 12/06/23.
//

#include "FuzzyChoquetIntegral.h"
#include "VideoUtils.h"
#include "VideoProcessingExecutor.h"
#include "Image.h"

#include <utility>
#include <iostream>
#include <algorithm>

__global__ void convert2Dto1D(png_bytep *input, png_bytep output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        for (int i = 0; i < 4; ++i) {
            output[idy * width * 4 + idx * 4 + i] = input[idy][idx * 4 + i];
        }
    }
}

VideoProcessingExecutor::VideoProcessingExecutor(std::string videoPath, std::string outputDirectory) {
    std::string videoName = std::filesystem::path(videoPath).filename().string();
    videoName = videoName.substr(0, videoName.find_last_of('.'));
    std::string framesDirectory = "Frames_from_video_" + videoName;
    std::filesystem::create_directory(framesDirectory);
    VideoUtils::videoToFrames(videoPath, framesDirectory);
    imagePaths = VideoUtils::getFilesInDirectory(framesDirectory);
    std::sort(imagePaths.begin(), imagePaths.end());
    this->outputDirectory = std::move(outputDirectory);
    this->videoPath = std::move(videoPath);
}

void VideoProcessingExecutor::processImages(const std::vector <std::string> &imagePaths,
                                            const std::string &outputDirectory) {
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
    VideoUtils::videoToFrames(videoPath, framesDirectory);

    this->processImages(imagePaths, outputDirectory);
    VideoUtils::framesToVideo(outputDirectory, "processed_" + videoName + ".avi");
}
