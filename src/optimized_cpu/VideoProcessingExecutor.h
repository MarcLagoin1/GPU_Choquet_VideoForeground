//
// Created by scott on 12/06/23.
//
#pragma once

#include "FuzzyChoquetIntegral.h"
#include <string>

class VideoProcessingExecutor {
private:
    std::string videoPath;
    std::string outputDirectory;
    std::vector<std::string> imagePaths;

public:
    VideoProcessingExecutor(std::string videoPath, std::string outputDirectory);

    VideoProcessingExecutor(std::string outputDirectory);

    static void process_single_frame(std::string backgroundPath, std::string fgPath);

    void processImages();

    void processVideo();
};