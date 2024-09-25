//

// Created by scott on 12/06/23.

//

#include "VideoProcessingExecutor.h"

#include "FuzzyChoquetIntegral.h"

#include "../utils.h"

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

    auto *bg = new Image(backgroundPath);

    bg->grayscale();

    auto *fg = new Image(fgPath);

    fg->grayscale();

    auto *output = new Image(bg->width, bg->height);

    FuzzyChoquetIntegral::process(fg, bg, output);

    output->save("output.png");
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

        auto *bg = new Image(backgroundPath);

        bg->grayscale();

        auto *fg = new Image(fgPath);

        fg->grayscale();

        auto *output = new Image(bg->width, bg->height);

        FuzzyChoquetIntegral::process(fg, bg, output);

        std::string outputFilePath = outputDirectory + "/" + std::filesystem::path(fgPath).filename().string();

        output->save(outputFilePath);

        std::cout << "Processed: " << fgPath << std::endl;
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

    //framesToVideo(outputDirectory, "processed_" + videoName + ".avi");
}
