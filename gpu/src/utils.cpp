//
// Created by pierre-louis.delcroix on 6/12/23.
//

#include "utils.h"

int extractNumber(const std::string &str) {
    size_t startPos = str.find_first_of("0123456789");
    size_t endPos = str.find_last_of("0123456789");
    if (startPos != std::string::npos && endPos != std::string::npos) {
        return std::stoi(str.substr(startPos, endPos - startPos + 1));
    }
    return 0;
}

bool naturalSortComparator(const std::string &a, const std::string &b) {
    int numA = extractNumber(a);
    int numB = extractNumber(b);

    return numA < numB;
}

void videoToFrames(const std::string &videoPath, const std::string &outputDir) {
    std::string command = "ffmpeg -i " + videoPath + " " + outputDir + "/frame%04d.png";
    system(command.c_str());
}

void framesToVideo(const std::string &inputDir, const std::string &outputPath) {
    std::string command =
            "ffmpeg -framerate 16 -i " + inputDir + "/frame%04d.png -c:v libx264 -pix_fmt yuv420p " +
            outputPath;
    system(command.c_str());
}

std::vector<std::string> getFilesInDirectory(const std::string &directoryPath) {
    std::vector<std::string> filePaths;

    for (const auto &entry: std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            filePaths.push_back(entry.path().string());
        }
    }

    return filePaths;
}