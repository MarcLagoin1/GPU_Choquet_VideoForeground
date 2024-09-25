
//
// Created by pierre-louis.delcroix on 6/12/23.
//

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

int extractNumber(const std::string &str);

bool naturalSortComparator(const std::string &a, const std::string &b);

void videoToFrames(const std::string &videoPath, const std::string &outputDir);

void framesToVideo(const std::string &inputDir, const std::string &outputPath);

std::vector<std::string> getFilesInDirectory(const std::string &directoryPath);

#endif//GPU_UTILS_H
