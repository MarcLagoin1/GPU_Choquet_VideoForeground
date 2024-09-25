#include "../benchmarks.h"
#include "VideoProcessingExecutor.h"
#include <iostream>

void single_frame() {
    std::string fg = "dataset/frames/1.png";
    std::string bg = "dataset/frames/2.png";

    VideoProcessingExecutor::process_single_frame(bg, fg);
}

void video() {
    std::string video_path = "dataset/video.avi";
    std::string outputDirectory = "dataset/output";

    VideoProcessingExecutor executor(video_path, outputDirectory);
    executor.processVideo();

}

int main() {
    auto start_time = std::chrono::steady_clock::now();

    //single_frame();
    video();

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double cpu_usage = get_cpu_usage();
    size_t memory_usage = get_memory_usage();

    print_benchmarks(duration, memory_usage, cpu_usage);

    return 0;
}