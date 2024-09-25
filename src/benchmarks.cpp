//
// Created by Pierre-Louis Delcroix on 12/06/2023.
//

#include "benchmarks.h"

// Function to measure CPU usage
double get_cpu_usage() {
    double cpu_percentage = 0.0;

    rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double kernel_time = usage.ru_stime.tv_sec + (usage.ru_stime.tv_usec / 1000000.0);
        double user_time = usage.ru_utime.tv_sec + (usage.ru_utime.tv_usec / 1000000.0);
        double total_time = kernel_time + user_time;
        double elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        cpu_percentage = (total_time / elapsed_time) * 100.0;
    }

    return cpu_percentage;
}

// Function to measure memory usage
size_t get_memory_usage() {
    size_t memory_usage = 0;
    FILE *file = fopen("/proc/self/statm", "r");
    if (file) {
        long pagesize = sysconf(_SC_PAGESIZE);
        unsigned long size, resident, share, text, lib, data, dt;
        if (fscanf(file, "%lu %lu %lu %lu %lu %lu %lu", &size, &resident, &share, &text, &lib, &data, &dt) == 7) {
            memory_usage = resident * pagesize;
        }
        fclose(file);
    }

    return memory_usage;
}

void print_benchmarks(std::chrono::duration<long long int, std::ratio<1LL, 1000LL>> duration, size_t memory_usage,
                      double cpu_usage) {
    std::cout << "Running Time: " << duration.count() << " milliseconds\n";
    std::cout << "CPU Usage: " << cpu_usage << "%\n";
    std::cout << "Memory Usage: " << memory_usage << " bytes\n";
}