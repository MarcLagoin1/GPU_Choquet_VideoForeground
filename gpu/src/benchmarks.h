//
// Created by Pierre-Louis Delcroix on 12/06/2023.
//

#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <chrono>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

double get_cpu_usage();

size_t get_memory_usage();

void print_benchmarks(std::chrono::duration<long long int, std::ratio<1LL, 1000LL>> duration, size_t memory_usage,
                      double cpu_usage);

#endif//BENCHMARKS_H
