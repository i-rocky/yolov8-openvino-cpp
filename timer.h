#pragma once
#include <chrono>


class Timer {
public:
    static std::chrono::steady_clock::time_point get_time();
    static void print_time(const std::chrono::high_resolution_clock::time_point& start, const std::string& message);
};
