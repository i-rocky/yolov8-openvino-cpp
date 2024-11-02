//
// Created by Rasel Rana Rocky on 11/2/24.
//

#include "timer.h"
#include <iostream>


std::chrono::steady_clock::time_point Timer::get_time()
{
    return std::chrono::high_resolution_clock::now();
}

void Timer::print_time(const std::chrono::high_resolution_clock::time_point& start, const std::string& message)
{
    const auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << message << ": " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - start).count() / 1000.0 << " milliseconds" << std::endl;
}
