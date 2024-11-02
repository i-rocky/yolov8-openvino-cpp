cmake_minimum_required(VERSION 3.12)
project(yolov8)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenSSL REQUIRED)
find_package(httplib COMPONENTS OpenSSL ZLIB Brotli)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(detect
        main.cpp
        inference.cpp
        handler.cpp
        handler.h
        timer.cpp
        timer.h
)

target_link_libraries(detect
        ${OpenCV_LIBS}
        openvino::runtime
        OpenSSL::SSL
        OpenSSL::Crypto
)
