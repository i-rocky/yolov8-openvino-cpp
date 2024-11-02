#pragma once

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include "inference.h"

class Handler {
    yolo::Inference *inference_;
    std::vector<std::string> mime_types_ = {"image/jpg", "image/jpeg", "image/png", "image/webp"};
    httplib::Server server_;
    std::mutex *mutex_;

    bool handles(const std::string& mime_type) const;
    static void split_url(const std::string& url, std::string& host, std::string& path);
    void apply_blur(cv::Mat& image) const;

    static httplib::Server::Handler handleOptionsRequest();
    httplib::Server::Handler handleImageRequest() const;
    static httplib::Server::Handler handleCors();
    static httplib::Server::ExceptionHandler handleException();
public:
    Handler(const std::string& model_path, const float& confidence_threshold, const float& NMS_threshold);
    ~Handler();
    void listen(const std::string& address, int port);
};
