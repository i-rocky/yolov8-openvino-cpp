//
// Created by Rasel Rana Rocky on 11/2/24.
//

#include "handler.h"

#include <opencv2/imgcodecs.hpp>

#include "timer.h"

Handler::Handler(const std::string& model_path, const float& confidence_threshold, const float& NMS_threshold)
{
    inference_ = new yolo::Inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
    mutex_ = new std::mutex();

    server_.Options("/", handleOptionsRequest());
    server_.Get("/", handleImageRequest());
    server_.set_post_routing_handler(handleCors());
    server_.set_exception_handler(handleException());
}

Handler::~Handler()
{
    delete inference_;
}

bool Handler::handles(const std::string& mime_type) const
{
    return std::any_of(mime_types_.begin(), mime_types_.end(), [&](const std::string& type)
    {
        return type == mime_type;
    });
}

void Handler::split_url(const std::string& url, std::string& host, std::string& path)
{
    const size_t path_start = url.find('/', url.find("://") + 3);

    if (path_start == std::string::npos) {
        host = url;
        path = "/";
    } else {
        host = url.substr(0, path_start);
        path = url.substr(path_start);
    }
}

httplib::Server::Handler Handler::handleOptionsRequest()
{
    return [&](const httplib::Request& req, httplib::Response& res)
    {
        res.set_content("OK", "text/plain");
    };
}

httplib::Server::Handler Handler::handleImageRequest() const
{
    return [&](const httplib::Request& req, httplib::Response& res)
    {
        const auto url = req.get_param_value("q");

        std::string host, path;
        split_url(url, host, path);

        httplib::Client client(host);
        auto result = client.Get(path);

        std::cout << "Status: " << result->status << " Content-Type: " << result->get_header_value("Content-Type") << std::endl;
        bool should_handle = handles(result->get_header_value("Content-Type"));
        if (!should_handle)
        {
            res.set_content(result->body, result->get_header_value("Content-Type"));
            return;
        }

        std::cout << "Handling: " << result->get_header_value("Content-Type") << std::endl;

        const std::vector<uchar> image_data( result->body.begin(), result->body.end());
        cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);

        if (image.empty())
        {
            std::cerr << "ERROR: image is empty" << std::endl;
            res.set_content(result->body, result->get_header_value("Content-Type"));
            return;
        }

        if (image.rows < 50 || image.cols < 50)
        {
            std::cerr << "ERROR: image is too small" << std::endl;
            res.set_content(result->body, result->get_header_value("Content-Type"));
            return;
        }

        apply_blur(image);

        const std::string full_image_name = "debug-_image_.jpg";
        cv::imwrite(full_image_name, image);

        res.set_file_content(full_image_name);
    };
}

void Handler::apply_blur(cv::Mat& image) const
{
    mutex_->lock();
    const auto time_start = Timer::get_time();
    inference_->RunInference(image);
    Timer::print_time(time_start, "Total time");
    mutex_->unlock();
}

httplib::Server::ExceptionHandler Handler::handleException()
{
    return [&](const httplib::Request& req, httplib::Response& res, const std::exception_ptr& ep)
    {
        try
        {
            rethrow_exception(ep);
        } catch (const std::exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            // print_stacktrace();
            std::cerr << "Stacktrace:" << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception" << std::endl;
        }
        const auto fmt = "<h1>Error 500</h1><p>%s</p>";
        res.set_content(fmt, "text/html");
        res.status = httplib::StatusCode::InternalServerError_500;
    };
}

void Handler::listen(const std::string& address, const int port)
{
    server_.listen(address, port);
}

httplib::Server::Handler Handler::handleCors()
{
    return [&](const httplib::Request& req, httplib::Response& res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Max-Age", "1728000");
    };
}
