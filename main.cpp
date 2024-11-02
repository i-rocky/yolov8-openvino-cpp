#include <iostream>
#include "handler.h"

void split_url(const std::string& url, std::string& host, std::string& path) {

}

int main(const int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    constexpr float confidence_threshold = 0.5;
    constexpr float NMS_threshold = 0.4;

    Handler handler(model_path, confidence_threshold, NMS_threshold);

    std::cout << "Server listening on port 8083" << std::endl;

    handler.listen("0.0.0.0", 8083);

    return 0;
}
