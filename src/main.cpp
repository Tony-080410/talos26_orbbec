#include "spdlog_hook.hpp"
#include <Eigen/Dense>
#include <oneapi/tbb/parallel_for.h>

int main()
{
    hook_cstream();
    std::cout << "This is an info message redirected to spdlog.";
    std::cerr << "This is an error message redirected to spdlog.";
    spdlog::info("This is a direct spdlog info message.");
    spdlog::error("This is a direct spdlog error message.");
    Eigen::MatrixXd mat(2,2);
    oneapi::tbb::parallel_for(0, 2, [&](int i){
        for(int j = 0; j < 2; ++j)
            mat(i,j) = i + j;
    });
    std::async()
    return 0;
}