//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_PROCESS_HPP
#define HISTOGRAM_PROJECT_PROCESS_HPP

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

namespace process {

    void process(const std::vector<uchar4> &inputImg, // Input image
                 uint imgWidth, uint imgHeight, // Image size
                 const std::vector<uchar4> &resultCPU, // Just for comparison
                 std::vector<uchar4> &output // Output image
    );

}
#endif //HISTOGRAM_PROJECT_PROCESS_HPP
