//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_PROCESS_HPP
#define HISTOGRAM_PROJECT_PROCESS_HPP

#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace process {
    void process(const std::vector<uchar4> &inputImg, // Input image
                 const uint imgWidth, const uint imgHeight, // Image size
                 std::vector<uchar4> &output // Output image
    );

    void processNormalizer(const std::vector<uchar4> &inputImg, // Input image
                           const uint imgWidth, const uint imgHeight, // Image size
                           std::vector<uchar4> &output);
}
#endif //HISTOGRAM_PROJECT_PROCESS_HPP
