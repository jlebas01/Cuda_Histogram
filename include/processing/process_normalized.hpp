//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_PROCESS_NORMALIZED_HPP
#define HISTOGRAM_PROJECT_PROCESS_NORMALIZED_HPP

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

namespace process {

    texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texInput;

    void processNormalizer(const std::vector<uchar4> &inputImg, // Input image
                           uint imgWidth, uint imgHeight, // Image size
                           std::vector<float4> &output);

    __device__ float4 normalizeRGB(float r, float g, float b, float o);

    __global__ void
    normalizePixel(size_t imgWidth, size_t imgHeight, float4 *output);
}
#endif //HISTOGRAM_PROJECT_PROCESS_HPP
