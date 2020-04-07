//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_PROCESS_RGB_TO_HSV_HPP
#define HISTOGRAM_PROJECT_PROCESS_RGB_TO_HSV_HPP

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

namespace process {

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgNormalized;


    void processRBG_to_HSV(const std::vector<float4> &inputImg, // Input image
                           uint imgWidth, uint imgHeight, // Image size
                           std::vector<float4> &output);

    __device__ float4 fHSV_from_RGB(float r, float g, float b, float a);


    __device__ bool RGBisNormalized(float r, float g, float b, float a);


    __global__ void
    RGB_to_HSV(size_t imgWidth, size_t imgHeight, float4 *output);

}
#endif //HISTOGRAM_PROJECT_PROCESS_HPP
