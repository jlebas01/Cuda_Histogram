//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_KERNEL_HPP
#define HISTOGRAM_PROJECT_KERNEL_HPP

#include <vector_types.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>

#include <utils/common.hpp>


namespace kernel {

    __device__ float4 fHSV_from_RGB(float r, float g, float b, float a);

    __device__ float4 fRGB_from_HSV(float h, float s, float v, float a);

    __device__ float clip(float n, float lower, float upper);

    __device__ float4 normalizeRGB(float r, float g, float b, float o);

    texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texInput;

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgNormalized;

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgHSV;

    __global__ void
    normalizePixel(size_t imgWidth, size_t imgHeight, float4 *output);

    __global__ void
    RGB_to_HSV(size_t imgWidth, size_t imgHeight, float4 *output);

    __global__ void
    HSV_to_RGB(size_t imgWidth, size_t imgHeight, uchar4 *output);

}

#endif //HISTOGRAM_PROJECT_KERNEL_HPP
