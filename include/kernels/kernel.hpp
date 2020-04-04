//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_KERNEL_HPP
#define HISTOGRAM_PROJECT_KERNEL_HPP

#include <cuda_runtime.h>
#include <vector_types.h>

namespace kernel {

    texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texInput;

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgNormalized;

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgHSV;

    __global__ void
    normalizePixel(const size_t imgWidth, const size_t imgHeight, float4 *output);

    __global__ void
    RGB_to_HSV(const size_t imgWidth, const size_t imgHeight, float4 *output);

    __global__ void
    HSV_to_RGB(const size_t imgWidth, const size_t imgHeight, uchar4 *output);

}

#endif //HISTOGRAM_PROJECT_KERNEL_HPP
