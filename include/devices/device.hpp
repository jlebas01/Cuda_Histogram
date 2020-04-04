//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_DEVICE_HPP
#define HISTOGRAM_PROJECT_DEVICE_HPP

#include <cuda_runtime.h>
#include <vector_types.h>

namespace device {

    __device__ float4 fHSV_from_RGB(float r, float g, float b);

    __device__ float4 fRGB_from_HSV(float h, float s, float v);

    __device__ float clip(float n, float lower, float upper);

    __device__ float4 normalizeRGB(float r, float g, float b, float o);

}

#endif //HISTOGRAM_PROJECT_DEVICE_HPP
