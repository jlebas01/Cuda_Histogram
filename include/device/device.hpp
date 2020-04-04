//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_DEVICE_HPP
#define HISTOGRAM_PROJECT_DEVICE_HPP

#include <cuda_runtime.h>
#include <vector_types.h>

__device__ float4 Hsv_FromRgbF(float r, float g, float b);

__device__ float clip(float n, float lower, float upper);

#endif //HISTOGRAM_PROJECT_DEVICE_HPP
