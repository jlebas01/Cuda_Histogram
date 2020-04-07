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

namespace process {

    texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texInput;

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgNormalized;

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgHSV;

    void process(const std::vector<uchar4> &inputImg, // Input image
                 uint imgWidth, uint imgHeight, // Image size
                 const std::vector<uchar4> &resultCPU, // Just for comparison
                 std::vector<uchar4> &output // Output image
    );

    void processHSV_to_RGB(const std::vector<float4> &inputImg, // Input image
                           uint imgWidth, uint imgHeight, // Image size
                           std::vector<uchar4> &output);

    void processRBG_to_HSV(const std::vector<float4> &inputImg, // Input image
                           uint imgWidth, uint imgHeight, // Image size
                           std::vector<float4> &output);

    void processNormalizer(const std::vector<uchar4> &inputImg, // Input image
                           uint imgWidth, uint imgHeight, // Image size
                           std::vector<uchar4> &output);
    __device__ float4 fHSV_from_RGB(float r, float g, float b, float a);

    __device__ float4 fRGB_from_HSV(float h, float s, float v, float a);

    __device__ float4 normalizeRGB(float r, float g, float b, float o);

    __device__ float clip(float n, float lower, float upper);

    __device__ bool RGBisNormalized(float r, float g, float b, float a);

    __device__ bool HSVisNormalized(float h, float s, float v, float a);


    __global__ void
    normalizePixel(size_t imgWidth, size_t imgHeight, float4 *output);

    __global__ void
    RGB_to_HSV(size_t imgWidth, size_t imgHeight, float4 *output);

    __global__ void
    HSV_to_RGB(size_t imgWidth, size_t imgHeight, uchar4 *output);
}
#endif //HISTOGRAM_PROJECT_PROCESS_HPP
