/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.hpp
* Author: Maxime MARIA
*/

#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <common.hpp>

namespace IMAC {
    void studentJob(const std::vector <uchar4> &inputImg, // Input image
                    const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
                    const uint matSize, // Matrix size (width or height)
                    const std::vector <uchar4> &resultCPU, // Just for comparison
                    std::vector <uchar4> &output // Output image
    );

    std::ostream &operator<<(std::ostream &os, const uchar4 &c);

    void compareImages(const std::vector <uchar4> &a, const std::vector <uchar4> &b);

    texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texInput;

    __constant__ float mat[255];

    __device__ float clip(float n, float lower, float upper);

    __global__ void convGPUTexture2D(size_t pitch, const size_t imgWidth, const size_t imgHeight, const uint matSize,
                                     uchar4 *output);
    __device__ float4 Hsv_FromRgbF(float r, float g, float b);
}

#endif
