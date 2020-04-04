//
// Created by jlebas01 on 04/04/2020.
//

#include <kernels/kernel.hpp>
#include <devices/device.hpp>

namespace kernel {
    __global__ void
    normalizePixel(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        float4 RGBcolorNomalized = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            uchar4 imgInput = tex2D(texInput, idx, idy);

            RGBcolorNomalized = device::normalizeRGB(imgInput.x, imgInput.y, imgInput.z, imgInput.w);

            const uint32_t idOut = idy * imgWidth + idx;
            output[idOut].x = static_cast<uint8_t>(RGBcolorNomalized.x);
            output[idOut].y = static_cast<uint8_t>(RGBcolorNomalized.y);
            output[idOut].z = static_cast<uint8_t>(RGBcolorNomalized.z);
            output[idOut].w = static_cast<uint8_t>(RGBcolorNomalized.w);
        }
    }

    __global__ void
    RGB_to_HSV(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        float4 HSVColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            float4 imgNormalized = tex2D(ImgNormalized, idx, idy);

            HSVColor = device::fHSV_from_RGB(imgNormalized.x, imgNormalized.y, imgNormalized.z);

            const uint32_t idOut = idy * imgWidth + idx;
            output[idOut].x = static_cast<float>(HSVColor.x);
            output[idOut].y = static_cast<float>(HSVColor.y);
            output[idOut].z = static_cast<float>(HSVColor.z);
            output[idOut].w = static_cast<float>(HSVColor.w);
        }
    }

    __global__ void
    HSV_to_RGB(const size_t imgWidth, const size_t imgHeight, uchar4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        float4 RGBColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            float4 imgHSV = tex2D(ImgHSV, idx, idy);

            RGBColor = device::fRGB_from_HSV(imgHSV.x, imgHSV.y, imgHSV.z);

            const uint32_t idOut = idy * imgWidth + idx;
            output[idOut].x = static_cast<uint8_t>(RGBColor.x*255.f);
            output[idOut].y = static_cast<uint8_t>(RGBColor.y*255.f);
            output[idOut].z = static_cast<uint8_t>(RGBColor.z*255.f);
            output[idOut].w = static_cast<uint8_t>(RGBColor.w*255.f);
        }
    }

}