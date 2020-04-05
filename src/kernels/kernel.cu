//
// Created by jlebas01 on 04/04/2020.
//

#include <kernels/kernel.hpp>


namespace kernel {

    __device__ float4 fHSV_from_RGB(float r, float g, float b) {
        float M = 0.0f, m = 0.0f, c = 0.0f;
        float4 HSVcolor = make_float4(0.f, 0.f, 0.f, 255.0f); //x : Hue, y : Saturation, z : Value, w : Opacity
        M = fmax(r, fmax(g, b));
        m = fmin(r, fmin(g, b));
        c = M - m;
        HSVcolor.z = M;
        if (c != 0.0f) {
            if (M == r) {
                HSVcolor.x = fmod(((g - b) / c), 6.0f);
            } else if (M == g) {
                HSVcolor.x = (b - r) / c + 2.0f;
            } else /*if(M==b)*/
            {
                HSVcolor.x = (r - g) / c + 4.0f;
            }
            HSVcolor.x *= 60.0f;
            HSVcolor.y = c / HSVcolor.z;
        }
        //}
        return HSVcolor;
    }

    __device__ float4 fRGB_from_HSV(float h, float s, float v) {
        float c = 0.0f, m = 0.0f, x = 0.0f;
        float4 color = make_float4(0.f, 0.f, 0.f, 1.0f);
        // if (Hsv_IsValid(h, s, v) == true) {
        c = v * s;
        x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
        m = v - c;
        if (h >= 0.0f && h < 60.0) {
            color = make_float4(c + m, x + m, m, 1.0f);
        } else if (h >= 60.0f && h < 120.0f) {
            color = make_float4(x + m, c + m, m, 1.0f);
        } else if (h >= 120.0f && h < 180.0f) {
            color = make_float4(m, c + m, x + m, 1.0f);
        } else if (h >= 180.0f && h < 240.0f) {
            color = make_float4(m, x + m, c + m, 1.0f);
        } else if (h >= 240.0f && h < 300.0f) {
            color = make_float4(x + m, m, c + m, 1.0f);
        } else if (h >= 300.0f && h < 360.0f) {
            color = make_float4(c + m, m, x + m, 1.0f);
        } else {
            color = make_float4(m, m, m, 1.0f);
        }
        //  }
        return color;
    }

    __device__ float clip(float n, float lower, float upper) {
        return fmax(lower, fmin(n, upper));
    }

    __device__ float4 normalizeRGB(float r, float g, float b, float o) {
        float4 RGBColorNormalized = make_float4(r / 255.f, g / 255.f, b / 255.f, o / 255.f);
        return RGBColorNormalized;
    }

    __global__ void
    normalizePixel(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);
        //uchar4 imgInput = make_uchar4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 RGBcolorNomalized = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            //uchar4 imgInput =  tex2D(texInput, idx, idy);

            uchar4 imgInput = make_uchar4(0.0f, 0.0f, 0.0f, 0.0f);

            RGBcolorNomalized = normalizeRGB(imgInput.x, imgInput.y, imgInput.z, imgInput.w);

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

            //float4 imgNormalized = tex2D(ImgNormalized, idx, idy);

            float4 imgNormalized = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            HSVColor = fHSV_from_RGB(imgNormalized.x, imgNormalized.y, imgNormalized.z);

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

            //float4 imgHSV = tex2D<float4>(ImgHSV, idx, idy);

            float4 imgHSV =make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            RGBColor = fRGB_from_HSV(imgHSV.x, imgHSV.y, imgHSV.z);

            const uint32_t idOut = idy * imgWidth + idx;
            output[idOut].x = static_cast<uint8_t>(RGBColor.x*255.f);
            output[idOut].y = static_cast<uint8_t>(RGBColor.y*255.f);
            output[idOut].z = static_cast<uint8_t>(RGBColor.z*255.f);
            output[idOut].w = static_cast<uint8_t>(RGBColor.w*255.f);
        }
    }

}