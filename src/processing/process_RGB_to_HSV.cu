//
// Created by jlebas01 on 04/04/2020.
//

#include <iostream>

#include <utils/tools.hpp>
#include <utils/common.hpp>

#include <processing/process_RGB_to_HSV.hpp>

#include <chrono/chronoGPU.hpp>

namespace process {

    __device__ bool RGBisNormalized(float r, float g, float b, float a) {
        return (0.0f <= r and r <= 1.0f) and (0.0f <= g and g <= 1.0f) and (0.0f <= b and b <= 1.0f) and
               (0.0f <= a and a <= 1.0f);
    }

    __device__ float4 fHSV_from_RGB(float r, float g, float b, float a) {
        float M = 0.0f, m = 0.0f, c = 0.0f;
        float h = 0.0f, s = 0.0f, v = 0.0f;
        if (!RGBisNormalized(r, g, b, a)) {
            printf("RGBA(%f, %f, %f, %f) isn't noramlized, file : %s ; line : %d\n", r, g, b, a, __FILE__, __LINE__);
            return make_float4(0.f, 0.f, 0.f, 0.f);
        }

        M = fmax(r, fmax(g, b));
        m = fmin(r, fmin(g, b));
        c = M - m;
        v = M;
        if (c != 0.0f) {
            if (M == r) {
                h = fabs(fmod(60.0f * ((g - b) / c) + 360.0f, 360.0f));
            } else if (M == g) {
                h = 60.0f * ((b - r) / c) + 120.0f;
            } else /*if(M==b)*/
            {
                h = 60.0f * ((r - g) / c) + 240.0f;
            }
            s = c / v;
        }
        return make_float4(h, s, v, a); //x : Hue, y : Saturation, z : Value, w : Opacity;
    }

    __global__ void
    RGB_to_HSV(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        uint32_t gridBlockDimX = gridDim.x * blockDim.x;
        uint32_t gridBlockDimY = gridDim.y * blockDim.y;

        float4 HSVColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            for (uint32_t x = idy; x < imgHeight; x += gridBlockDimX) {
                for (uint32_t y = idx; y < imgWidth; y += gridBlockDimY) {

                    float4 imgNormalized = tex2D<float4>(ImgNormalized, float(idx + 0.5f), float(idy + 0.5f));

                    HSVColor = fHSV_from_RGB(imgNormalized.x, imgNormalized.y, imgNormalized.z, imgNormalized.w);

                    const uint32_t idOut = x * imgWidth + y;
                    output[idOut] = HSVColor;
                }
            }
        }
    }

    void processRBG_to_HSV(const std::vector<float4> &inputImg, // Input image
                           const uint imgWidth, const uint imgHeight, // Image size
                           std::vector<float4> &output) {

        float4 *dev_inputF4 = nullptr;
        float4 *dev_outputF4 = nullptr;

        chrono::ChronoGPU chrGPU;

        const size_t ImgSize = imgHeight * imgWidth;
        size_t ImgBytes = ImgSize * sizeof(float4);

        size_t width = imgWidth;
        size_t height = imgHeight;
        size_t widthBytes = width * sizeof(float4);


        size_t offset = 0;
        size_t pitch;
        size_t spitch = widthBytes;

        ImgNormalized.addressMode[0] = cudaAddressModeBorder;
        ImgNormalized.addressMode[1] = cudaAddressModeBorder;
        ImgNormalized.filterMode = cudaFilterModePoint;
        ImgNormalized.normalized = false;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        const uint32_t blockSizeX = (imgWidth % 32 == 0 ? imgWidth / 32 : imgWidth / 32 + 1);
        const uint32_t blockSizeY = (imgHeight % 32 == 0 ? imgHeight / 32 : imgHeight / 32 + 1);


        /*********************************************************************************/
        std::cout << "Allocating arrays: " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMalloc((void **) &dev_outputF4, ImgBytes));
        HANDLE_ERROR(cudaMallocPitch((void **) &dev_inputF4, &pitch, widthBytes, height));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Copy data from host to devices (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy2D((void **) dev_inputF4, pitch, (void **) inputImg.data(), spitch, widthBytes, height,
                                  cudaMemcpyHostToDevice));
        chrGPU.stop();
        std::cout << "Bind 2D Texture with devices Input " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaBindTexture2D(&offset, ImgNormalized, dev_inputF4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();
        RGB_to_HSV <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>>(width, height, dev_outputF4);
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        cudaDeviceSynchronize();

        /*********************************************************************************/
        std::cout << "Copy data from devices to host (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_outputF4, ImgBytes, cudaMemcpyDeviceToHost));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /***********************************************outputArray**********************************/

        /**FREE**AND**UNBIND**/
        cudaUnbindTexture(ImgNormalized);

        cudaFree(dev_inputF4);
        cudaFree(dev_outputF4);
    }
}