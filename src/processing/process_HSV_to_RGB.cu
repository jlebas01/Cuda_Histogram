//
// Created by jlebas01 on 04/04/2020.
//

#include <iostream>

#include <utils/tools.hpp>
#include <utils/common.hpp>

#include <processing/process_HSV_to_RGB.hpp>

#include <chrono/chronoGPU.hpp>


namespace process {

#define SIZE_HISTO 256
#define SIZE_CDF SIZE_HISTO * 2

    __global__ void calculateHistogram(unsigned int *Histogram, size_t width, size_t height) {

        const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;

        uint32_t gridBlockDimX = gridDim.x * blockDim.x;
        uint32_t gridBlockDimY = gridDim.y * blockDim.y;

        for (uint32_t y = tidY; y < width; y += gridBlockDimY) {
            for (uint32_t x = tidX; x < height; x += gridBlockDimX) {
                float4 HSVColor = tex2D<float4>(ImgHSV, (float) (y), (float) (x));
                atomicAdd(&Histogram[static_cast<int>(HSVColor.z * 255.0f)], 1);
            }
        }
    }


    __global__ void calcCDFnormalized(const unsigned int *histo, float *cdf, size_t width, size_t height) {
        for (int i = 0; i <= threadIdx.x; i++) {
            cdf[threadIdx.x] += (float) histo[i];
        }
        cdf[threadIdx.x] *= 1.0f / float((width * height));
    }

    __global__ void calcCDF(float *cdf, unsigned int *histo, int imageWidth, int imageHeight, int length) {

        __shared__ float partialScan[SIZE_CDF];
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < SIZE_CDF && i < 256) {
            partialScan[i] = (float) histo[i] / (float) (imageWidth * imageHeight);

        }
        __syncthreads();

        for (unsigned int stride = 1; stride <= SIZE_HISTO; stride *= 2) {
            unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
            if (index < SIZE_CDF && index < length)
                partialScan[index] += partialScan[index - stride];
            __syncthreads();
        }

        for (unsigned int stride = SIZE_HISTO / 2; stride > 0; stride /= 2) {
            __syncthreads();
            unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
            if (index + stride < SIZE_CDF && index + stride < length) {
                partialScan[index + stride] += partialScan[index];
            }
        }

        __syncthreads();
        if (i < SIZE_CDF && i < 256) {
            cdf[i] += partialScan[threadIdx.x];
        }
    }

    __device__ bool HSVisNormalized(float h, float s, float v, float a) {
        return (0.0f <= h and h <= 360.0f) and (0.0f <= s and s <= 1.0f) and (0.0f <= v and v <= 1.0f) and
               (0.0f <= a and a <= 1.0f);
    }

    __device__ float4 fRGB_from_HSV(float h, float s, float v, float a) {
        float c = 0.0f, m = 0.0f, x = 0.0f;
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (!HSVisNormalized(h, s, v, a)) {
            printf("HSVA(%f, %f, %f, %f) isn't normalized, file : %s ; line : %d\n", h, s, v, a, __FILE__, __LINE__);
            return make_float4(0.f, 0.f, 0.f, 0.f);
        }
        c = v * s;
        x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
        m = v - c;
        if (h >= 0.0f && h < 60.0) {
            r = c + m;
            g = x + m;
            b = m;
        } else if (h >= 60.0f && h < 120.0f) {
            r = x + m;
            g = c + m;
            b = m;
        } else if (h >= 120.0f && h < 180.0f) {
            r = m;
            g = c + m;
            b = x + m;
        } else if (h >= 180.0f && h < 240.0f) {
            r = m;
            g = x + m;
            b = c + m;
        } else if (h >= 240.0f && h < 300.0f) {
            r = x + m;
            g = m;
            b = c + m;
        } else if (h >= 300.0f && h < 360.0f) {
            r = c + m;
            g = m;
            b = x + m;
        } else {
            r = m;
            g = m;
            b = m;
        }
        return make_float4(r, g, b, a);;
    }

    __global__ void
    HSV_to_RGB(const size_t imgWidth, const size_t imgHeight, const float *cdf, uchar4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        uint32_t gridBlockDimX = gridDim.x * blockDim.x;
        uint32_t gridBlockDimY = gridDim.y * blockDim.y;

        float4 RGBColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {


                for (uint32_t x = idy; x < imgHeight; x += gridBlockDimX) {
                    for (uint32_t y = idx; y < imgWidth; y += gridBlockDimY) {

                    float4 imgHSV = tex2D<float4>(ImgHSV, float(y), float(x));

                    float cdfmin = cdf[0];
                    float z = ((cdf[static_cast<int>(imgHSV.z * 255.0f)] - cdfmin) / (1.0F - cdfmin));

                    RGBColor = fRGB_from_HSV(imgHSV.x, imgHSV.y, z, imgHSV.w);

                    const uint32_t idOut = x * imgWidth + y;
                    output[idOut].x = static_cast<uint8_t>(RGBColor.x * 255.f);
                    output[idOut].y = static_cast<uint8_t>(RGBColor.y * 255.f);
                    output[idOut].z = static_cast<uint8_t>(RGBColor.z * 255.f);
                    output[idOut].w = static_cast<uint8_t>(RGBColor.w * 255.f);
                }
            }
        }
    }

    void processHSV_to_RGB(const std::vector<float4> &inputImg, // Input image
                           const uint imgWidth, const uint imgHeight, // Image size
                           std::vector<uchar4> &output) {

        float4 *dev_inputF4 = nullptr;
        uchar4 *dev_outputU4 = nullptr;
        float *cdf = nullptr;
        unsigned int *Histogram = nullptr;

        chrono::ChronoGPU chrGPU;

        const size_t ImgSize = imgHeight * imgWidth;
        size_t ImgBytes = ImgSize * sizeof(uchar4);


        size_t width = imgWidth;
        size_t height = imgHeight;
        size_t widthBytes = width * sizeof(float4);

        size_t offset = 0;
        size_t pitch;
        size_t spitch = widthBytes;

        ImgHSV.addressMode[0] = cudaAddressModeBorder;
        ImgHSV.addressMode[1] = cudaAddressModeBorder;
        ImgHSV.filterMode = cudaFilterModePoint;
        ImgHSV.normalized = false;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        const uint32_t blockSizeX = (imgWidth % 32 == 0 ? imgWidth / 32 : imgWidth / 32 + 1);
        const uint32_t blockSizeY = (imgHeight % 32 == 0 ? imgHeight / 32 : imgHeight / 32 + 1);


        /*********************************************************************************/
        std::cout << "Allocating arrays: " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMalloc((void **) &dev_outputU4, ImgBytes));
        HANDLE_ERROR(cudaMalloc((void **) &cdf, SIZE_HISTO * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **) &Histogram, SIZE_HISTO * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMallocPitch((void **) &dev_inputF4, &pitch, widthBytes, height));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Copy data from host to devices (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        chrGPU.start();
        HANDLE_ERROR(
                cudaMemcpy2D((void **) dev_inputF4, pitch, (void **) inputImg.data(), spitch, widthBytes, height,
                             cudaMemcpyHostToDevice));
        chrGPU.stop();
        std::cout << "Bind 2D Texture with devices Input " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaBindTexture2D(&offset, ImgHSV, dev_inputF4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();

        calculateHistogram <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>>(Histogram, width, height);
        calcCDFnormalized<<< 1, 256 >>>(Histogram, cdf, width, height);
        HSV_to_RGB <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>>(width, height, cdf, dev_outputU4);
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        cudaDeviceSynchronize();

        /*********************************************************************************/
        std::cout << "Copy data from devices to host (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_outputU4, ImgBytes, cudaMemcpyDeviceToHost));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /***********************************************outputArray**********************************/

        /**FREE**AND**UNBIND**/
        cudaUnbindTexture(ImgHSV);

        cudaFree(dev_inputF4);
        cudaFree(dev_outputU4);
    }

}