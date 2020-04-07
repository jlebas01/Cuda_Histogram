#include <iostream>

#include <utils/tools.hpp>
#include <utils/common.hpp>

#include <processing/process_normalized.hpp>

#include <chrono/chronoGPU.hpp>

namespace process {

    __device__ float4 normalizeRGB(float r, float g, float b, float o) {
        float4 RGBColorNormalized = make_float4(r / 255.f, g / 255.f, b / 255.f, o / 255.f);
        return RGBColorNormalized;
    }

    __global__ void
    normalizePixel(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        uint32_t gridBlockDimX = gridDim.x * blockDim.x;
        uint32_t gridBlockDimY = gridDim.y * blockDim.y;

        float4 RGBcolorNomalized = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            for (uint32_t x = idy; x < imgHeight; x += gridBlockDimX) {
                for (uint32_t y = idx; y < imgWidth; y += gridBlockDimY) {

                    uchar4 imgInput = tex2D<uchar4>(texInput, float(y), float(x));

                    RGBcolorNomalized = normalizeRGB(float(imgInput.x), float(imgInput.y), float(imgInput.z),
                                                     float(imgInput.w));

                    const uint32_t idOut = x * imgWidth + y;
                    output[idOut] = RGBcolorNomalized;
                }
            }
        }
    }

    void processNormalizer(const std::vector<uchar4> &inputImg, // Input image
                           const uint imgWidth, const uint imgHeight, // Image size
                           std::vector<float4> &output) {

        uchar4 *dev_inputU4 = nullptr;
        float4 *dev_outputF4 = nullptr;

        chrono::ChronoGPU chrGPU;

        const size_t ImgSize = imgHeight * imgWidth;
        size_t ImgBytes = ImgSize * sizeof(float4);

        size_t width = imgWidth;
        size_t height = imgHeight;
        size_t widthBytes = width * sizeof(uchar4);


        size_t offset = 0;
        size_t pitch;
        size_t spitch = widthBytes;

        texInput.addressMode[0] = cudaAddressModeBorder;
        texInput.addressMode[1] = cudaAddressModeBorder;
        texInput.filterMode = cudaFilterModePoint;
        texInput.normalized = false;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

        const uint32_t blockSizeX = (imgWidth % 32 == 0 ? imgWidth / 32 : imgWidth / 32 + 1);
        const uint32_t blockSizeY = (imgHeight % 32 == 0 ? imgHeight / 32 : imgHeight / 32 + 1);


        /*********************************************************************************/
        std::cout << "Allocating arrays: " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMalloc((void **) &dev_outputF4, ImgBytes));
        HANDLE_ERROR(cudaMallocPitch((void **) &dev_inputU4, &pitch, widthBytes, height));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Copy data from host to devices (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        /*for (auto &it : inputImg){
            std::cout << static_cast<int>(it.x) << " " << static_cast<int>(it.y) << " " << static_cast<int>(it.z) << std::endl;
        }*/
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy2D((void **) dev_inputU4, pitch, (void **) inputImg.data(), spitch, widthBytes, height,
                                  cudaMemcpyHostToDevice));
        chrGPU.stop();
        std::cout << "Bind 2D Texture with devices Input " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaBindTexture2D(&offset, texInput, dev_inputU4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << " height : " << height << std::endl;
        chrGPU.start();
        normalizePixel <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>>(width, height, dev_outputF4);
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        cudaDeviceSynchronize();

        /*********************************************************************************/
        std::cout << "Copy data from devices to host (output arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        //chrGPU.start();
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_outputF4, ImgBytes, cudaMemcpyDeviceToHost));
        //chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /***********************************************outputArray**********************************/

        /**FREE**AND**UNBIND**/
        cudaUnbindTexture(texInput);

        cudaFree(dev_inputU4);
        cudaFree(dev_outputF4);
    }

}