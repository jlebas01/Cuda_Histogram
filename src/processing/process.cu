//
// Created by jlebas01 on 04/04/2020.
//

#include <iostream>

#include <utils/tools.hpp>
#include <utils/common.hpp>

#include <processing/process.hpp>

#include <chrono/chronoGPU.hpp>

#include <kernels/kernel.hpp>


namespace process {

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

        /*kernel::texInput.addressMode[0] = cudaAddressModeBorder;
        kernel::texInput.addressMode[1] = cudaAddressModeBorder;
        kernel::texInput.filterMode = cudaFilterModePoint;
        kernel::texInput.normalized = false;*/

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
       /* for (auto &it : inputImg){
            std::cout << static_cast<int>(it.x) << " " << static_cast<int>(it.y) << " " << static_cast<int>(it.z) << std::endl;
        }*/
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy2D((void **) dev_inputU4, pitch, (void **) inputImg.data(), spitch, widthBytes, height,
                                  cudaMemcpyHostToDevice));
        chrGPU.stop();
        std::cout << "Bind 2D Texture with devices Input " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaBindTexture2D(&offset, kernel::texInput, dev_inputU4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();
        kernel::normalizePixel <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>> (width, height, dev_outputF4);
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
        cudaUnbindTexture(kernel::texInput);

        cudaFree(dev_inputU4);
        cudaFree(dev_outputF4);
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

        kernel::ImgNormalized.addressMode[0] = cudaAddressModeBorder;
        kernel::ImgNormalized.addressMode[1] = cudaAddressModeBorder;
        kernel::ImgNormalized.filterMode = cudaFilterModePoint;
        kernel::ImgNormalized.normalized = false;

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
        HANDLE_ERROR(cudaBindTexture2D(&offset, kernel::ImgNormalized, dev_inputF4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();
        kernel::RGB_to_HSV <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>> (width, height, dev_outputF4);
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
        cudaUnbindTexture(kernel::ImgNormalized);

        cudaFree(dev_inputF4);
        cudaFree(dev_outputF4);
    }

    void processHSV_to_RGB(const std::vector<float4> &inputImg, // Input image
                           const uint imgWidth, const uint imgHeight, // Image size
                           std::vector<uchar4> &output) {

        float4 *dev_inputF4 = nullptr;
        uchar4 *dev_outputU4 = nullptr;

        chrono::ChronoGPU chrGPU;

        const size_t ImgSize = imgHeight * imgWidth;
        size_t ImgBytes = ImgSize * sizeof(uchar4);

        size_t width = imgWidth;
        size_t height = imgHeight;
        size_t widthBytes = width * sizeof(float4);


        size_t offset = 0;
        size_t pitch;
        size_t spitch = widthBytes;

        kernel::ImgHSV.addressMode[0] = cudaAddressModeBorder;
        kernel::ImgHSV.addressMode[1] = cudaAddressModeBorder;
        kernel::ImgHSV.filterMode = cudaFilterModePoint;
        kernel::ImgHSV.normalized = false;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        const uint32_t blockSizeX = (imgWidth % 32 == 0 ? imgWidth / 32 : imgWidth / 32 + 1);
        const uint32_t blockSizeY = (imgHeight % 32 == 0 ? imgHeight / 32 : imgHeight / 32 + 1);


        /*********************************************************************************/
        std::cout << "Allocating arrays: " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMalloc((void **) &dev_outputU4, ImgBytes));
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
        HANDLE_ERROR(cudaBindTexture2D(&offset, kernel::ImgHSV, dev_inputF4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();
        kernel::HSV_to_RGB <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>> (width, height, dev_outputU4);
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
        cudaUnbindTexture(kernel::ImgHSV);

        cudaFree(dev_inputF4);
        cudaFree(dev_outputU4);
    }

    void process(const std::vector<uchar4> &inputImg, // Input image
                 const uint imgWidth, const uint imgHeight, // Image size
                 const std::vector<uchar4> &resultCPU, // Just for comparison
                 std::vector<uchar4> &output // Output image
    ) {

        std::vector<float4> outputF4(imgWidth*imgHeight);
        std::vector<float4> inputF4(imgWidth*imgHeight);

        processNormalizer(inputImg,imgWidth,imgHeight,outputF4);

        inputF4.swap(outputF4);

        outputF4.clear();

        processRBG_to_HSV(inputF4,imgWidth,imgHeight,outputF4);

        inputF4.clear();

        inputF4.swap(outputF4);

        outputF4.clear();

        processHSV_to_RGB(inputF4,imgWidth,imgHeight,output);

        //utils::compareImages(resultCPU, output);

    }
}