/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Jérémie LE BASTARD
*/

#include <histogram_normalizer.hpp>
#include <chrono/chronoGPU.hpp>
#include <devices/device.hpp>
#include <utils/tools.hpp>
#include <vector>

namespace IMAC {

    __global__ void convGPUTexture2D(const size_t imgWidth, const size_t imgHeight, const uint matSize,
                                     uchar4 *output) {
        int idx = (blockIdx.x * blockDim.x + threadIdx.x);
        int idy = (blockIdx.y * blockDim.y + threadIdx.y);

        if (idx < imgWidth && idy < imgHeight) {
            float3 sum = make_float3(0.f, 0.f, 0.f);
            for (uint j = 0; j < matSize; ++j) {
                for (uint i = 0; i < matSize; ++i) {
                    int dX = idx + i - matSize / 2;
                    int dY = idy + j - matSize / 2;

                    // Handle borders
                    if (dX < 0)
                        dX = 0;

                    if (dX >= imgWidth)
                        dX = imgWidth - 1;

                    if (dY < 0)
                        dY = 0;

                    if (dY >= imgHeight)
                        dY = imgHeight - 1;

                    //const int idMat = j * matSize + i;

                    uchar4 input = tex2D(texInput, dX, dY);

                    sum.x += (float) input.x;
                    sum.y += (float) input.y;
                    sum.z += (float) input.z;
                }
            }
            const int idOut = idy * imgWidth + idx;
            output[idOut].x = (uchar) device::clip(sum.x, 0.f, 255.f);
            output[idOut].y = (uchar) device::clip(sum.y, 0.f, 255.f);
            output[idOut].z = (uchar) device::clip(sum.z, 0.f, 255.f);
            output[idOut].w = 255;
        }

    }

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
                    const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
                    const uint matSize, // Matrix size (width or height)
                    const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
    ) {


        uchar4 *dev_input = nullptr;
        uchar4 *dev_output = nullptr;

        chrono::ChronoGPU chrGPU;

        const size_t ImgSize = imgHeight * imgWidth;
        size_t ImgBytes = ImgSize * sizeof(uchar4);
        const size_t matBytes = matSize * matSize * sizeof(float);

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


        /*********************************************************************************/
        std::cout << "Allocating arrays: " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMalloc((void **) &dev_output, ImgBytes));
        HANDLE_ERROR(cudaMallocPitch((void **) &dev_input, &pitch, widthBytes, height));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Copy data from host to devices (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy2D((void **) dev_input, pitch, (void **) inputImg.data(), spitch, widthBytes, height,
                                  cudaMemcpyHostToDevice));
        chrGPU.stop();
        std::cout << "Put Matrix in devices's constant memory " << (matBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpyToSymbol(mat, matConv.data(), matBytes));
        chrGPU.stop();
        std::cout << "Bind 2D Texture with devices Input " << (ImgBytes >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaBindTexture2D(&offset, texInput, dev_input, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        const uint32_t blockSizeX = (imgWidth % 32 == 0 ? imgWidth / 32 : imgWidth / 32 + 1);
        const uint32_t blockSizeY = (imgHeight % 32 == 0 ? imgHeight / 32 : imgHeight / 32 + 1);

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();
        convGPUTexture2D <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>>(width, height, matSize, dev_output);
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        cudaDeviceSynchronize();

        /*********************************************************************************/
        std::cout << "Copy data from devices to host (input arrays) " << (ImgBytes >> 20) << " MB on Device"
                  << std::endl;
        chrGPU.start();
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, ImgBytes, cudaMemcpyDeviceToHost));
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /***********************************************outputArray**********************************/
        cudaUnbindTexture(texInput);
        cudaFree(dev_input);
        cudaFree(dev_output);

        utils::compareImages(resultCPU, output);

    }
}