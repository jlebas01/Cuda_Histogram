//
// Created by jlebas01 on 04/04/2020.
//

#include <iostream>

#include <utils/tools.hpp>
#include <utils/common.hpp>

#include <processing/process.hpp>

#include <chrono/chronoGPU.hpp>


namespace process {

    __device__ float clip(float n, float lower, float upper) {
        return fmax(lower, fmin(n, upper));
    }

    __device__ bool RGBisNormalized(float r, float g, float b, float a) {
        return (0.0f <= r and r <= 1.0f) and (0.0f <= g and g <= 1.0f) and (0.0f <= b and b <= 1.0f) and
               (0.0f <= a and a <= 1.0f);
    }

    __device__ bool HSVisNormalized(float h, float s, float v, float a) {
        return (0.0f <= h and h <= 360.0f) and (0.0f <= s and s <= 1.0f) and (0.0f <= v and v <= 1.0f) and
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
                h = fabs(fmod(60.0f*((g - b) / c)+360.0f, 360.0f));
            } else if (M == g) {
                h = 60.0f*((b - r) / c) + 120.0f;
            } else /*if(M==b)*/
            {
                h = 60.0f*((r - g) / c) + 240.0f;
            }
             s = c / v;
        }
        return make_float4(h,s,v, a); //x : Hue, y : Saturation, z : Value, w : Opacity;
    }

    __device__ float4 fRGB_from_HSV(float h, float s, float v, float a) {
        float c = 0.0f, m = 0.0f, x = 0.0f;
        float r = 0.0f, g = 0.0f ,b =0.0f;
        if (!HSVisNormalized(h, s, v,a)) {
            printf("HSVA(%f, %f, %f, %f) isn't normalized, file : %s ; line : %d\n", h, s, v, a, __FILE__, __LINE__);
            return make_float4(0.f, 0.f, 0.f, 0.f);
        }
        c = v * s;
        x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
        m = v - c;
        if (h >= 0.0f && h < 60.0) {
            r = c+m;
            g = x+m;
            b = m;
        } else if (h >= 60.0f && h < 120.0f) {
            r = x + m;
            g = c+m;
            b = m;
        } else if (h >= 120.0f && h < 180.0f) {
            r = m;
            g = c+m;
            b = x+m;
        } else if (h >= 180.0f && h < 240.0f) {
            r = m;
            g = x+m;
            b = c+m;
        } else if (h >= 240.0f && h < 300.0f) {
            r = x + m;
            g = m;
            b = c+m;
        } else if (h >= 300.0f && h < 360.0f) {
            r = c + m;
            g = m;
            b = x+m;
        } else {
            r = m;
            g = m;
            b = m;
        }
        return make_float4(r,g,b,a);;
    }


    __device__ float4 normalizeRGB(float r, float g, float b, float o) {
        float4 RGBColorNormalized = make_float4(r / 255.f, g / 255.f, b / 255.f, o / 255.f);
        return RGBColorNormalized;
    }


    __global__ void
    normalizePixel(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        float4 RGBcolorNomalized = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            uchar4 imgInput = tex2D<uchar4>(texInput, float(idx + 0.5f), float(idy + 0.5f));

            RGBcolorNomalized = normalizeRGB(float(imgInput.x), float(imgInput.y), float(imgInput.z),
                                             float(imgInput.w));

            //printf("RGBcolorNormalized : %f %f %f %f \n", RGBcolorNomalized.x, RGBcolorNomalized.y, RGBcolorNomalized.z, RGBcolorNomalized.w);

            const uint32_t idOut = idy * imgWidth + idx;
            output[idOut] = RGBcolorNomalized;
        }
    }

    __global__ void
    RGB_to_HSV(const size_t imgWidth, const size_t imgHeight, float4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        float4 HSVColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            float4 imgNormalized = tex2D(ImgNormalized, float(idx + 0.5f), float(idy + 0.5f));

            //printf("imgNormalized : %f %f %f %f \n", imgNormalized.x, imgNormalized.y, imgNormalized.z, imgNormalized.w);

            HSVColor = fHSV_from_RGB(imgNormalized.x, imgNormalized.y, imgNormalized.z, imgNormalized.w);

            //printf("HSVColor : %f %f %f %f \n", HSVColor.x, HSVColor.y, HSVColor.z, HSVColor.w);

            const uint32_t idOut = idy * imgWidth + idx;
            output[idOut] = HSVColor;
        }
    }

    __global__ void
    HSV_to_RGB(const size_t imgWidth, const size_t imgHeight, uchar4 *output) {
        uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
        uint32_t idy = (blockIdx.y * blockDim.y + threadIdx.y);

        float4 RGBColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (idx < imgWidth && idy < imgHeight) {

            float4 imgHSV = tex2D(ImgHSV, float(idx + 0.5f), float(idy + 0.5f));


            //printf("imgHSV : %f %f %f %f \n", imgHSV.x, imgHSV.y, imgHSV.z, imgHSV.w);

            RGBColor = fRGB_from_HSV(imgHSV.x, imgHSV.y, imgHSV.z, imgHSV.w);

            // printf("RGBcolor : %f %f %f %f \n", RGBColor.x, RGBColor.y, RGBColor.z, RGBColor.w);

            const uint32_t idOut = idy * imgWidth + idx;
            /*output[idOut].x = static_cast<uint8_t>(RGBColor.x);
            output[idOut].y = static_cast<uint8_t>(RGBColor.y);
            output[idOut].z = static_cast<uint8_t>(RGBColor.z);
            output[idOut].w = static_cast<uint8_t>(RGBColor.w);*/
            /*output[idOut].x = static_cast<uint8_t>(clip(RGBColor.x, 0.f, 255.f));
            output[idOut].y = static_cast<uint8_t>(clip(RGBColor.y, 0.f, 255.f));
            output[idOut].z = static_cast<uint8_t>(clip(RGBColor.z, 0.f, 255.f));
            output[idOut].w = static_cast<uint8_t>(clip(RGBColor.w, 0.f, 255.f));*/
            output[idOut].x = static_cast<uint8_t>(RGBColor.x * 255.f);
            output[idOut].y = static_cast<uint8_t>(RGBColor.y * 255.f);
            output[idOut].z = static_cast<uint8_t>(RGBColor.z * 255.f);
            output[idOut].w = static_cast<uint8_t>(RGBColor.w * 255.f);
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
        HANDLE_ERROR(cudaBindTexture2D(&offset, ImgHSV, dev_inputF4, channelDesc, width, height,
                                       pitch)); // pitch instead ImgBytes
        chrGPU.stop();
        std::cout << " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        /*********************************************************************************/

        /*********************************************************************************/
        std::cout << "Process on GPU -- Kernel " << std::endl;
        std::cout << "width : " << width << "height : " << height << std::endl;
        chrGPU.start();
        HSV_to_RGB <<< dim3(blockSizeX, blockSizeY), dim3(32, 32) >>>(width, height, dev_outputU4);
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

    void process(const std::vector<uchar4> &inputImg, // Input image
                 const uint imgWidth, const uint imgHeight, // Image size
                 const std::vector<uchar4> &resultCPU, // Just for comparison
                 std::vector<uchar4> &output // Output image
    ) {

        std::vector<float4> outputF4(imgWidth * imgHeight);
        std::vector<float4> inputF4(imgWidth * imgHeight);
        std::cout << imgWidth << " : " << imgHeight << std::endl;
        processNormalizer(inputImg, imgWidth, imgHeight, outputF4);
//        processNormalizer(inputImg,imgWidth,imgHeight,output);

        inputF4.swap(outputF4);

        outputF4.clear();

        processRBG_to_HSV(inputF4, imgWidth, imgHeight, outputF4);

        inputF4.clear();

        inputF4.swap(outputF4);

        outputF4.clear();

        processHSV_to_RGB(inputF4, imgWidth, imgHeight, output);

        //utils::compareImages(resultCPU, output);

    }
}