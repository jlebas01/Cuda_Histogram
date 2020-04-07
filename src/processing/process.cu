//
// Created by jlebas01 on 04/04/2020.
//

#include <iostream>

#include <utils/tools.hpp>
#include <utils/common.hpp>

#include <processing/process.hpp>
//#include <processing/process_Histogram.hpp>
#include <processing/process_normalized.hpp>
#include <processing/process_RGB_to_HSV.hpp>
#include <processing/process_HSV_to_RGB.hpp>
#include <processing/process.hpp>

#include <chrono/chronoGPU.hpp>


namespace process {

    __device__ float clip(float n, float lower, float upper) {
        return fmax(lower, fmin(n, upper));
    }

    __device__ void addData256(volatile unsigned int *s_WarpHist, unsigned int data, unsigned int threadTag) {
        unsigned int count;
        do {
            count = s_WarpHist[data] & 0x07FFFFFFU;
            count = threadTag | (count + 1);
            s_WarpHist[data] = count;
        } while (s_WarpHist[data] != count);
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