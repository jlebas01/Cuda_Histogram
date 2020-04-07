/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: main.cu
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <algorithm>

#include <cstdint>
#include <chrono/chronoCPU.hpp>
#include <processing/process.hpp>
#include <lodepng.h>
#include <utils/conv_utils.hpp>

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name (required)" << std::endl
					<< " \t -c <C>: <C> convolution type (required)" << std::endl 
					<< " \t --- " << utils::BUMP_3x3 << " = Bump 3x3" << std::endl
					<< " \t --- " << utils::SHARPEN_5x5 << " = Sharpen 5x5" << std::endl
					<< " \t --- " << utils::EDGE_DETECTION_7x7 << " = Edge detection 7x7" << std::endl
					<< " \t --- " << utils::MOTION_BLUR_15x15 << " = Motion Blur 15x15" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	float clampf(const float val, const float min , const float max) 
	{
		return std::min<float>(max, std::max<float>(min, val));
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char * fileName= "../images/chateau.png";
		uint convType;
		
		// Get input image
		std::vector<uint8_t> inputUchar;
		uint imgWidth;
		uint imgHeight;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(inputUchar, imgWidth, imgHeight, fileName, LCT_RGBA);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		// Convert to uchar4 for exercise convenience
		std::vector<uchar4> input;
		input.resize(inputUchar.size() / 4);
		for (uint i = 0; i < input.size(); ++i)
		{
			const uint id = 4 * i;
			input[i].x = inputUchar[id];
			input[i].y = inputUchar[id + 1];
			input[i].z = inputUchar[id + 2];
			input[i].w = inputUchar[id + 3];
		}
		inputUchar.clear();
		std::cout << "Image has " << imgWidth << " x " << imgHeight << " pixels (RGBA)" << std::endl;

		// Create 2 output images
		std::vector<uchar4> outputCPU(imgWidth * imgHeight);
		std::vector<uchar4> outputGPU(imgWidth * imgHeight);

		
		std::cout << input.size() << " - " << outputCPU.size() << " - " << outputGPU.size() << std::endl;

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputCPUName = name  + "_CPU" + ext;
		std::string outputGPUName = name  + "_GPU" + ext;
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		process::process(input, imgWidth, imgHeight, outputCPU, outputGPU);

		std::cout << "Save image as: " << outputGPUName << std::endl;
		error = lodepng::encode(outputGPUName, reinterpret_cast<uint8_t *>(outputGPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}

	exit(EXIT_SUCCESS);
}
