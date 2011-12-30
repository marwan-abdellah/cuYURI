/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: add_2D.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "add_2D.h"

#define TYPE float
#define PRINT_RES 1

namespace add_2D
{
	namespace globalMem
	{
		char* appName = "add_2D_GM";

		void run(int argc, char** argv);
	}
}

void add_2D::globalMem::run(int argc, char** argv)
{
	float* inArr1_H;
	float* inArr2_H;
	float* outArr_H;

	float* inArr1_D;
	float* inArr2_D;
	float* outArr_D;

	int arrayWidth = 16 ;
	int arrayHeight = 16; 

	stlTimer _timer; 

	// GPU configuration structure 
	gpuConfig_XD conf;

	// 2D pitch 
	pitchSize __pitch; 

	// Initalize application 
	app::initApp(add_2D::globalMem::appName);

	// Configure GPU
	conf.numThreads = dim3(arrayWidth, arrayHeight, 1); 
	conf.numThreadsPerBlock = dim3(4, 4, 1); 
	conf.numBlocksPerGrid = dim3(conf.numThreads.x/conf.numThreadsPerBlock.x, conf.numThreads.y/conf.numThreadsPerBlock.y, 1); 

	// Benchmarking file initialization 
	benchFile::header(add_2D::globalMem::appName); 
	benchFile::message("Kernel size " + util::intToString(conf.numThreads.x) + " X " + util::intToString(conf.numThreads.y)); 
	benchFile::message("Number of threads per block " + util::intToString(conf.numThreadsPerBlock.x) + " X " + util::intToString(conf.numThreadsPerBlock.x));
	benchFile::message("Number of blocks per grid " + util::intToString(conf.numBlocksPerGrid.x) + " X " +  util::intToString(conf.numBlocksPerGrid.y));

	// Host allocation 
	inArr1_H = arr_2D::flat::createArray <TYPE> (arrayWidth, arrayHeight); 
	inArr2_H = arr_2D::flat::createArray <TYPE> (arrayWidth, arrayHeight); 
	outArr_H = arr_2D::flat::createArray <TYPE> (arrayWidth, arrayHeight); 

	// Device Allocation 
	__pitch = CU::arr_2D::globalMem::getPitch <TYPE> (arrayWidth, arrayHeight, (TYPE) NULL); 
	inArr1_D = CU::arr_2D::globalMem::createArray_D <TYPE> (arrayWidth, arrayHeight); 
	inArr2_D = CU::arr_2D::globalMem::createArray_D <TYPE> (arrayWidth, arrayHeight); 
	outArr_D = CU::arr_2D::globalMem::createArray_D <TYPE> (arrayWidth, arrayHeight);

	// Filling host arrays <SEQ>
	arr_2D::flat::fillArray_SEQ <TYPE> (inArr1_H, arrayWidth, arrayHeight);
	arr_2D::flat::fillArray_SEQ <TYPE> (inArr2_H, arrayWidth, arrayHeight);
	arr_2D::flat::zeroArray <TYPE> (outArr_H, arrayWidth, arrayHeight);

	// Print input arrays to the trace file 
	traceFile::header("Input array # 1"); 
	arr_2D::flat::printArray <TYPE> (inArr1_H, arrayWidth, arrayHeight, true); 
	traceFile::header("Input array # 2"); 
	arr_2D::flat::printArray <TYPE> (inArr2_H, arrayWidth, arrayHeight, true);

	// Uploading data to GPU
	bench::cpu::startTimer(& _timer); 
	CU::arr_2D::globalMem::uploadData <TYPE> (inArr1_H, inArr1_D, __pitch, arrayWidth, arrayHeight); 
	CU::arr_2D::globalMem::uploadData <TYPE> (inArr2_H, inArr2_D, __pitch, arrayWidth, arrayHeight);
	CU::arr_2D::globalMem::uploadData <TYPE> (outArr_H, outArr_D, __pitch, arrayWidth, arrayHeight); 
	bench::cpu::stopTimer(& _timer); 
	benchFile::header("Benchmarking GPU "); 
	benchFile::bench(& _timer, "Upload time"); 

	// Run Kernel 
	bench::cpu::startTimer(& _timer);
	CU::arr_2D::globalMem::cuAddArrays_f (inArr1_D, inArr2_D, outArr_D, __pitch, arrayWidth, arrayHeight, &conf);
	bench::cpu::stopTimer(& _timer); 
	benchFile::bench(& _timer, "Kernel time");

	// Download resulting array 
	bench::cpu::startTimer(& _timer); 
	CU::arr_2D::globalMem::downloadData <TYPE> (outArr_H, outArr_D, __pitch, arrayWidth, arrayHeight);
	bench::cpu::stopTimer(& _timer); 
	benchFile::bench(& _timer, "Download time"); 

#ifdef PRINT_RES
	// Print results to check to trace file 
	traceFile::header("Output array"); 
	arr_2D::flat::printArray <TYPE> (outArr_H, arrayWidth, arrayHeight, true);
#endif 

	// Releasing memory 
	/* @Host */
	arr_2D::flat::clearArray <TYPE> (inArr1_H); 
	arr_2D::flat::clearArray <TYPE> (inArr2_H); 
	arr_2D::flat::clearArray <TYPE> (outArr_H); 

	/* @Device */
	CU::arr_2D::globalMem::clearArray_D <TYPE> (inArr1_D); 
	CU::arr_2D::globalMem::clearArray_D <TYPE> (inArr2_D); 
	CU::arr_2D::globalMem::clearArray_D <TYPE> (outArr_D); 

	// Exitting application 
	app::exitApp(); 
}