/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: mul_1D.cpp
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "mul_1D.h"

#define TYPE float
#define PRINT_RES 0

namespace mul_1D
{
	char* appName = "mul_1D";

	void mul_1D::run_mul_1D(int argc, char** argv);
}

void mul_1D::run_mul_1D(int argc, char** argv)
{
	float* inArr1_H;
	float* inArr2_H;
	float* outArr_H;

	float* inArr1_D;
	float* inArr2_D;
	float* outArr_D;

	int arrayLenght = 1024 * 50; 
	int numThreadsPerBlock = 256; 

	stlTimer _timer; 

	// GPU configuration structure 
	gpuConfig_1D conf;

	// Initalize application 
	app::initApp(mul_1D::appName);

	// Configure GPU
	conf.numThreads = arrayLenght; 
	conf.numThreadsPerBlock = 256; 
	conf.numBlocksPerGrid = (conf.numThreads + conf.numThreadsPerBlock - 1) / conf.numThreadsPerBlock;

	// Benchmarking file initialization 
	benchFile::header(mul_1D::appName); 
	benchFile::message("Kernel size " + util::intToString(conf.numThreads)); 
	benchFile::message("Number of threads per block " + util::intToString(conf.numThreadsPerBlock));
	benchFile::message("Number of blocks per grid " + util::intToString(conf.numBlocksPerGrid));

	// Host allocation 
	inArr1_H = arr_1D::createArray <TYPE> (arrayLenght); 
	inArr2_H = arr_1D::createArray <TYPE> (arrayLenght); 
	outArr_H = arr_1D::createArray <TYPE> (arrayLenght); 

	// Device Allocation 
	inArr1_D = CU::arr_1D::createArray_D <TYPE> (arrayLenght); 
	inArr2_D = CU::arr_1D::createArray_D <TYPE> (arrayLenght); 
	outArr_D = CU::arr_1D::createArray_D <TYPE> (arrayLenght);

	// Filling host arrays 
	arr_1D::fillArray_SEQ <TYPE> (inArr1_H, arrayLenght);
	arr_1D::fillArray_SEQ <TYPE> (inArr2_H, arrayLenght);
	arr_1D::zeroArray <TYPE> (outArr_H, arrayLenght);

	// Host execution 
	bench::cpu::startTimer(& _timer);
	arr_1D::mulArrays(inArr1_H, inArr2_H, outArr_H, arrayLenght); 
	bench::cpu::stopTimer(& _timer);
	benchFile::header("Benchmarking CPU "); 
	benchFile::bench(& _timer, "CPU time"); 

	// Null the array for receivng GPU result
	arr_1D::zeroArray <TYPE> (outArr_H, arrayLenght);

	// Uploading data to GPU
	bench::cpu::startTimer(& _timer); 
	CU::arr_1D::uploadData <TYPE> (inArr1_H, inArr1_D, arrayLenght); 
	CU::arr_1D::uploadData <TYPE> (inArr2_H, inArr2_D, arrayLenght);
	CU::arr_1D::uploadData <TYPE> (outArr_H, outArr_D, arrayLenght); 
	bench::cpu::stopTimer(& _timer);
	benchFile::header("Benchmarking GPU "); 
	benchFile::bench(& _timer, "Upload time"); 

	// Run Kernel 
	bench::cpu::startTimer(& _timer);
	CU::arr_1D::cuMulArrays_f (inArr1_D, inArr2_D, outArr_D, arrayLenght, &conf);
	bench::cpu::stopTimer(& _timer); 
	benchFile::bench(& _timer, "Kernel time");

	// Download resulting array 
	bench::cpu::startTimer(& _timer); 
	CU::arr_1D::downloadData <TYPE> (outArr_H, outArr_D, arrayLenght);
	bench::cpu::stopTimer(& _timer); 
	benchFile::bench(& _timer, "Download time"); 

#ifdef PRINT_RES
	// Print results to check to trace file 
	arr_1D::printArray <TYPE> (outArr_H, arrayLenght, true);
#endif 

	/* Releasing memory */
	/* @Host */
	arr_1D::clearArray <TYPE> (inArr1_H); 
	arr_1D::clearArray <TYPE> (inArr2_H); 
	arr_1D::clearArray <TYPE> (outArr_H); 

	/* @Device */
	CU::arr_1D::clearArray_D <TYPE> (inArr1_D); 
	CU::arr_1D::clearArray_D <TYPE> (inArr2_D); 
	CU::arr_1D::clearArray_D <TYPE> (outArr_D); 

	app::exitApp(); 
}