#include "cuGPUConf.h"
#include "cuIncludes.h"
#include "cuGlobals.h"
#include "MACROS/MACROS.h"

namespace cuUtils
{

void GetGPUConf()
{
    int deviceCount = 0;
    cudaError_t errorID = cudaGetDeviceCount(&deviceCount);

    if (errorID != cudaSuccess)
    {
        INFO("cudaGetDeviceCount FAILS, Error: " +
             CATS(cudaGetErrorString(errorID)) +
             " - Error ID:" +ITS((int) errorID));

        EXIT(0);
    }
    else
        INFO("Number of GPU(s) connected to the system : " + ITS(deviceCount));

    // Iterate over all the devices and list their configuration
    int devCtr = 0;
    for (devCtr = 0; devCtr < deviceCount; ++devCtr)
    {

    }

    // Device propoerties structure
    cudaDeviceProp deviceProp;

    // Driver/Runtime versions
    int driverVersion = 0;
    int runtimeVersion = 0;

    // Query the device
    cudaGetDeviceProperties(&deviceProp, devCtr);

    INFO("Device name : " + CATS(deviceProp.name));


//#if CUDART_VERSION >= 2020
    // Get driver version
    cudaDriverGetVersion(&driverVersion);

    // Get run time version
    cudaRuntimeGetVersion(&runtimeVersion);

//    INFO("CUDA Driver/Runtime version : " +
//         CATS((driverVersion / 1000)) + "." + CATS(((driverVersion % 100) / 10)) +
//         CATS((runtimeVersion / 1000) + "." + CATS((runtimeVersion % 100) / 10)));
//#endif

    printf("blahh %u \n", deviceProp.totalConstMem);
    printf("blahh %u \n", deviceProp.maxThreadsPerMultiProcessor);
    INFO("Total amount of constant memory: " + FTS(deviceProp.totalConstMem));
    INFO("Maximum number of threads per multiprocessor: " + (deviceProp.maxThreadsPerMultiProcessor));


}

}
