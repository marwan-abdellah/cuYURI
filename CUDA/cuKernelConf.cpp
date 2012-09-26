#include "cuKernelConf.h"
#include "Utilities/MACROS.h"


kernelConf* cuUtils::AutoGenKernelConf_1D(const int N)
{
    // Automatic configuration for the kernel
    kernelConf* autoConf = MEM_ALLOC_1D_GENERIC(kernelConf, 1);

    int threadsPerBlock;

    // Set the threadsPerBlock according to N
    if (2 <= N && N < 4)
    {
        threadsPerBlock = 2;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (4 <= N && N < 8)
    {
        threadsPerBlock = 4;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (8 <= N && N < 16)
    {
        threadsPerBlock = 8;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (16 <= N && N < 32)
    {
        threadsPerBlock = 16;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (32 <= N && N < 64)
    {
        threadsPerBlock = 32;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (64 <= N && N < 128)
    {
        threadsPerBlock = 64;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (128 <= N && N < 256)
    {
        threadsPerBlock = 128;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (256 <= N && N < 512)
    {
        threadsPerBlock = 256;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }
    if (N >= 512)
    {
        threadsPerBlock = 512;
        INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));
    }

    // Automatic kernel configuration according to N or the array length
    autoConf->cuBlock = dim3(threadsPerBlock, 1, 1);
    autoConf->cuGrid = dim3(N / threadsPerBlock, 1, 1);

    INFO("Generated kernel configuration is "
         "[" + ITS(autoConf->cuBlock.x) +
         "," + ITS(autoConf->cuBlock.y) +
         "," + ITS(autoConf->cuBlock.z) + "] Block & " +
         "[" + ITS(autoConf->cuGrid.x) +
         "," + ITS(autoConf->cuGrid.y) +
         "," +ITS(autoConf->cuGrid.z) + "] Grid");

    return autoConf;
}


///dim3 cuBlock(4, 1, 1);
/// dim3 cuGrid(N / cuBlock.x, 1, 1);
///
