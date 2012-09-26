/*********************************************************************
 * Copyright © 2011-2012,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.

 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/
#include "cuUtilities.h"
#include "Globals.h"


template <typename T>
int cuUtils::Upload_1D_Array(T* hostArray, T* deviceArray, int N)
{
    LOG();

    INFO("Uploading vector to device");

    // Required  device memory in bytes
    const int reqDevMem = N * sizeof(T);

    if (reqDevMem < 1024)
    {
        INFO("Memory Required: " + ITS(reqDevMem) + " Bytes");
    }
    else
    {
        if (reqDevMem < (1024 * 1024))
        {
            INFO("Memory Required: " +
                 ITS(reqDevMem / 1024) + " KBytes");
        }
        else
        {
            if (reqDevMem < (1024 * 1024 * 1024))
            {
                INFO("Memory Required: " +
                     ITS(reqDevMem / (1024 * 1024)) + " MBytes");
            }
            else
            {
                INFO("Memory Required: " +
                    ITS(reqDevMem / (1024 * 1024 * 1024)) + " GBytes");
            }
        }
    }

    // Copy array from the host side to the device side
    cutilSafeCall(cudaMemcpy(deviceArray, hostArray,
                             reqDevMem, cudaMemcpyHostToDevice));

    return SUCCESS;
}

template <typename T>
int cuUtils::Download_1D_Array(T* hostArray, T* deviceArray, int N)
{
    LOG();

    INFO("Downloading vector to host");

    const int reqDevMem = N * sizeof(int);

    // Copy array form the device side to the host side
    cutilSafeCall(cudaMemcpy(hostArray, deviceArray,
                             reqDevMem, cudaMemcpyDeviceToHost));

    return SUCCESS;
}

template <typename T>
T* cuUtils::Create_Device_Vector(const int N)
{
    LOG();

    INFO("Creating device vector");

    // Vector size in bytes
    const int vectorSize = N * sizeof(T);

    // Pointer to the device vector
    T* deviceVecPtr;

    // Allocate the device vector
    cudaMalloc((void**)(&deviceVecPtr), vectorSize);

    return deviceVecPtr;
}

template <typename T>
int cuUtils::Free_Device_Vector(T* deviceVecPtr)
{
    LOG();

    INFO("Freeing device memory");

    // Free device vector
    cudaFree(deviceVecPtr);

    return SUCCESS;
}
































int cuUtils::upload_1D_float(float* hostArr, float* devArr, int size_X)
{
	LOG();

    int devMem = size_X * sizeof(float);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_float(float* hostArr, float* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(float);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL" + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(float);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_int(int* hostArr, int* devArr, int size_X)
{
    LOG();

    int devMem = size_X * sizeof(int);

    cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

    return 0;
}

int cuUtils::download_1D_float(float* hostArr, float* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_float(float* hostArr, float* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::upload_1D_double(double* hostArr, double* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(double);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_double(double* hostArr, double* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(double);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(double);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_double(double* hostArr, double* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(double);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_double(double* hostArr, double* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(double);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(double);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::upload_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}


	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::upload_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftDoubleComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftDoubleComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftDoubleComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftDoubleComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftDoubleComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftDoubleComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}
