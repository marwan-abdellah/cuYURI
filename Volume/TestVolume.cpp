#include "Vector3.h"
#include "Volume.h"
#include "Globals.h"
#include "Utilities/MACROS.h"

volumeImage* Volume::CreateTestVolume(const int N)
{
    INFO("Creating TEST Volume");

    // Allocating volume image object
    volumeImage* volImg = MEM_ALLOC_1D_GENERIC(volumeImage, 1);

    // Saving the volume dimensions in the volume structure
    volImg->volSize.NX = N;
    volImg->volSize.NY = N;
    volImg->volSize.NZ = N;
    volImg->volSize.NU = N;

    // Allocating volume data array (RGBA)
    const int volSize = N * N * N * 4;
    volImg->volPtr = MEM_ALLOC_1D_GENERIC(char, volSize);

    for(int ctrX = 0; ctrX < N; ctrX++)
    {
        for(int ctrY = 0; ctrY < N; ctrY++)
        {
            for(int ctrZ = 0; ctrZ < N; ctrZ++)
            {
                volImg->volPtr[(ctrX * 4)
                        + (ctrY * N * 4) + (ctrZ * N * N * 4)] = ctrZ % 250;
                volImg->volPtr[(ctrX * 4)
                        + 1 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = ctrY % 250;
                volImg->volPtr[(ctrX * 4)
                        + 2 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                volImg->volPtr[(ctrX * 4)
                        + 3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 230;

                Vector3 pVec = Vector3(ctrX, ctrY, ctrZ)- Vector3(N - 20, N - 30, N - 30);

                bool cubeTest = (pVec.length() < 42);
                if(cubeTest)
                    volImg->volPtr[(ctrX * 4) + 3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 0;

                pVec = Vector3(ctrX,ctrY,ctrZ)- Vector3(N/2,N/2,N/2);
                cubeTest = (pVec.length() < 24);
                if(cubeTest)
                    volImg->volPtr[(ctrX*4)+3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 0;


                if(ctrX > 20 && ctrX < 40 && ctrY > 0
                        && ctrY < N && ctrZ > 10 && ctrZ < 50)
                {
                    volImg->volPtr[(ctrX * 4)
                            + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 100;
                    volImg->volPtr[(ctrX * 4)
                            + 1 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                    volImg->volPtr[(ctrX * 4)
                            + 2 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = ctrY % 100;
                    volImg->volPtr[(ctrX * 4)
                            + 3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                }

                if(ctrX > 50 && ctrX < 70 && ctrY > 0
                        && ctrY < N && ctrZ > 10 &&  ctrZ < 50)
                {
                    volImg->volPtr[(ctrX* 4)
                            + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                    volImg->volPtr[(ctrX* 4)
                            + 1 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                    volImg->volPtr[(ctrX* 4)
                            + 2 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = ctrY % 100;
                    volImg->volPtr[(ctrX* 4)
                            + 3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                }

                if(ctrX > 80 && ctrX < 100 && ctrY > 0
                        && ctrY < N && ctrZ > 10 &&  ctrZ < 50)
                {
                    volImg->volPtr[(ctrX * 4)
                            + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                    volImg->volPtr[(ctrX * 4)
                            + 1 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 70;
                    volImg->volPtr[(ctrX * 4)
                            + 2 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = ctrY % 100;
                    volImg->volPtr[(ctrX * 4)
                            + 3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 250;
                }

                pVec = Vector3(ctrX, ctrY, ctrZ) - Vector3(24, 24, 24);
                cubeTest = (pVec.length() < 40);
                if(cubeTest)
                    volImg->volPtr[(ctrX * 4) + 3 + (ctrY * N * 4) + (ctrZ * N * N * 4)] = 0;
            }
        }
    }

    INFO("Creating TEST Volume DONE");

    return volImg;
}
