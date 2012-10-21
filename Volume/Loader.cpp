#include "Loader.h"
#include "MACROS/MACROS.h"
#include "Utilities/Utils.h"

volumeSize* Volume::LoadHeader(const char *path)
{


    // Atructure allocation
    volumeSize* iVolDim = MEM_ALLOC_1D_GENERIC(volumeSize, 1);

    // Getting the header file "<FILE_NAME>.hdr"
    char hdrFile[1000];
    sprintf(hdrFile, "%s.hdr", path);

    // Input file stream
    std::ifstream ifile;

    // Open file
    ifile.open(hdrFile, std::ios::in);

    // Double checking for the existance of the header file
    if (!ifile.fail())
    {
        // Check point success
        INFO("Openging header file : " + CCATS(hdrFile));
    }
    else
    {
        // Check point failing
        INFO("Error OPENING header file : " + CCATS(hdrFile));
        EXIT( 0 );
    }

    // Reading the dimensions in XYZ order
    ifile >> iVolDim->NX;
    ifile >> iVolDim->NY;
    ifile >> iVolDim->NZ;

    INFO("Volume Dimensions : "
        + STRG( "[" ) + ITS( iVolDim->NX ) + STRG( "]" ) + " x "
        + STRG( "[" ) + ITS( iVolDim->NX ) + STRG( "]" ) + " x "
        + STRG( "[" ) + ITS( iVolDim->NX ) + STRG( "]" ));

    // Closing the innput stream
    ifile.close();

    INFO("Reading volume header DONE");

    return iVolDim;
}

volumeImage* Volume::LoadVolume(const char* path)
{


    // Loading the volume file
    char volFile[1000];
    sprintf(volFile, "%s.img", path);

    // Checking for the existance of the volume file
    if (!volFile)
    {
        // Check point failing
        INFO("Error FINDING raw volume file : " + CCATS(volFile));
        EXIT( 0 );
    }
    else
    {
        // Check point success
        INFO("Opening raw volume file : " + CCATS(volFile));
    }

    // Allocating volume structre
    volumeImage* iVolume = MEM_ALLOC_1D_GENERIC(volumeImage, 1);

    // Load header
    volumeSize* iVolSize = MEM_ALLOC_1D_GENERIC(volumeSize, 1);
    iVolSize = Volume::LoadHeader(path);

    // Reading the header file to get volume dimensions
    iVolume->volSize.NX = (iVolSize->NX);
    iVolume->volSize.NY = (iVolSize->NY);
    iVolume->volSize.NZ = (iVolSize->NZ);

    if (iVolume->volSize.NX == iVolume->volSize.NY &&
            iVolume->volSize.NX == iVolume->volSize.NZ)
    {
        iVolume->volSize.NU = iVolume->volSize.NX;
        INFO("Loaded volume has unified dimensiosn of:" + ITS(iVolume->volSize.NU));
    }
    else
    {
        INFO("NON UNIFIED VOLUME HAS BEEN LOADDED - UNIFICATION REQUIRED");
    }

    // Volume flat size
    const int volSize = iVolume->volSize.NX *
            iVolume->volSize.NY * iVolume->volSize.NZ;
    INFO("Volume flat size : " + ITS(volSize));

    /* @Volume size in bytes */
    const int volSizeBytes = sizeof(char) * volSize;
    INFO("Volume size in MBbytes: " + FTS(volSizeBytes / (1024 * 1024)));

    /* @ Allocating volume */
    iVolume->volPtr = MEM_ALLOC_1D_GENERIC(char, volSize);
    INFO("Preparing volume data & meta-data");

    // Opening volume file
    FILE* ptrFile = fopen(volFile, "rb");

    // Double checking for the existance of the volume file
    if (!ptrFile)
    {
        // Check point failing
        INFO("Error FINDING raw volume file : " + CCATS(volFile));
        EXIT( 0 );
    }

    // Read the volume raw file
    size_t imageSize = fread(iVolume->volPtr,
                             1,
                             volSizeBytes,
                             ptrFile);

    // Checking if the volume was loaded or not
    if (!imageSize)
    {
        INFO("Error READING raw volume file : " + CCATS(volFile));
        EXIT(0);
    }

    INFO("Reading volume raw file DONE");

    return iVolume;
}
