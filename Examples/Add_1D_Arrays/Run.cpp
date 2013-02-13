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

#include "ex_Add_1D_Arrays.h"
#include <cstdlib>
#include "stdio.h"
#include "Macros/MemoryMACROS.h"
#include "Memory/FlatVolume.h"
#include "Memory/Volume.h"
#include "Memory/FlatImage.h"
#include "Memory/Image.h"
#include "Memory/Vector.h"



int main(int argc, char* argv [])
{
    // Run the example
    //ex_Add_1D_Arrays::run(argc, argv);


    Yuri::Memory::Volume<int>* volume ;
    volume = MEM_ALLOC_VOLUME(int, 4,4,4);

    const int NX = volume->getNX();
    const int NY = volume->getNY();
    const int NZ = volume->getNZ();

    int counter = 0;
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            for (int k = 0; k < NZ; k++)
                volume->getPtrVolumeData()[i][j][k] = counter++;

    counter = 0;
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            for (int k = 0; k < NZ; k++)
                printf("%d \n", volume->getPtrVolumeData()[i][j][k]);


    delete (volume);

    return 0;
}
