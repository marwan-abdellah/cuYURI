#####################################################################
# Copyright © 2011-2012,
# Marwan Abdellah: <abdellah.marwan@gmail.com>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#####################################################################

#########################################################
# @NOTE: You have to append all the ".cu" interface
# implementations in this cmake file to have them
# compiled.
#########################################################

SET(CUDA_INTERFACES_DIR ${CMAKE_SOURCE_DIR}/CUDA/Interfaces)

SET(CUDA_SOURCES    ${CUDA_INTERFACES_DIR}/Fill_1D_Array_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Fill_1D_Array_Normalized_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Fill_1D_Array_RND_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Copy_1D_Array_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Add_1D_Arrays_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Multiply_1D_Arrays_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Subtract_1D_Arrays_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Divide_1D_Arrays_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Constant_Add_1D_Array_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Constant_Multiply_1D_Array_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Constant_Subtract_1D_Array_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Constant_Divide_1D_Array_Impl.cu
)
