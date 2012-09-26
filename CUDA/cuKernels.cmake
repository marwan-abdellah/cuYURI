# --------------------------------------------------------
# @NOTE: You have to append all the ".cu" interfaces in this
#       cmake file to have them compiled
# --------------------------------------------------------

SET(CUDA_INTERFACES_DIR ${CMAKE_SOURCE_DIR}/CUDA/Interfaces)

SET(CUDA_SOURCES    ${CUDA_INTERFACES_DIR}/Fill_1D_Array_RND_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Copy_1D_Array_Impl.cu
                    ${CUDA_INTERFACES_DIR}/Add_1D_Arrays_Impl.cu
)
