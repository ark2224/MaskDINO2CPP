cmake_minimum_required(VERSION 3.10)
project(MaskDINO2 VERSION 1.0)

list(APPEND CMAKE_PREFIX_PATH "<libtorch-path>")
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(MaskDINO2 MaskDINO2.cpp main.cpp)
target_link_libraries(MaskDINO2 "${TORCH_LIBRARIES}")
set_property(TARGET MaskDINO2 PROPERTY CXX_STANDARD 17)
# including header files below in the /include directory:
target_include_directories(MaskDINO2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MaskDINO2
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:MaskDINO2>)
endif (MSVC)