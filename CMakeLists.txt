cmake_minimum_required(VERSION 3.10)

# Set the project name

# get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (PROJNAME "engine")
Project(${PROJNAME} CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(VK_ENABLE_BETA_EXTENSIONS "Enable beta extensions provided by the Vulkan SDK" OFF)

# Find the SDL2 library

if(WIN32)
    set(SDL2_DIR "E:/SDL2/cmake")
    set(SDL2_INCLUDE_DIR E:/SDL2/include)
endif()

# Include SDL2 headers
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
# include_directories(${CMAKE_SOURCE_DIR}/external/LiteMath)

############################################################################################################################
# Link nvpro_core
#

set (BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

############################################################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

find_package(OpenMP REQUIRED)
find_package(SDL2 REQUIRED)
# find_package(Vulkan REQUIRED COMPONENTS glslc glslangValidator shaderc_combined)

_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_IMGUI()

############################################################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
#
_add_nvpro_core_lib()

_add_project_definitions(${PROJNAME})

############################################################################################################################
# Add the executable
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c src/* include/*)

SET(VULKAN_TARGET_ENV vulkan1.2)

UNSET(GLSL_SOURCES)
UNSET(SPV_OUTPUT)
file(GLOB_RECURSE GLSL_HEADER_FILES "shaders/*.h")
file(GLOB_RECURSE GLSL_SOURCE_FILES "shaders/*.glsl")
foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    _compile_GLSL(${GLSL} "shaders/${FILE_NAME}.spv" GLSL_SOURCES SPV_OUTPUT)
endforeach(GLSL)

list(APPEND GLSL_SOURCES ${GLSL_HEADER_FILES})
source_group("Shader Files" FILES ${GLSL_SOURCES})

add_executable(${PROJNAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${GLSL_SOURCES})

############################################################################################################################
# Source code group
#
source_group("Source Files" FILES ${SOURCE_FILES})

# Link the SDL2 library to the executable
target_link_libraries(${PROJNAME} PUBLIC ${SDL2_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJNAME} PUBLIC ${PLATFORM_LIBRARIES} nvpro_core) # link nvpro_core libraries

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} PUBLIC debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} PUBLIC optimized ${RELEASELIB})
endforeach(RELEASELIB)

_finalize_target( ${PROJNAME} )

install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/shaders")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}/shaders")
install(DIRECTORY "../scenes" CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(DIRECTORY "../scenes" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")

# Set path to executable
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
