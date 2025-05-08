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
include_directories(${CMAKE_SOURCE_DIR}/src/app)
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
add_executable(${PROJNAME})
_add_project_definitions(${PROJNAME})

#--------------------------------------------------------------------------------------------------
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c src/* src/app/* common/*)
file(GLOB EXTRA_COMMON ${TUTO_KHR_DIR}/common/*.*)
list(APPEND COMMON_SOURCE_FILES ${EXTRA_COMMON})
include_directories(${TUTO_KHR_DIR}/common/)

#--------------------------------------------------------------------------------------------------
# GLSL to SPIR-V custom build
compile_glsl_directory(
	SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders" 
	DST "${CMAKE_CURRENT_SOURCE_DIR}/spv" 
	VULKAN_TARGET "vulkan1.2"
	DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
)

#--------------------------------------------------------------------------------------------------
# Sources
target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES} ${HEADER_FILES})
target_sources(${PROJNAME} PUBLIC ${COMMON_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${GLSL_SOURCES} ${GLSL_HEADERS})

#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"       FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Sources"      FILES ${SOURCE_FILES})
source_group("Headers"      FILES ${HEADER_FILES})
source_group("Shader Sources"  FILES ${GLSL_SOURCES})
source_group("Shader Headers"  FILES ${GLSL_HEADERS})

# Link the SDL2 library to the executable
target_link_libraries(${PROJNAME} ${SDL2_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core) # link nvpro_core libraries

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#--------------------------------------------------------------------------------------------------
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJNAME} )

install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/spv")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}/spv")
