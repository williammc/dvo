cmake_minimum_required(VERSION 2.8)

project(dvo)
get_filename_component(DVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
set(DVO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DDVO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# Look for our CMake modules paths
set(CMAKER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/cmaker)
list(APPEND CMAKE_MODULE_PATH ${CMAKER_ROOT}/cmake
                              ${CMAKER_ROOT}/cmake/modules
                              ${CMAKE_CURRENT_SOURCE_DIR}/cmake
                              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
                              ${CMAKE_CURRENT_BINARY_DIR})

if(NOT THE_PROJECT_ROOT OR THE_PROJECT_ROOT STREQUAL "")
  set(THE_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
endif()

include(CMakerCommonUtil)  # include useful macros
include(CMakerCommonSettings)

set(DVO_CXX_FLAGS)
add_definitions(-DDVO_ROOT="${DVO_ROOT}")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(DVO_CXX_FLAGS " -stdlib=libc++ ")  # prevent default libc++ on new mac
endif()

if(NOT WIN32)
  set(DVO_CXX_FLAGS " ${DVO_CXX_FLAGS} -std=c++11 ") 
endif()
cmaker_print_status("DVO_CXX_FLAGS:${DVO_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS ${DVO_CXX_FLAGS})
cmaker_common_build_setting()
## Dependencies ================================================================
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/DVODependencies_Config.cmake")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DVODependencies_Config.cmake.in"
       DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/DVODependencies_Config.cmake.in"
              "${CMAKE_CURRENT_BINARY_DIR}/DVODependencies_Config.cmake")
endif()
include(DVODependencies_Config)

# Slick
set(Slick_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/slick-1.1)
list(APPEND DVO_INC_DIRS ${Slick_ROOT}/external/eigen-3.2.2 
                         ${Slick_ROOT}/external/eigen-3.2.2/unsupported
                         ${Slick_ROOT})

# Sophus
list(APPEND DVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/sophus)

## Boost
if(WIN32)
  set(Boost_USE_STATIC_LIBS  ON)
endif(WIN32)

find_package(Boost REQUIRED COMPONENTS serialization)

if(NOT Boost_FOUND)
  cmaker_print_error("Cannot find boost")
endif()

list(APPEND DVO_INC_DIRS ${Boost_INCLUDE_DIRS})
list(APPEND DVO_LIB_DIRS ${Boost_LIBRARY_DIR})
cmaker_print_status("Boost libs: ${Boost_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)
set(OCV_LIBS)
foreach(NAME ${OpenCV_LIBRARIES})
  set(NAME ${NAME}${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH})
  list(APPEND OCV_LIBS debug;${NAME}d;optimized;${NAME})
endforeach()
list(APPEND DVO_INC_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND DVO_LIB_DIRS ${OpenCV_LIB_DIR})
list(APPEND DVO_EXTERNAL_LIBS ${OCV_LIBS})
cmaker_print_status("OpenCV include dirs:${OpenCV_INCLUDE_DIRS}")
cmaker_print_status("OpenCV libs:${OpenCV_LIBRARIES}")
get_filename_component(OpenCV_BINARY_DIR "${OpenCV_LIB_DIR}/../bin" ABSOLUTE)
list(APPEND THE_DEPEDENCIES_BINARY_PATHS ${OpenCV_BINARY_DIR})

# TBB
# set(TBB_INSTALL_DIR "D:/deps_msvc_common/tbb42_20140601oss")
# set(TBB_ARCH_PLATFORM "ia32")
# set(TBB_COMPILER "vc12")
find_package(TBB REQUIRED)
list(APPEND DVO_INC_DIRS ${TBB_INCLUDE_DIR})
cmaker_cull_library_paths(TBB_LIBRARIES)
cmaker_cull_library_paths(TBB_DEBUG_LIBRARIES)
set(TBB_LIBS)
foreach(lib ${TBB_LIBRARIES})
  list(APPEND TBB_LIBS optimized ${lib})
endforeach()
foreach(lib ${TBB_DEBUG_LIBRARIES})
  list(APPEND TBB_LIBS debug ${lib})
endforeach()
list(APPEND DVO_EXTERNAL_LIBS ${TBB_LIBS})
list(APPEND DVO_LIB_DIRS ${TBB_LIBRARY_DIRS})
cmaker_print_status("Test tbb lib:${TBB_INSTALL_DIR}/lib/${TBB_ARCH_PLATFORM}/${TBB_COMPILER}")
set(TBB_BINARY_DIR ${TBB_INSTALL_DIR}/bin/${TBB_ARCH_PLATFORM}/${TBB_COMPILER})
list(APPEND THE_DEPEDENCIES_BINARY_PATHS ${TBB_BINARY_DIR})

cmaker_print_status("TBB_LIBRARIES:${TBB_LIBRARIES}")
cmaker_print_status("TBB_BINARY_DIR:${TBB_BINARY_DIR}")

##==============================================================================
include_directories(${DVO_INC_DIRS})
link_directories(${DVO_LIB_DIRS})
set(DVO_INC_DIRS_PARENT_SCOPE ${DVO_INC_DIRS} PARENT_SCOPE)
set(DVO_LIB_DIRS_PARENT_SCOPE ${DVO_LIB_DIRS} PARENT_SCOPE)
set(sub_dirs . core util)

set(REG_EXT "[^/]*([.]cpp|[.]c|[.]cc|[.]h|[.]hpp)$")
foreach(sub_dir ${sub_dirs})
  file(GLOB dvo_${sub_dir}_sources "${CMAKE_CURRENT_SOURCE_DIR}/dvo/${sub_dir}/*.cc")
  file(GLOB dvo_${sub_dir}_headers "${CMAKE_CURRENT_SOURCE_DIR}/dvo/${sub_dir}/*.h")
  list(APPEND dvo_sources ${dvo_${sub_dir}_sources}
                            ${dvo_${sub_dir}_headers})
  source_group("${sub_dir}" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/${REG_EXT}")
endforeach()

## The library =================================================================
set(ALL_LIBRARIES ${DVO_EXTERNAL_LIBS})
if(SM_USE_COTIRE)
  cmaker_add_library_cotire(dvo ${SM_LIB_TYPE} ${dvo_sources})
else()
  cmaker_add_library(dvo ${SM_LIB_TYPE} ${dvo_sources})
endif()
set_property(TARGET dvo PROPERTY FOLDER "external/dvo")

set(ALL_LIBRARIES dvo ${DVO_EXTERNAL_LIBS})
set(DVO_ALL_LIBRARIES_PARENT_SCOPE ${ALL_LIBRARIES} PARENT_SCOPE)

## Tests
set(tests test_tracking_dense_matcher
          test_tracking_dense_matcher_stream
          #test_tracking_dense_matcher_live
          )
foreach(test ${tests})
  add_executable(${test} dvo/test/${test}.cc)
  target_link_libraries(${test} dvo  ${DVO_EXTERNAL_LIBS})
  set_property(TARGET ${test} PROPERTY FOLDER "external/dvo/tests")
  if(SM_USE_COTIRE)
    set_target_properties(${test} PROPERTIES COTIRE_UNITY_LINK_LIBRARIES_INIT "COPY")
    cotire(${test})
  endif()
endforeach()

# Useful Windows paths settings ================================================
# Useful paths configuration for windows 
if(WIN32)
  if(MSVC)
    if(CMAKE_CL_64)
      set(WIN32_ARCH x64)
    else()
      set(WIN32_ARCH x86)
    endif()
    if(MSVC_VERSION EQUAL 1400)
      set(StructuralModeling_RUNTIME vc8)
    elseif(MSVC_VERSION EQUAL 1500)
      set(StructuralModeling_RUNTIME vc9)
    elseif(MSVC_VERSION EQUAL 1600)
      set(StructuralModeling_RUNTIME vc10)
      get_filename_component(VC_IDE_PATH $ENV{VS100COMNTOOLS}/../IDE ABSOLUTE)
    elseif(MSVC_VERSION EQUAL 1700)
      set(StructuralModeling_RUNTIME vc11)
      get_filename_component(VC_IDE_PATH $ENV{VS110COMNTOOLS}/../IDE ABSOLUTE)
    elseif(MSVC_VERSION EQUAL 1800)
      set(StructuralModeling_RUNTIME vc12)
      get_filename_component(VC_IDE_PATH $ENV{VS120COMNTOOLS}/../IDE ABSOLUTE)
    endif()
    get_filename_component(VC_PATH "${VC_IDE_PATH}/../../VC" ABSOLUTE)
  endif()

  cmaker_print_status("THE_DEPEDENCIES_BINARY_PATHS:${THE_DEPEDENCIES_BINARY_PATHS}")
  list(APPEND THE_DEPEDENCIES_BINARY_PATHS "${VC_IDE_PATH}/../../VC" 
"${VC_IDE_PATH}/../../VC/bin")
  configure_file("${CMAKER_ROOT}/scripts/set_paths.bat.in" 
                 "${CMAKE_CURRENT_BINARY_DIR}/set_paths.bat")
  configure_file("${CMAKER_ROOT}/scripts/set_paths.bat.in" 
                 "${THE_LIB_RUNTIME_OUTPUT_DIRECTORY}/set_paths.bat")
  configure_file("${CMAKER_ROOT}/scripts/set_paths_and_run_vc.bat.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/set_paths_and_run_vc.bat")
  set(DVO_THE_DEPEDENCIES_BINARY_PATHS_PARENT_SCOPE ${THE_DEPEDENCIES_BINARY_PATHS} PARENT_SCOPE)
endif()

# useful configuration (e.x.: put scripts in right places) =====================
if(WIN32)
  configure_file("${CMAKER_ROOT}/scripts/run_ninja_build.bat.in" 
                 "${CMAKE_CURRENT_BINARY_DIR}/run_ninja_build.bat")
  if(NOT EXISTS "${THE_LIB_RUNTIME_OUTPUT_DIRECTORY}/run_windbg.bat")
    configure_file("${CMAKER_ROOT}/scripts/run_windbg.bat.in" 
                 "${THE_LIB_RUNTIME_OUTPUT_DIRECTORY}/run_windbg.bat")
  endif()
  set(THE_BUILD_CMD "run_ninja_build.bat")
  set(THE_FILE_REGEX "^(.*)\\\\(([0-9]*)\\\\)(\\\\s:\\\\s).*error(.*)")
else()
  # configure it to a temp location
  configure_file("${CMAKER_ROOT}/scripts/run_make_build.sh.in" 
                 "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/run_make_build.sh")
  # now copy the temporary into the final destination, setting the permissions
  file(COPY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/run_make_build.sh"
          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
          FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
          GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  set(THE_BUILD_CMD "run_make_build.sh")
  set(THE_FILE_REGEX "^(.*):([0-9]+):([0-9]+).*error(.*)")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/DVO.sublime-project.in"
               "${CMAKE_CURRENT_BINARY_DIR}/DVO.sublime-project" @ONLY)

