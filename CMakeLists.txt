cmake_minimum_required(VERSION 2.8)

project(dvo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
set(DVO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DDVO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

## Dependencies ===============================================================
# Sophus
find_path(Sophus_INCLUDE_DIR NAMES sophus/sophus.hpp
            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../sophus)
include_directories(${Sophus_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
get_filename_component(OpenCV_BINARY_DIR "${OpenCV_LIB_DIR}/../bin" ABSOLUTE)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND DVO_EXTERNAL_LIBS ${OpenCV_LIBRARIES})

##=============================================================================
set(sub_dirs . core util)

set(REG_EXT "[^/]*([.]cpp|[.]c|[.]cc|[.]h|[.]hpp)$")
foreach(sub_dir ${sub_dirs})
  file(GLOB dvo_${sub_dir}_sources "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/*.cc")
  file(GLOB dvo_${sub_dir}_headers "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/*.h")
  list(APPEND dvo_sources ${dvo_${sub_dir}_sources}
                            ${dvo_${sub_dir}_headers})
  source_group("${sub_dir}" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/${REG_EXT}")
endforeach()

## The library ================================================================
set(ALL_LIBRARIES ${DVO_EXTERNAL_LIBS} ${SM_EXTERNAL_LIBS})
if(SM_USE_COTIRE)
  cmaker_add_library_cotire(dvo ${SM_LIB_TYPE} ${dvo_sources})
else()
  cmaker_add_library(dvo ${SM_LIB_TYPE} ${dvo_sources})
endif()
set_property(TARGET dvo PROPERTY FOLDER "external/dvo")

set(ALL_LIBRARIES dvo ${DVO_EXTERNAL_LIBS} )
## Tests
set(tests test_tracking_dense_matcher test_tracking_dense_matcher_stream
          test_tracking_dense_matcher_live)
foreach(test ${tests})
  add_executable(${test} test/${test}.cc)
  target_link_libraries(${test} dvo  ${DVO_EXTERNAL_LIBS})
  set_property(TARGET ${test} PROPERTY FOLDER "external/dvo/tests")
  if(SM_USE_COTIRE)
    set_target_properties(${test} PROPERTIES COTIRE_UNITY_LINK_LIBRARIES_INIT "COPY")
    cotire(${test})
  endif()
endforeach()