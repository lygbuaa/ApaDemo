CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(ApaDemo)

ADD_DEFINITIONS(-std=c++11)
SET(CMAKE_BUILD_TYPE "Debug")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC ") #-fsingle-precision-constant
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0 -fPIC -rdynamic -export-dynamic")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")

SET(SRC_LIST
src/main.cc
)
SET(INCLUDE_LIST
    ${PROJECT_SOURCE_DIR}/inc/
    # ${PROJECT_SOURCE_DIR}/ort/onnxruntime-linux-aarch64-1.13.1/include
    ${PROJECT_SOURCE_DIR}/ort/onnxruntime-linux-x64-1.13.1/include
)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# set(ONNXRUNTIME_LIB ${CMAKE_SOURCE_DIR}/ort/onnxruntime-linux-aarch64-1.13.1/lib/libonnxruntime.so)
set(ONNXRUNTIME_LIB ${CMAKE_SOURCE_DIR}/ort/onnxruntime-linux-x64-1.13.1/lib/libonnxruntime.so)

find_package(OpenCV REQUIRED)

MESSAGE(STATUS "\nThis is PROJECT name: " ${PROJECT_NAME})
MESSAGE(STATUS "This is PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
MESSAGE(STATUS "This is BINARY dir: " ${PROJECT_BINARY_DIR})
MESSAGE(STATUS "This is exe dir: " ${EXECUTABLE_OUTPUT_PATH})
MESSAGE(STATUS "This is CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "This is CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "This is SRC_LIST: " ${SRC_LIST})
MESSAGE(STATUS "This is INCLUDE_LIST: " ${INCLUDE_LIST})
MESSAGE(STATUS "This is opencv inlcude: " ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "This is opencv lib: " ${OpenCV_LIBS})
MESSAGE(STATUS "This is onnxruntime: " ${ONNXRUNTIME_LIB})

ADD_EXECUTABLE(ApaDemo ${SRC_LIST})

INCLUDE_DIRECTORIES(
    ${INCLUDE_LIST}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    )

link_directories(
    /usr/local/lib
    ${LIBRARY_OUTPUT_PATH}
)

TARGET_LINK_LIBRARIES(ApaDemo
-ldl
-lpthread
${OpenCV_LIBS}
${ONNXRUNTIME_LIB}
)
