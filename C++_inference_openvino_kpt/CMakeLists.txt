cmake_minimum_required(VERSION 3.16)
project(openvino_kpt_class)

set(CMAKE_CXX_STANDARD 11)
set(ngraph_DIR "/opt/intel/openvino_2022/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2022/runtime/cmake")

find_package(OpenCV 4 REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        #OpenVINO推理引擎的头文件
        /opt/intel/openvino_2022/runtime/include/ie/
        /opt/intel/openvino_2022/runtime/include/ngraph/
        /opt/intel/openvino_2022/runtime/include/openvino/

)
include_directories(
        ${OpenCV_LIBRARY_DIRS}
)
add_executable(openvino_kpt_class ${sourcefile} yolov7_kpt.cpp yolov7_kpt.h demo_kpt.cpp)
target_link_libraries(
        openvino_kpt_class
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
)



