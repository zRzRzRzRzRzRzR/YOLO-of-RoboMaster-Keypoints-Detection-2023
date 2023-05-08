cmake_minimum_required(VERSION 3.0.2)
project(DH_camera_read)

add_compile_options(-std=c++17)


set(CMAKE_EXE_LINKER_FLAGS
        -I$(GENICAM_ROOT_V2_3)/library/CPP/include
        -I$(DAHENG_ROOT)/sdk/include
        )

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

add_executable(
        DH_camera_read
        src/CamWrapper.cpp
        include/CamWrapper.h
        include/CamWrapperDH.h
        include/DxImageProc.h
        include/GxIAPI.h
        src/DH_CAM.cpp include/DH_CAM.h)

target_link_libraries(
        DH_camera_read
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        -lglog
        -lgxiapi
)


