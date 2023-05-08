#ifndef CAMERA_DH_CAM_H
#define CAMERA_DH_CAM_H

#include"CamWrapper.h"
#include"CamWrapperDH.h"
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Mat img_src1;
Mat img_src2;
Camera *camera_left = nullptr;
Camera *camera_right = nullptr;
Camera *camera_car = nullptr;

#define REORD_PATH  "/home/zr/record/"
#define RADAR_MODE true
//#define VIDEO //是否展示视频
#define RECORD //是否保存视频
#define LEFT_SN "FGV22100003"
#define RIGHT_SN "FGV22100004"
#define CAR_SN "FGV22100004"
inline static std::string getCurrentTime();
void radar();
void car();
#endif //CAMERA_DH_CAM_H