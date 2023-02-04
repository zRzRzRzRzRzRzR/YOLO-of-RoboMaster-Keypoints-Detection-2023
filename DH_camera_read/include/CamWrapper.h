//
// Created by whoismz on 1/2/22.
//
#ifndef GMASTER_WM_NEW_CAMWRAPPER_H
#define GMASTER_WM_NEW_CAMWRAPPER_H
#include <opencv2/opencv.hpp>
class Camera {
public:
    virtual bool init(int roi_x, int roi_y, int roi_w, int roi_h,
                      float exposure, float gain, bool isEnergy) = 0;

    virtual void setParam(float exposure, float gain) = 0;

    virtual bool start() = 0;

    virtual void stop() = 0;

    virtual bool init_is_successful() = 0;

    virtual bool read(cv::Mat &src) = 0;
};

#endif //GMASTER_WM_NEW_CAMWRAPPER_H
