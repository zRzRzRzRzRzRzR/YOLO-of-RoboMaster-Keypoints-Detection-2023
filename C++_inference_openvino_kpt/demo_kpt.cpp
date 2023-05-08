#include "yolov7_kpt.h"
#define VIDEO_PATH "red-win.MP4"
yolo_kpt DEMO;
std::vector<yolo_kpt::Object> result;
cv::TickMeter meter;
int main() {
    cv::VideoCapture cap;
    cap.open(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cout << "相机没有打开或没有找到视频" << std::endl;
        return 0;
    }
    while (true) {
        cv::Mat src_img;
        bool ret = cap.read(src_img);
        if (!ret) break;
        meter.start();
        result = DEMO.work(src_img);
        meter.stop();
        printf("Time: %f\n", meter.getTimeMilli());
        meter.reset();
    }
    cv::destroyAllWindows();
    return 0;
}