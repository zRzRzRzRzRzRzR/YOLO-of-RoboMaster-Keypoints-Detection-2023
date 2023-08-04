#include "openvino_detect.h"
#define VIDEO_PATH "/home/zr/4.avi"
yolo_detct DEMO;
std::vector<yolo_detct::Object> result;
cv::TickMeter meter;
int main() {
    cv::VideoCapture cap;
    cap.open(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cout << "NO Videos" << std::endl;
        return 0;
    }
    while (true) {
        cv::Mat src_img;
        bool ret = cap.read(src_img);
        if (!ret) break;
        meter.start();
        result = DEMO.work(src_img);
        for (auto i: result) {
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox:" << i.rect << std::endl;
        }
        meter.stop();
        printf("Time: %f\n", meter.getTimeMilli());
        meter.reset();
    }
    cv::destroyAllWindows();
    return 0;
}