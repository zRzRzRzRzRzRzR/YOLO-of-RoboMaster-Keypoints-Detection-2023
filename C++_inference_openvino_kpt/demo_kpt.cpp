#include "yolov7_kpt.h"
//#define VIDEO_PATH "/home/zr/Downloads/test_for_net.mp4"
#define VIDEO_PATH "/home/zr/Downloads/cut3.avi" //视频测试位置
yolo_kpt DEMO; //实例化
std::vector<yolo_kpt::Object> result; //储存结果
cv::TickMeter meter;
float last_conf = 0.0; //存储上一帧的置信度，在自己的工程中需要移植。
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
        for (auto i: result) {
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox:" << i.rect << std::endl;
            if (KPT_NUM != 0) {
                std::cout << "kpt:";
                for (auto kpt: i.kpt)
                    std::cout << kpt << " ";
                std::cout << std::endl;
            }
        }
    }
    cv::destroyAllWindows();
    return 0;
}