#include "DH_CAM.h"
inline static std::string getCurrentTime() {
    std::time_t result = std::time(nullptr);
    std::string ret;
    ret.resize(64);
    int wsize = sprintf((char *)&ret[0], "%s", std::ctime(&result));
    ret.resize(wsize);
    return ret;
}
void radar()
{

    VideoCapture cap1(0);
    VideoCapture cap2(0);
    string path_left = REORD_PATH + getCurrentTime() + "__left.avi";
    string path_right = REORD_PATH + getCurrentTime() + "__right.avi";
    cv::VideoWriter writer_left(path_left, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1280, 1024));
    cv::VideoWriter writer_right(path_right, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1280, 1024));
    camera_left = new DHCamera(LEFT_SN);
    camera_right = new DHCamera(RIGHT_SN);
    camera_left->init(0, 0, 1280, 1024, 20000, 10, false);
    camera_right->init(0, 0, 1280, 1024, 20000, 10, false);
    while (waitKey(1)!=27) {
        if((!camera_left->start())||(!camera_right->start()))
        {
            cout<<"NO Camera or Just one,please check"<<endl;
            return ;
        }
        camera_left->read(img_src1);
        camera_right->read(img_src2);
        if (img_src1.empty() || img_src2.empty()) {
            cout << "IMG IS EMPTY" << endl;
            return ;
        }
#ifdef VIDEO
        imshow("LEFT", img_src1);
        imshow("RIGHT", img_src2);
        waitKey(1);
#endif
#ifdef RECORD
        writer_left << img_src1;
        writer_right << img_src2;
#endif
    }
    cap1.release();
    cap2.release();
    return ;
}
void car()
{

    VideoCapture cap_car(0);
    string path_car = REORD_PATH + getCurrentTime() + "__car.avi";
    cv::VideoWriter writer_car(path_car, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1280, 1024));
    camera_car = new DHCamera(CAR_SN);
    camera_car->init(0, 0, 640, 384, 10000, 10, false);
    while (waitKey(1)!=27) {
        if(!camera_car->start())
        {
            cout<<"NO Camera or Just one,please check"<<endl;
            return ;
        }
        camera_left->read(img_src1);
        camera_right->read(img_src2);
        if (img_src1.empty() || img_src2.empty()) {
            cout << "IMG IS EMPTY" << endl;
            return ;
        }
#ifdef VIDEO
        imshow("dst_car", img_src1);
        waitKey(1);
#endif
#ifdef RECORD
        writer_car << img_src1;
#endif
    }
    cap_car.release();
    return ;
}
int main(int argc,char **argv)
{
#if RADAR_MODE
    radar();
#elif 
//    car();
#endif
    return 0;
}