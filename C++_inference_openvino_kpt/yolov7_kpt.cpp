#include "yolov7_kpt.h"

yolo_kpt::yolo_kpt() {
    model = core.read_model(MODEL_PATH);
    std::shared_ptr<ov::Model> model = core.read_model(MODEL_PATH);
    compiled_model = core.compile_model(model, DEVICE);
    auto input_port = compiled_model.input();
    infer_request = compiled_model.create_infer_request();
    input_tensor1 = infer_request.get_input_tensor(0);
}

cv::Mat yolo_kpt::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd) {
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;
    resize(src, resize_img, cv::Size(inside_w, inside_h));
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padd.push_back(padd_w);
    padd.push_back(padd_h);
    padd.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

cv::Rect yolo_kpt::scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h) {
    cv::Rect scaled_box;
    scaled_box.width = box.width / padd[2];
    scaled_box.height = box.height / padd[2];
    scaled_box.x = std::max(std::min((float) ((box.x - padd[0]) / padd[2]), (float) (raw_w - 1)), 0.f);
    scaled_box.y = std::max(std::min((float) ((box.y - padd[1]) / padd[2]), (float) (raw_h - 1)), 0.f);
    return scaled_box;
}

std::vector<cv::Point2f>
yolo_kpt::scale_box_kpt(std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx) {
    std::vector<cv::Point2f> scaled_points;
    for (int ii = 0; ii < 4; ii++) {
        points[idx * 4 + ii].x = std::max(
                std::min((points[idx * 4 + ii].x - padd[0]) / padd[2], (float) (raw_w - 1)), 0.f);
        points[idx * 4 + ii].y = std::max(
                std::min((points[idx * 4 + ii].y - padd[1]) / padd[2], (float) (raw_h - 1)), 0.f);
        scaled_points.push_back(points[idx * 4 + ii]);

    }
    return scaled_points;
}

void yolo_kpt::drawPred(int classId, float conf, cv::Rect box, std::vector<cv::Point2f> point, cv::Mat &frame,
                        const std::vector<std::string> &classes) { //画图部分
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;
    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    for (int i = 0; i < KPT_NUM; i++)
        cv::circle(frame, point[i], 2, cv::Scalar(255, 0, 0), 2);
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
    y0 = std::max(int(y0), labelSize.height);
    cv::rectangle(frame, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
                  cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
}

void yolo_kpt::generate_proposals(int stride, const float *feat, std::vector<Object> &objects) { //后处理部分，重做检测层
    int feat_w = IMG_SIZE / stride;
    int feat_h = IMG_SIZE / stride;
#if IMG_SIZE == 640
    float anchors[18] = {11, 10, 19, 15,28, 22, 39, 34, 64, 48, 92, 76, 132, 110, 197, 119,  265, 162}; // 6 4 0
#elif IMG_SIZE == 416
    float anchors[18] = {12, 10, 19, 14, 26, 20, 37, 31, 61, 46, 88, 68, 113, 105, 193, 108, 245, 149}; // 4 1 6
#elif IMG_SIZE == 320
    float anchors[18] = {6,5, 9,7, 13, 9, 18, 15, 30, 23, 46, 37, 60, 52, 94, 56, 125, 72}; // 3 2 0
#endif
    int anchor_group = 0;
    if (stride == 8)
        anchor_group = 0;
    if (stride == 16)
        anchor_group = 1;
    if (stride == 32)
        anchor_group = 2;

    for (int anchor = 0; anchor < ANCHOR; anchor++) { //对每个anchor进行便利
        for (int i = 0; i < feat_h; i++) { // self.grid[i][..., 0:1]
            for (int j = 0; j < feat_w; j++) { // self.grid[i][..., 1:2]
                //每个tensor包含的数据是[x,y,w,h,conf,cls1pro,cls2pro,...clsnpro,kpt1.x,kpt1.y,kpt1.conf,kpt2...kptm.conf]
                //一共的长度应该是 (5 + CLS_NUM + KPT_NUM * 3)
                float box_prob = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                      i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                      j * (5 + CLS_NUM + KPT_NUM * 3) + 4];
                box_prob = sigmoid(box_prob);
                if (box_prob < CONF_THRESHOLD) continue; // 删除置信度低的bbox

                float kptx[5], kpty[5], kptp[5];
                // xi,yi,pi 是每个关键点的xy坐标和置信度,最新的代码用不到pi,但是用户可以根据自己需求添加
                float x = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 0];
                float y = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 1];
                float w = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 2];
                float h = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 3];
                if (KPT_NUM != 0)
                    for (int k = 0; k < KPT_NUM; k++) {
                        kptx[k] = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       j * (5 + CLS_NUM + KPT_NUM * 3) + 5 + CLS_NUM + k * 3];
                        kpty[k] = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       j * (5 + CLS_NUM + KPT_NUM * 3) + 5 + CLS_NUM + k * 3 + 1];
                        kptp[k] = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       j * (5 + CLS_NUM + KPT_NUM * 3) + 5 + CLS_NUM + k * 3 + 2];

                        //对关键点进行后处理(python 代码)
                        //x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)) * self.stride[i]  # xy
                        //x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)) * self.stride[i]  # xy
                        kptx[k] = (kptx[k] * 2 - 0.5 + j) * stride;
                        kpty[k] = (kpty[k] * 2 - 0.5 + i) * stride;
                    }
                double max_prob = 0;
                int idx = 0;
                for (int k = 5; k < CLS_NUM + 5; k++) {
                    double tp = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                     i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                     j * (5 + CLS_NUM + KPT_NUM * 3) + k];
                    tp = sigmoid(tp);
                    if (tp > max_prob)
                        max_prob = tp, idx = k;
                } //选取最高置信度的类别
                float cof = std::min(box_prob * max_prob, 1.0);
//                float cof = std::min(box_prob * max_prob + CONF_REMAIN, 1.0); // 添加上一帧的保留权重
                if (cof < CONF_THRESHOLD) continue; // 在判断是否符合任何一个类别
                last_conf = cof;

                //xywh的后处理(python 代码)
                //xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                //wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                x = (sigmoid(x) * 2 - 0.5 + j) * stride;
                y = (sigmoid(y) * 2 - 0.5 + i) * stride;
                w = pow(sigmoid(w) * 2, 2) * anchors[anchor_group * 6 + anchor * 2];
                h = pow(sigmoid(h) * 2, 2) * anchors[anchor_group * 6 + anchor * 2 + 1];
                //将中心点变为左上点，转换为OpenCV rect类型
                float r_x = x - w / 2;
                float r_y = y - h / 2;
                Object obj;
                obj.rect.x = r_x;
                obj.rect.y = r_y;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = idx - 5;
                obj.prob = cof;
                if (KPT_NUM != 0)
                    for (int k = 0; k < KPT_NUM; k++)
                        obj.kpt.push_back(cv::Point2f(kptx[k], kpty[k]));
                objects.push_back(obj);
            }
        }
    }
}

std::vector<yolo_kpt::Object> yolo_kpt::work(cv::Mat src_img) {
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;
    cv::Mat img;
    std::vector<float> padd;
    cv::Mat boxed = letter_box(src_img, img_h, img_w, padd);
    cv::cvtColor(boxed, img, cv::COLOR_BGR2RGB);
    auto data1 = input_tensor1.data<float>();
    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                int out_index = c * img_h * img_w + h * img_w + w;
                data1[out_index] = float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    infer_request.infer(); //推理并获得三个提取头
    auto output_tensor_p8 = infer_request.get_output_tensor(0);
    const float *result_p8 = output_tensor_p8.data<const float>();
    auto output_tensor_p16 = infer_request.get_output_tensor(1);
    const float *result_p16 = output_tensor_p16.data<const float>();
    auto output_tensor_p32 = infer_request.get_output_tensor(2);
    const float *result_p32 = output_tensor_p32.data<const float>();
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    std::vector<Object> objects16;
    std::vector<Object> objects32;
    generate_proposals(8, result_p8, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    generate_proposals(16, result_p16, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals(32, result_p32, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < proposals.size(); i++) {
        classIds.push_back(proposals[i].label);
        confidences.push_back(proposals[i].prob);
        boxes.push_back(proposals[i].rect);
        for (auto ii: proposals[i].kpt)
            points.push_back(ii);
    }
    std::vector<int> picked;
    std::vector<float> picked_useless; //SoftNMS
    std::vector<Object> object_result;

    //SoftNMS 要求OpenCV>=4.6.0
//    cv::dnn::softNMSBoxes(boxes, confidences, picked_useless, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    for (size_t i = 0; i < picked.size(); i++) {
        cv::Rect scaled_box = scale_box(boxes[picked[i]], padd, src_img.cols, src_img.rows);
        std::vector<cv::Point2f> scaled_point;
        if (KPT_NUM != 0)
            scaled_point = scale_box_kpt(points, padd, src_img.cols, src_img.rows, picked[i]);
        Object obj;
        obj.rect = scaled_box;
        obj.label = classIds[picked[i]];
        obj.prob = confidences[picked[i]];
        if (KPT_NUM != 0)
            obj.kpt = scaled_point;
        object_result.push_back(obj);

#ifdef VIDEO
        drawPred(classIds[picked[i]], confidences[picked[i]], scaled_box, scaled_point, src_img,
                 class_names);
#endif
    }
#ifdef VIDEO
    cv::imshow("Inference frame", src_img);
    cv::waitKey(1);
#endif
    return object_result;
}
